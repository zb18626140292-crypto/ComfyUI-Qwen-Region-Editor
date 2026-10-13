"""Crop with context for native Qwen editing, then strictly composite inside the ROI."""
import math
import uuid
import logging

import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

ALIGNMENT_MODES = ["关闭", "自动（主体特征）", "自动（框外上下文）", "手动"]


def _estimate_alignment(source, edited, local_box, subject=True):
    """Estimate edited->source similarity from matched image features.

    ROI features are useful when the model moves only the face inside the ROI.
    Context mode explicitly excludes the entire ROI and its guide/feather band.
    Generated additions have no source correspondence and are rejected by ratio
    matching and RANSAC. A failed/ambiguous estimate must remain an identity.
    """
    identity = np.eye(3, dtype=np.float64)
    try:
        import cv2
    except ImportError:
        return identity, "自动对齐未应用：此环境没有 OpenCV；可切换手动。"
    sh, sw = source.shape[1:3]
    if min(sh, sw) < 32:
        return identity, "自动对齐未应用：裁剪太小，缺少可匹配纹理。"
    factor = min(1.0, 1024 / max(sh, sw))
    dw, dh = max(1, round(sw * factor)), max(1, round(sh * factor))
    arrays = []
    for tensor in (source, edited):
        rgb = (tensor[0, :, :, :3].detach().float().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        if (dw, dh) != (sw, sh):
            rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        arrays.append(rgb)
    rgb_source, rgb_edit = arrays
    gray_source, gray_edit = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in arrays]
    if min(float(gray_source.std()), float(gray_edit.std())) < 2.0:
        return identity, "自动对齐未应用：图像纹理不足；可切换手动。"
    x0, y0, x1, y1 = local_box
    rect = (math.floor(x0 * dw/sw), math.floor(y0 * dh/sh),
            math.ceil(x1 * dw/sw), math.ceil(y1 * dh/sh))

    def match_area(use_subject):
        allowed = np.zeros((dh, dw), dtype=np.uint8) if use_subject else np.full((dh, dw), 255, np.uint8)
        ax, ay, bx, by = rect
        if use_subject:
            margin = max(2, round(8 * factor))
            allowed[min(dh, ay+margin):max(ay+margin, by-margin),
                    min(dw, ax+margin):max(ax+margin, bx-margin)] = 255
        else:
            # Exclude the editable rectangle even where its feather mask is zero.
            margin = max(4, round(16 * factor))
            allowed[max(0, ay-margin):min(dh, by+margin),
                    max(0, ax-margin):min(dw, bx+margin)] = 0
        masks = []
        for rgb in arrays:
            # A position guide is artificial and must not become an anchor.
            magenta = ((rgb[..., 0] > 180) & (rgb[..., 2] > 180) & (rgb[..., 1] < 110)).astype(np.uint8)
            marker = cv2.dilate(magenta, np.ones((9, 9), np.uint8))
            masks.append(np.where(marker, 0, allowed).astype(np.uint8))
        if min(np.count_nonzero(m) for m in masks) < 256:
            return None, "可匹配范围不足"
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        ks, ds = sift.detectAndCompute(gray_source, masks[0])
        ke, de = sift.detectAndCompute(gray_edit, masks[1])
        if ds is None or de is None or min(len(ks), len(ke)) < 6:
            return None, "有效特征不足"
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forwards = matcher.knnMatch(de, ds, k=2)
        backwards = matcher.knnMatch(ds, de, k=2)
        reverse = {m.queryIdx: m.trainIdx for pair in backwards if len(pair) == 2
                   for m, n in [pair] if m.distance < 0.8*n.distance}
        good = [m for pair in forwards if len(pair) == 2 for m, n in [pair]
                if m.distance < 0.72*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
        if len(good) < 6:
            return None, "一致匹配不足"
        from_edit = np.float32([ke[m.queryIdx].pt for m in good])
        to_source = np.float32([ks[m.trainIdx].pt for m in good])
        affine, inliers = cv2.estimateAffinePartial2D(
            from_edit, to_source, method=cv2.RANSAC, ransacReprojThreshold=2.5,
            maxIters=3000, confidence=0.995, refineIters=10)
        if affine is None or inliers is None:
            return None, "无法估计可靠变换"
        accepted = inliers[:, 0].astype(bool)
        count = int(accepted.sum())
        if count < 6 or count / len(good) < 0.4:
            return None, "匹配置信不足"
        points = to_source[accepted]
        required_w = (x1-x0) if use_subject else sw
        required_h = (y1-y0) if use_subject else sh
        span = np.ptp(points, axis=0)
        if span[0] < max(8, 0.15*required_w*dw/sw) or span[1] < max(8, 0.15*required_h*dh/sh):
            return None, "特征过于集中，无法可靠判断缩放"
        matrix = identity.copy()
        matrix[:2] = affine
        downscale = np.diag([dw/sw, dh/sh, 1.0])
        matrix = np.linalg.inv(downscale) @ matrix @ downscale
        if not np.isfinite(matrix).all():
            return None, "变换数值无效"
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        center = np.array([sw/2, sh/2, 1.0])
        displacement = matrix @ center - center
        limit = min(128, max(16, 0.15*min(sw, sh)))
        if not (0.85 <= scale <= 1.15) or abs(rotation) > 5 or np.linalg.norm(displacement[:2]) > limit:
            return None, "位移、缩放或旋转超过保守范围"
        # Prevent spurious resampling for an unchanged crop.
        if np.linalg.norm(displacement[:2]) < 0.05 and abs(scale-1) < 0.0001 and abs(rotation) < 0.005:
            matrix = identity.copy()
        if not np.array_equal(matrix, identity):
            # RANSAC can still agree on repeated texture. Require an independent
            # improvement on shared image texture before applying its transform.
            texture_source = gray_source.astype(np.float32)
            texture_edit = gray_edit.astype(np.float32)
            texture_source -= cv2.GaussianBlur(texture_source, (0, 0), 2.0)
            texture_edit -= cv2.GaussianBlur(texture_edit, (0, 0), 2.0)
            warped_texture = cv2.warpAffine(texture_edit, affine, (dw, dh), flags=cv2.INTER_LINEAR)
            warped_mask = cv2.warpAffine(masks[1], affine, (dw, dh), flags=cv2.INTER_NEAREST)
            shared = (masks[0] > 0) & (masks[1] > 0) & (warped_mask > 0)
            if int(shared.sum()) < 256:
                return None, "变换后有效纹理不足"

            def correlation(first, second):
                a, b = first[shared].astype(np.float64), second[shared].astype(np.float64)
                a, b = a-a.mean(), b-b.mean()
                denominator = np.linalg.norm(a)*np.linalg.norm(b)
                return float(np.dot(a, b)/denominator) if denominator > 1e-6 else -1.0

            before = correlation(texture_source, texture_edit)
            after = correlation(texture_source, warped_texture)
            if after < 0.25 or after-before < 0.01:
                return None, "变换未可靠改善纹理匹配（可能是重复纹理或非刚性变化）"
        label = "主体特征" if use_subject else "框外上下文"
        detail = f"自动对齐（{label}）：匹配 {count}/{len(good)}，中心位移 x {displacement[0]:+.2f}px、y {displacement[1]:+.2f}px，缩放 {scale:.4f}，旋转 {rotation:+.2f}°"
        return matrix, detail

    try:
        matrix, detail = match_area(subject)
        if matrix is None and subject:
            reason = detail
            matrix, detail = match_area(False)
            if matrix is not None:
                detail += f"（主体{reason}，使用上下文）"
        if matrix is None:
            return identity, f"自动对齐未应用：{detail}；保持原坐标，可切换手动。"
        return matrix, detail
    except cv2.error as error:
        logging.warning("[Qwen Region] OpenCV alignment failed: %s", error)
        return identity, "自动对齐未应用：特征估计失败；保持原坐标，可切换手动。"


def _aligned_patch(edit, local_box, matrix):
    """Sample only the target ROI; invalid source coverage never adds black pixels."""
    x0, y0, x1, y1 = local_box
    if np.array_equal(matrix, np.eye(3)):
        return edit[:, y0:y1, x0:x1], edit.new_ones((1, y1-y0, x1-x0, 1))
    inverse = np.linalg.inv(matrix)
    h, w = edit.shape[1:3]
    ys, xs = torch.meshgrid(torch.arange(y0, y1, device=edit.device, dtype=torch.float32),
                            torch.arange(x0, x1, device=edit.device, dtype=torch.float32), indexing="ij")
    source_x = xs*float(inverse[0, 0]) + ys*float(inverse[0, 1]) + float(inverse[0, 2])
    source_y = xs*float(inverse[1, 0]) + ys*float(inverse[1, 1]) + float(inverse[1, 2])
    valid = ((source_x >= 0) & (source_x <= w-1) & (source_y >= 0) & (source_y <= h-1))
    grid = torch.stack((2*source_x/max(1, w-1)-1, 2*source_y/max(1, h-1)-1), dim=-1)[None]
    patch = F.grid_sample(edit.float().movedim(-1, 1), grid,
                          mode="bilinear", padding_mode="zeros", align_corners=True).movedim(1, -1).to(edit.dtype)
    return patch, valid[None, :, :, None].to(edit.dtype)


def _skin_samples(rgb):
    r, g, b = rgb.unbind(-1)
    high, low = rgb.amax(-1), rgb.amin(-1)
    saturation = (high-low) / high.clamp_min(1e-6)
    light = 0.2126*r + 0.7152*g + 0.0722*b
    return (torch.isfinite(rgb).all(-1) & (light > 0.14) & (light < 0.92)
            & (r >= g-0.035) & (r >= b+0.008) & (g >= b-0.025)
            & (high-low > 0.008) & (saturation < 0.58))


def _correct_skin_color(source, edited, support, valid, local_box, strength, sample_mask=None):
    h, w = source.shape[1:3]
    stride = max(1, math.ceil(max(h, w) / 768))
    original = source[:, ::stride, ::stride, :3].float()
    generated = edited[:, ::stride, ::stride, :3].float()
    differences = original-generated
    eligible = (valid[:, ::stride, ::stride, 0] > 0)
    eligible &= torch.isfinite(original).all(-1) & torch.isfinite(generated).all(-1)
    eligible &= differences.abs().amax(-1) < 0.12
    if sample_mask is not None:
        selected = eligible & (sample_mask[:, ::stride, ::stride] > 0.5)
        location = "指定蒙版"
    else:
        eligible &= _skin_samples(original) & _skin_samples(generated)
        selected = eligible & support[:, ::stride, ::stride]
        location = "框内肤色"
        if int(torch.count_nonzero(selected)) < 64:
            x0, y0, x1, y1 = local_box
            ring = torch.zeros_like(support)
            ring[:, max(0, y0-32):min(h, y1+32), max(0, x0-32):min(w, x1+32)] = True
            selected = eligible & (ring & ~support)[:, ::stride, ::stride]
            location = "周边保留区"
    count = int(torch.count_nonzero(selected))
    info = {"sample_area": location, "samples": count, "applied": False, "offset_rgb": [0.0, 0.0, 0.0]}
    if count < 64:
        return edited, "肤色取样不足，跳过颜色校正。", info
    values = differences[selected]
    shift = torch.quantile(values, 0.5, dim=0)
    deviation = torch.quantile((values-shift).abs(), 0.5, dim=0)
    if float(deviation.max()) > 0.025:
        return edited, "取样色差不一致，跳过颜色校正。", info
    # Offset only: retain contrast and linework; large changes are not treated as a color cast.
    offset = shift.clamp(-0.04, 0.04) * strength
    x0, y0, x1, y1 = local_box
    result = edited.clone()
    result[:, y0:y1, x0:x1, :3] = (edited[:, y0:y1, x0:x1, :3].float()+offset).clamp(0, 1).to(edited.dtype)
    info.update(applied=True, offset_rgb=offset.detach().cpu().tolist())
    maximum = float(offset.abs().max()) * 255
    return result, f"{location}取样 {count} 点，轻量校正最大 {maximum:.2f}/255。", info


def _micro_feather(alpha, pixels):
    if pixels == 0:
        return alpha
    support = (alpha[..., 0] > 0).float()[:, None]
    current = support
    ramp = support.clone()
    for _ in range(pixels):
        current = 1-F.max_pool2d(F.pad(1-current, (1, 1, 1, 1), value=1), 3, stride=1)
        ramp += current
    # One pixel gives a half-weight boundary, with full strength immediately inside.
    return alpha * (ramp[:, 0, :, :, None] / (pixels+1)).to(alpha)


def _image_size(image):
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("图像必须是 [batch, height, width, 3/4] 的 IMAGE。")
    if image.shape[0] != 1:
        raise ValueError("区域框选每次处理一张图；Qwen 2.1 原生参考编码也只使用首张图。请先拆分批次。")
    h, w = image.shape[1:3]
    if min(h, w) < 1:
        raise ValueError("输入图像为空。")
    return int(w), int(h)


def _box(w, h, x, y, width, height):
    values = (x, y, width, height)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("选框坐标必须是有限数值。")
    if not (0 <= x < 100 and 0 <= y < 100 and width > 0 and height > 0):
        raise ValueError("选框左上角需在 0–100% 内，宽高必须大于 0。")
    x0 = min(w - 1, max(0, math.floor(w * x / 100)))
    y0 = min(h - 1, max(0, math.floor(h * y / 100)))
    x1 = min(w, max(x0 + 1, math.ceil(w * min(100, x + width) / 100)))
    y1 = min(h, max(y0 + 1, math.ceil(h * min(100, y + height) / 100)))
    return x0, y0, x1, y1


def _mask(w, h, box, feather, device, dtype):
    x0, y0, x1, y1 = box
    rw, rh = x1 - x0, y1 - y0
    feather = min(int(feather), max(0, (min(rw, rh) - 1) // 2))
    alpha = torch.ones((rh, rw), device=device, dtype=dtype)
    if feather > 0:
        xs = torch.arange(rw, device=device)
        ys = torch.arange(rh, device=device)
        dx = torch.minimum(xs, rw - 1 - xs).to(dtype)
        dy = torch.minimum(ys, rh - 1 - ys).to(dtype)
        # Feather inwards: support NEVER expands beyond the selected rectangle.
        alpha = torch.minimum((dx / feather).clamp(0, 1)[None, :],
                              (dy / feather).clamp(0, 1)[:, None])
    mask = torch.zeros((1, h, w), device=device, dtype=dtype)
    mask[:, y0:y1, x0:x1] = alpha
    return mask


def _guide(crop, local_box):
    """Draw the position hint OUTSIDE the editable pixels, avoiding marker leakage."""
    result = crop.clone()
    x0, y0, x1, y1 = local_box
    h, w = crop.shape[1:3]
    gap, thickness = 3, 2
    color = result.new_tensor([1.0, 0.0, 1.0])
    left, right = max(0, x0-gap-thickness), min(w, x1+gap+thickness)
    top, bottom = max(0, y0-gap-thickness), min(h, y1+gap+thickness)
    if y0 >= gap + thickness:
        result[:, y0-gap-thickness:y0-gap, left:right, :3] = color
    if y1 + gap + thickness <= h:
        result[:, y1+gap:y1+gap+thickness, left:right, :3] = color
    if x0 >= gap + thickness:
        result[:, top:bottom, x0-gap-thickness:x0-gap, :3] = color
    if x1 + gap + thickness <= w:
        result[:, top:bottom, x1+gap:x1+gap+thickness, :3] = color
    return result


def _preview(image):
    # This is a nonpersistent UI preview; all processing uses the original tensor.
    rgb = image[0, :, :, :3].detach().float().clamp(0, 1).cpu()
    pil = Image.fromarray((rgb.numpy() * 255).round().astype("uint8"))
    pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    name = f"qwen_region_preview_{uuid.uuid4().hex}.png"
    import os
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    pil.save(os.path.join(folder_paths.get_temp_directory(), name))
    return {"filename": name, "subfolder": "", "type": "temp"}


class QwenRegionSelect:
    @classmethod
    def INPUT_TYPES(cls):
        pct = {"default": 25.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}
        return {"required": {
            "image": ("IMAGE",),
            "edit_text": ("STRING", {"default": "在选定区域添加一个清晰的黑色汉字‘牛’，自然贴合皮肤。", "multiline": True}),
            "x": ("FLOAT", {**pct, "tooltip": "选框左边距离图像左边的百分比。也可在预览中拖动。"}),
            "y": ("FLOAT", {**pct, "tooltip": "选框上边距离图像上边的百分比。"}),
            "width": ("FLOAT", {**pct, "default": 15.0, "min": 0.01}),
            "height": ("FLOAT", {**pct, "default": 15.0, "min": 0.01}),
            "context_pixels": ("INT", {"default": 192, "min": 0, "max": 4096, "step": 1, "tooltip": "框外保留多少原图像素作为上下文；这些像素不会贴回。"}),
            "feather_pixels": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1, "tooltip": "向框内羽化。框外像素始终保持原图。"}),
            "position_guide": ("BOOLEAN", {"default": True, "tooltip": "在编辑区域外围绘制品红定位框，协助模型定位。提示框本身不会回贴。"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "QWEN_REGION", "STRING")
    RETURN_NAMES = ("参考裁剪", "原图区域遮罩", "回贴信息", "区域编辑提示词")
    FUNCTION = "select"
    CATEGORY = "Qwen/区域编辑"
    DESCRIPTION = "自由框选、上下文裁剪、定位提示；接原生 TextEncodeQwenImage21，再用区域回贴恢复原图尺寸。"

    def select(self, image, edit_text, x, y, width, height,
               context_pixels=192, feather_pixels=8, position_guide=True):
        w, h = _image_size(image)
        box = _box(w, h, x, y, width, height)
        x0, y0, x1, y1 = box
        pad = max(0, int(context_pixels))
        # Context is visible to the model but is excluded from the final composite.
        cx0, cy0 = max(0, x0-pad), max(0, y0-pad)
        cx1, cy1 = min(w, x1+pad), min(h, y1+pad)
        crop = image[:, cy0:cy1, cx0:cx1].clone()
        local = (x0-cx0, y0-cy0, x1-cx0, y1-cy0)
        mask = _mask(w, h, box, feather_pixels, image.device, image.dtype)
        ref = _guide(crop, local) if position_guide else crop
        cw, ch = cx1-cx0, cy1-cy0
        coords = f"左边 {local[0]/cw:.1%}、上边 {local[1]/ch:.1%}、右边 {local[2]/cw:.1%}、下边 {local[3]/ch:.1%}"
        instruction = str(edit_text).strip()
        if not instruction:
            raise ValueError("请填写区域编辑指令。")
        guide_text = "图中的品红色定位线仅用于指示编辑位置，生成结果中移除定位线。" if position_guide else ""
        prompt = (f"对这张参考图做局部编辑。目标矩形在当前裁剪图中的归一化位置为：{coords}。"
                  f"{guide_text}仅在目标矩形内执行：{instruction} "
                  "保留原有主体、姿态、五官、背景、光照和其余细节。新增内容完整放在目标矩形内部，边缘自然融合。")
        info = {"version": 1, "source_size": (w, h), "box": box,
                "crop_box": (cx0, cy0, cx1, cy1), "mask": mask,
                "source_channels": image.shape[-1]}
        return {"ui": {"qwen_region_preview": [_preview(image)], "qwen_region_size": [w, h]},
                "result": (ref, mask, info, prompt)}


class QwenRegionComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "edited_crop": ("IMAGE",),
                             "region": ("QWEN_REGION",)}, "optional": {
            "alignment": (ALIGNMENT_MODES, {"default": "关闭", "tooltip": "主体模式用选框内保留的特征校正局部移动；上下文模式只用框外。低置信时不应用自动补偿。旧工作流默认关闭。"}),
            "offset_x": ("FLOAT", {"default": 0.0, "min": -2048, "max": 2048, "step": 0.01, "round": 0.01, "tooltip": "把生成内容向右移动多少原图像素；负值向左。自动模式下为额外微调。"}),
            "offset_y": ("FLOAT", {"default": 0.0, "min": -2048, "max": 2048, "step": 0.01, "round": 0.01, "tooltip": "把生成内容向下移动多少原图像素；负值向上。"}),
            "scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.001, "round": 0.001, "tooltip": "围绕裁剪中心缩放生成内容。先自动对齐，再应用这里的缩放和位移。"}),
            "color_mode": (["关闭", "轻量肤色"], {"default": "关闭", "tooltip": "自动过滤明显深色、高饱和图案，用对应肤色估计少量 RGB 色偏。取样不足或不一致时跳过；不改变结构和对比度。"}),
            "color_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "轻量颜色校正强度，建议从 0.3 起步。0 为关闭；图案颜色也会有少量变化。"}),
            "edge_feather_pixels": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1, "tooltip": "只向实际编辑蒙版内过渡，不扩大选区。建议 1 像素；0 完全关闭。不会模糊整个生成图或纹身线条。"}),
            "color_sample_mask": ("MASK", {"tooltip": "可选颜色取样蒙版，必须与原图同尺寸；白色区域用于估计色偏，不改变最终回贴范围。"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("局部编辑完成图",)
    FUNCTION = "composite"
    CATEGORY = "Qwen/区域编辑"
    DESCRIPTION = "先对齐，可选轻量肤色校正与 1 像素向内微羽化，再按实际蒙版回贴。框外逐像素保留；颜色取样不可靠时跳过，旧工作流默认关闭新功能。"

    def composite(self, original, edited_crop, region, alignment="关闭", offset_x=0.0, offset_y=0.0, scale=1.0,
                  color_mode="关闭", color_strength=0.3, edge_feather_pixels=0, color_sample_mask=None):
        w, h = _image_size(original)
        _image_size(edited_crop)
        if region.get("version") != 1 or tuple(region.get("source_size", ())) != (w, h):
            raise ValueError("回贴信息与原图尺寸不匹配。请连接同一个原图和框选节点。")
        cx0, cy0, cx1, cy1 = region["crop_box"]
        x0, y0, x1, y1 = region["box"]
        if not (0 <= cx0 <= x0 < x1 <= cx1 <= w and 0 <= cy0 <= y0 < y1 <= cy1 <= h):
            raise ValueError("回贴坐标超出原图。")
        mask = region["mask"]
        if tuple(mask.shape) != (1, h, w):
            raise ValueError("回贴遮罩尺寸错误。")
        if alignment not in ALIGNMENT_MODES:
            raise ValueError("未知的对齐模式。")
        if not all(math.isfinite(float(v)) for v in (offset_x, offset_y, scale)) or not (0.5 <= scale <= 2):
            raise ValueError("补偿位移必须为有限值，缩放需在 0.5–2.0 之间。")
        if color_mode not in ("关闭", "轻量肤色"):
            raise ValueError("未知的颜色校正模式。")
        if not math.isfinite(float(color_strength)) or not 0 <= color_strength <= 1:
            raise ValueError("颜色校正强度需要为 0–1 的有限数值。")
        if not isinstance(edge_feather_pixels, int) or edge_feather_pixels < 0:
            raise ValueError("微羽化像素需要为非负整数。")
        # Match Qwen's resized crop back to the original crop, never scale the full image.
        edit = edited_crop[:, :, :, :3].to(device=original.device, dtype=original.dtype)
        if tuple(edit.shape[1:3]) != (cy1-cy0, cx1-cx0):
            edit = F.interpolate(edit.movedim(-1, 1), size=(cy1-cy0, cx1-cx0),
                                 mode="bicubic", align_corners=False).movedim(1, -1)
        edit = edit.clamp(0, 1)
        local = (x0-cx0, y0-cy0, x1-cx0, y1-cy0)
        matrix = np.eye(3, dtype=np.float64)
        status = "对齐关闭，沿用原坐标回贴。"
        if alignment.startswith("自动"):
            source_crop = original[:, cy0:cy1, cx0:cx1, :3]
            matrix, status = _estimate_alignment(source_crop, edit, local, subject=alignment == "自动（主体特征）")
        if alignment != "关闭":
            center_x, center_y = (cx1-cx0)/2, (cy1-cy0)/2
            adjustment = np.array([[scale, 0, center_x*(1-scale)+offset_x],
                                   [0, scale, center_y*(1-scale)+offset_y], [0, 0, 1]], dtype=np.float64)
            matrix = adjustment @ matrix
            if alignment == "手动" or offset_x != 0 or offset_y != 0 or scale != 1:
                status += f" 手动补偿：x {offset_x:+.2f}px、y {offset_y:+.2f}px、缩放 {scale:.4f}。"
        color_info = None
        if color_mode != "关闭" and color_strength > 0:
            sample = None
            if color_sample_mask is not None:
                if not torch.is_tensor(color_sample_mask) or tuple(color_sample_mask.shape) != (1, h, w):
                    raise ValueError("颜色取样蒙版必须是与原图同尺寸的单张 MASK。")
                if not bool(torch.isfinite(color_sample_mask).all()):
                    raise ValueError("颜色取样蒙版含非有限值。")
                sample = color_sample_mask[:, cy0:cy1, cx0:cx1].to(original)
            aligned, coverage = _aligned_patch(edit, (0, 0, cx1-cx0, cy1-cy0), matrix)
            support = mask[:, cy0:cy1, cx0:cx1].to(original.device) > 0
            aligned, color_status, color_info = _correct_skin_color(
                original[:, cy0:cy1, cx0:cx1, :3], aligned, support, coverage, local,
                color_strength, sample)
            lx0, ly0, lx1, ly1 = local
            patch = aligned[:, ly0:ly1, lx0:lx1]
            valid = coverage[:, ly0:ly1, lx0:lx1]
            status += " " + color_status
        else:
            patch, valid = _aligned_patch(edit, local, matrix)
        alpha = mask[:, y0:y1, x0:x1].to(original).unsqueeze(-1)
        alpha = alpha * valid
        alpha = _micro_feather(alpha, edge_feather_pixels)
        if edge_feather_pixels:
            status += f" 向内微羽化 {edge_feather_pixels}px。"
        uncovered = int(torch.count_nonzero(valid == 0))
        if uncovered:
            status += f" {uncovered} 个越界像素保留原图。"
        result = original.clone()
        source = original[:, y0:y1, x0:x1, :3]
        # Difference form makes identity edits bit-exact, including feathered edges.
        result[:, y0:y1, x0:x1, :3] = torch.where(alpha > 0, source + (patch-source) * alpha, source)
        if alignment == "关闭" and color_mode == "关闭" and edge_feather_pixels == 0:
            return (result,)  # Preserve the existing three-input Python/API contract.
        logging.info("[Qwen Region] %s", status)
        ui = {"qwen_alignment_info": [status]}
        if color_info is not None:
            ui["qwen_color_info"] = [color_info]
        return {"ui": ui, "result": (result,)}
