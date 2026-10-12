"""Native Qwen reference-latent initialization and conservative masked sampling."""
import logging

import torch
import torch.nn.functional as F


def _erode(mask):
    # Outside of the crop is protected too; no wrap-around at image boundaries.
    return 1 - F.max_pool2d(F.pad(1-mask, (1, 1, 1, 1), value=1), 3, stride=1)


def _inward_feather(support, pixels):
    if pixels == 0:
        return support
    current = support
    result = torch.zeros_like(support)
    for _ in range(pixels):
        current = _erode(current)
        result += current / pixels
    return result


class QwenRegionMaskedLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING", {"tooltip": "同一个 TextEncodeQwenImage21 的正向条件，必须接入 VAE；使用其中第一张参考图的已编码潜变量。"}),
            "latent_template": ("LATENT", {"tooltip": "同一个 Qwen 编码节点的 LATENT，用于检查采样尺寸；空白 samples 会替换成原参考图潜变量。"}),
            "region": ("QWEN_REGION",),
            "boundary_lock_pixels": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "向框内保留的原图边界宽度，单位为原图像素；越大可编辑范围越小。"}),
            "blend_pixels": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "在实际可编辑范围内羽化回贴，严格不扩大到框外。"}),
        }, "optional": {
            "edit_mask": ("MASK", {"tooltip": "可选精细蒙版，必须与原图同尺寸。白色编辑、黑色保留，并与选框相交；不接时编辑整个选框。"}),
        }}

    RETURN_TYPES = ("LATENT", "MASK", "QWEN_REGION")
    RETURN_NAMES = ("原图初始化与采样蒙版", "实际编辑蒙版（裁剪）", "受保护的回贴信息")
    FUNCTION = "prepare"
    CATEGORY = "Qwen/区域编辑"
    DESCRIPTION = "从 Qwen 条件复用原参考图潜变量并附加 noise_mask，使原生采样器每步保护未选区域。关闭框选节点的定位线，回贴连接本节点第三个输出。"

    def prepare(self, positive, latent_template, region, boundary_lock_pixels=4, blend_pixels=8, edit_mask=None):
        if not isinstance(latent_template, dict) or not torch.is_tensor(latent_template.get("samples")):
            raise ValueError("请连接同一个 Qwen 编码节点的潜变量。")
        template = latent_template["samples"]
        references = []
        for entry in positive:
            if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
                refs = entry[1].get("reference_latents")
                if refs is not None and len(refs):
                    references.append(refs[0])
        if not references or not torch.is_tensor(references[0]):
            raise ValueError("正向条件没有原图潜变量；请给 TextEncodeQwenImage21 连接 VAE 和区域参考图。")
        reference = references[0]
        if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[1] != 64:
            raise ValueError("此节点仅用于 Qwen Image 2.1 的单张 64 通道图像潜变量。")
        if template.shape != reference.shape:
            raise ValueError("参考图潜变量与采样尺寸不同；请连接同一 Qwen 编码节点的条件和潜变量，不接自定义空潜空间。")
        for other in references[1:]:
            if not torch.is_tensor(other) or other.shape != reference.shape or not torch.equal(other, reference):
                raise ValueError("不同条件时段的第一参考图不一致，无法确定保留哪张原图。")
        if not torch.is_floating_point(reference) or not bool(torch.isfinite(reference).all()):
            raise ValueError("参考图潜变量需为有限浮点张量。")
        if region.get("version") != 1:
            raise ValueError("请连接自由框选节点的回贴信息。")
        w, h = region["source_size"]
        cx0, cy0, cx1, cy1 = region["crop_box"]
        x0, y0, x1, y1 = region["box"]
        if not (0 <= cx0 <= x0 < x1 <= cx1 <= w and 0 <= cy0 <= y0 < y1 <= cy1 <= h):
            raise ValueError("区域坐标超出原图。")
        source_mask = region["mask"]
        if not torch.is_tensor(source_mask) or tuple(source_mask.shape) != (1, h, w):
            raise ValueError("原图区域蒙版尺寸错误。")
        if not bool(torch.isfinite(source_mask).all()):
            raise ValueError("区域蒙版含非有限值。")
        if any(not isinstance(v, int) or not 0 <= v <= 64 for v in (boundary_lock_pixels, blend_pixels)):
            raise ValueError("边缘保护和回贴过渡需要 0–64 的整数像素。")
        mask = source_mask.to(device=reference.device, dtype=torch.float32).clamp(0, 1)
        if edit_mask is not None:
            if not torch.is_tensor(edit_mask) or tuple(edit_mask.shape) != (1, h, w):
                raise ValueError("精细蒙版必须是与原图同尺寸的单张 MASK。")
            if not bool(torch.isfinite(edit_mask).all()):
                raise ValueError("精细蒙版含非有限值。")
            mask = mask * (edit_mask.to(mask) >= 0.5)
        crop_mask = mask[:, cy0:cy1, cx0:cx1]
        support = (crop_mask[:, None] > 0).float()
        for _ in range(boundary_lock_pixels):
            support = _erode(support)
        lh, lw = reference.shape[-2:]
        # Only fully selected latent cells may change. Partial boundary cells are
        # locked, rather than letting bilinear mask resize spill outside the ROI.
        latent_mask = (F.interpolate(support, size=(lh, lw), mode="area") >= 1-1e-6).float()
        changed_cells = int(torch.count_nonzero(latent_mask))
        if changed_cells == 0:
            raise ValueError("实际采样蒙版为空：选框/精细蒙版过小或边缘保护过宽。请扩大选区或减小边缘保护。")
        ch, cw = cy1-cy0, cx1-cx0
        effective = F.interpolate(latent_mask, size=(ch, cw), mode="nearest") * support
        paste = _inward_feather(effective, blend_pixels)[:, 0] * crop_mask
        if not bool(torch.any(paste > 0)):
            raise ValueError("回贴过渡覆盖了整个选区，请减小 blend_pixels 或扩大选区。")
        result_region = region.copy()
        final_mask = torch.zeros_like(source_mask)
        final_mask[:, cy0:cy1, cx0:cx1] = paste.to(final_mask)
        result_region["mask"] = final_mask
        result = latent_template.copy()
        # Never mutate cached conditioning/reference tensors or the template.
        result["samples"] = reference.clone()
        result["noise_mask"] = latent_mask
        ratio = changed_cells / (lh*lw) * 100
        status = (f"原图初始化：{lw*16}×{lh*16}，64 通道；可编辑 {changed_cells}/{lh*lw} 个潜空间单元（{ratio:.2f}%）。"
                  "每步保护蒙版外，回贴只使用实际选区。请将采样 denoise 设为 0.65–0.90 起步；大框内仍可被改画。")
        logging.info("[Qwen Region Masked] %s", status)
        return {"ui": {"qwen_inpaint_info": [status]},
                "result": (result, effective[:, 0], result_region)}
