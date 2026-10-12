"""Place reference artwork before sampling, then limit Qwen's pixel contribution."""
import math

import torch
import torch.nn.functional as F


def _image(image, name):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f"{name}需要单张 RGB/RGBA IMAGE。")
    if not torch.is_floating_point(image) or not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name}需要有限浮点像素。")
    return image.shape[2], image.shape[1]


def _resize(tensor, height, width):
    return F.interpolate(tensor.movedim(-1, 1), size=(height, width), mode="bicubic",
                         align_corners=False, antialias=True).movedim(1, -1)


class QwenRegionPlaceDecal:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",),
            "artwork": ("IMAGE", {"tooltip": "图二：需要放到选区的文字或图案。单张图片。"}),
            "region": ("QWEN_REGION",),
            "background": (["去浅色背景", "保留图片背景"], {"default": "去浅色背景", "tooltip": "白底或浅灰底的彩色文字选去浅色背景；白色图案自身也可能被去除，必要时使用保留背景或精细蒙版。"}),
            "fit": (["等比完整", "拉伸铺满"], {"default": "等比完整", "tooltip": "等比完整保留字形；拉伸铺满会改变长宽比例。"}),
            "padding_pixels": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip": "图案与选框边缘之间留多少原图像素。"}),
            "opacity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            "light_threshold": ("FLOAT", {"default": 0.90, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "RGB 三个分量均高于此值的浅色会去掉。"}),
            "key_softness": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.25, "step": 0.005}),
            "fusion_margin_pixels": ("INT", {"default": 16, "min": 0, "max": 64, "tooltip": "允许 Qwen 融合的图案周边范围；始终限制在选框内。"}),
        }, "optional": {
            "artwork_mask": ("MASK", {"tooltip": "可选：与图二同尺寸，白色保留图案、黑色透明。连接后替代自动去浅色背景；普通 LoadImage 的 MASK 是透明通道的反相，不要直接连接。"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "QWEN_REGION")
    RETURN_NAMES = ("按框预贴完成图", "预贴参考裁剪", "图案范围（原图尺寸）", "图案融合区域")
    FUNCTION = "place"
    CATEGORY = "Qwen/区域编辑"
    DESCRIPTION = "先去浅色背景并将图二按选框缩放居中放入原图，图案无需模型重新生成。框外保持原图；再将预贴裁剪送入 Qwen 轻度融合。"

    def place(self, original, artwork, region, background="去浅色背景", fit="等比完整",
              padding_pixels=8, opacity=1.0, light_threshold=0.90, key_softness=0.05,
              fusion_margin_pixels=16, artwork_mask=None):
        w, h = _image(original, "原图")
        aw, ah = _image(artwork, "图二")
        if region.get("version") != 1 or tuple(region.get("source_size", ())) != (w, h):
            raise ValueError("选框信息与原图不匹配。")
        x0, y0, x1, y1 = region["box"]
        cx0, cy0, cx1, cy1 = region["crop_box"]
        if not (0 <= cx0 <= x0 < x1 <= cx1 <= w and 0 <= cy0 <= y0 < y1 <= cy1 <= h):
            raise ValueError("选区或裁剪坐标无效。")
        roi_mask = region["mask"]
        if not torch.is_tensor(roi_mask) or tuple(roi_mask.shape) != (1, h, w) or not bool(torch.isfinite(roi_mask).all()):
            raise ValueError("选框蒙版尺寸或像素无效。")
        if background not in ("去浅色背景", "保留图片背景") or fit not in ("等比完整", "拉伸铺满"):
            raise ValueError("背景或缩放方式无效。")
        if any(not isinstance(v, int) or not 0 <= v <= limit for v, limit in ((padding_pixels, 256), (fusion_margin_pixels, 64))):
            raise ValueError("留边或融合范围无效。")
        if not all(math.isfinite(float(v)) for v in (opacity, light_threshold, key_softness)) or not (0 < opacity <= 1 and .5 <= light_threshold <= 1 and .005 <= key_softness <= .25):
            raise ValueError("透明度或背景去除参数无效。")
        art = artwork.to(device=original.device, dtype=torch.float32).clamp(0, 1)
        alpha = art[..., 3:4] if art.shape[-1] == 4 else torch.ones_like(art[..., :1])
        if artwork_mask is not None:
            if not torch.is_tensor(artwork_mask) or tuple(artwork_mask.shape) != (1, ah, aw) or not bool(torch.isfinite(artwork_mask).all()):
                raise ValueError("图案蒙版需要与图二同尺寸的单张 MASK。")
            alpha = alpha * artwork_mask.to(alpha).clamp(0, 1)[..., None]
        elif background == "去浅色背景":
            alpha = alpha * ((light_threshold-art[..., :3].amin(dim=-1, keepdim=True))/key_softness).clamp(0, 1)
        foreground = torch.nonzero(alpha[0, ..., 0] > 1/255, as_tuple=False)
        if foreground.numel() == 0:
            raise ValueError("图二去背景后为空。请改为保留图片背景、提高浅色阈值，或检查图案蒙版。")
        ay0, ax0 = foreground.amin(dim=0).tolist()
        ay1, ax1 = (foreground.amax(dim=0)+1).tolist()
        rgb = art[:, ay0:ay1, ax0:ax1, :3]
        alpha = alpha[:, ay0:ay1, ax0:ax1]
        inner_w, inner_h = x1-x0-2*padding_pixels, y1-y0-2*padding_pixels
        if min(inner_w, inner_h) <= 0:
            raise ValueError("选框太小，留边覆盖了整个选区。请减小留边或扩大选框。")
        if fit == "等比完整":
            factor = min(inner_w/rgb.shape[2], inner_h/rgb.shape[1])
            tw, th = min(inner_w, max(1, round(rgb.shape[2]*factor))), min(inner_h, max(1, round(rgb.shape[1]*factor)))
        else:
            tw, th = inner_w, inner_h
        # Premultiplied filtering prevents the discarded light background from
        # leaking into translucent artwork edges during resize.
        color_alpha = _resize(rgb*alpha, th, tw).clamp(0, 1)
        alpha = _resize(alpha, th, tw).clamp(0, 1)
        rgb = (color_alpha/alpha.clamp_min(1e-6)).clamp(0, 1)
        tx, ty = x0+(x1-x0-tw)//2, y0+(y1-y0-th)//2
        local_roi = roi_mask[:, ty:ty+th, tx:tx+tw].to(alpha).clamp(0, 1)[..., None]
        alpha = alpha * local_roi * opacity
        if not bool(torch.any(alpha > 0)):
            raise ValueError("图案没有落入有效选区，请检查框选蒙版。")
        placed = original.clone()
        base = original[:, ty:ty+th, tx:tx+tw, :3]
        a = alpha.to(base)
        placed[:, ty:ty+th, tx:tx+tw, :3] = torch.where(a > 0, base+(rgb.to(base)-base)*a, base)
        art_mask = original.new_zeros((1, h, w))
        art_mask[:, ty:ty+th, tx:tx+tw] = alpha[..., 0].to(art_mask)
        # The context image may be read by Qwen, but neither its noise mask nor
        # its final paste mask may extend beyond the selected original pixels.
        support = (art_mask[:, None, y0:y1, x0:x1] > 0).float()
        if fusion_margin_pixels:
            r = fusion_margin_pixels
            support = F.max_pool2d(support, 2*r+1, stride=1, padding=r)
        fusion_mask = torch.zeros_like(roi_mask)
        fusion_mask[:, y0:y1, x0:x1] = support[:, 0].to(roi_mask)*roi_mask[:, y0:y1, x0:x1].clamp(0, 1)
        result_region = region.copy()
        result_region["mask"] = fusion_mask
        return (placed, placed[:, cy0:cy1, cx0:cx1].clone(), art_mask, result_region)


class QwenRegionDecalBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "placed_crop": ("IMAGE",),
            "generated_crop": ("IMAGE",),
            "qwen_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                       "tooltip": "Qwen 结果占比。0 完整保留预贴图案；默认 0.15 保留 85% 预贴图，避免模型将图案擦掉。提高会增加融合和改画。"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("保留图案的融合裁剪",)
    FUNCTION = "blend"
    CATEGORY = "Qwen/区域编辑"
    DESCRIPTION = "在预贴图案与 Qwen 结果之间做受限混合，再交给区域回贴。默认以预贴图为主，避免文字或图案消失。"

    def blend(self, placed_crop, generated_crop, qwen_strength=.15):
        w, h = _image(placed_crop, "预贴裁剪")
        _image(generated_crop, "生成裁剪")
        if not math.isfinite(float(qwen_strength)) or not 0 <= qwen_strength <= .5:
            raise ValueError("Qwen 融合占比需要为 0–0.5 的有限数值。")
        if qwen_strength == 0:
            return (placed_crop.clone(),)
        edit = generated_crop[..., :3].to(placed_crop)
        if tuple(edit.shape[1:3]) != (h, w):
            edit = _resize(edit, h, w)
        result = placed_crop.clone()
        result[..., :3] = (placed_crop[..., :3]+(edit.clamp(0, 1)-placed_crop[..., :3])*qwen_strength).clamp(0, 1)
        return (result,)
