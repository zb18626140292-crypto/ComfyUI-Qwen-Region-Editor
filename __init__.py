from .region_nodes import QwenRegionSelect, QwenRegionComposite
from .masked_nodes import QwenRegionMaskedLatent
from .decal_nodes import QwenRegionPlaceDecal, QwenRegionDecalBlend

NODE_CLASS_MAPPINGS = {
    "QwenRegionSelect": QwenRegionSelect,
    "QwenRegionComposite": QwenRegionComposite,
    "QwenRegionMaskedLatent": QwenRegionMaskedLatent,
    "QwenRegionPlaceDecal": QwenRegionPlaceDecal,
    "QwenRegionDecalBlend": QwenRegionDecalBlend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenRegionSelect": "Qwen · 自由框选编辑区域",
    "QwenRegionComposite": "Qwen · 区域回贴（保留原图）",
    "QwenRegionMaskedLatent": "Qwen · 原图初始化与蒙版锁定",
    "QwenRegionPlaceDecal": "Qwen · 图二按框预贴",
    "QwenRegionDecalBlend": "Qwen · 保留图案并融合",
}
WEB_DIRECTORY = "./web"
