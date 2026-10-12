# ComfyUI Qwen Region Editor

Qwen Image 2.1 区域编辑节点：在原图预览中自由框选位置，结合参考图和普通提示词生成内容，并只将允许编辑的区域回贴到原图。

支持放大框选、上下文裁剪、定位线、原图潜变量初始化、采样蒙版、区域回贴，以及可选的自动或手动对齐。

## 安装

在 `ComfyUI/custom_nodes` 目录执行：

```bash
git clone https://github.com/zb18626140292-crypto/ComfyUI-Qwen-Region-Editor.git comfyui_qwen_region_editor
```

也可以下载仓库 ZIP，将源码文件夹放到 `ComfyUI/custom_nodes/comfyui_qwen_region_editor`。请确认该目录下直接包含 `__init__.py`。

需要 ComfyUI 环境中的 `torch`、`numpy` 和 `Pillow`。其中 torch 由 ComfyUI 环境统一管理；其余依赖列在 `requirements.txt`。缺少依赖时，请使用运行 ComfyUI 的同一个 Python 环境安装。

Windows portable 安装示例，在 portable 根目录执行：

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\comfyui_qwen_region_editor\requirements.txt
```

安装后重启 ComfyUI，再在浏览器按 Ctrl+F5 刷新。搜索 `Qwen`，或在 `Qwen/区域编辑` 分类中添加节点。

自动对齐可选依赖 OpenCV；默认对齐关闭，不需要安装 OpenCV。需要使用自动对齐时，在同一 Python 环境安装 `opencv-python`。

## 节点

| 节点类 | 显示名称 | 作用 |
| --- | --- | --- |
| QwenRegionSelect | Qwen · 自由框选编辑区域 | 自由框选，输出参考裁剪、原图遮罩、回贴信息和区域提示词 |
| QwenRegionMaskedLatent | Qwen · 原图初始化与蒙版锁定 | 复用第一张参考图的潜变量，设置采样蒙版，输出受保护的回贴信息 |
| QwenRegionComposite | Qwen · 区域回贴（保留原图） | 将生成裁剪回贴到允许编辑的区域，保留框外原图 |
| QwenRegionPlaceDecal | Qwen · 图二按框预贴 | 旧版工作流使用的图案预贴功能 |
| QwenRegionDecalBlend | Qwen · 保留图案并融合 | 旧版工作流使用的图案保留与融合功能 |

参考人物、物体或文字并重新生成时，使用前三个节点组合即可。后两个节点保留用于兼容需要预贴图案的旧工作流。

## 推荐连接方法

需要提供原生 `TextEncodeQwenImage21` 节点的 ComfyUI 环境，以及支持 `noise_mask` 的采样器。

1. 原图接入 `QwenRegionSelect.image`。点击「放大框选」拖动矩形，或用 `x`、`y`、`width`、`height` 百分比调整位置。
2. 将选框节点的「参考裁剪」连接到 Qwen 编码器的第一张图像输入；人物、物体等参考图接后续图像输入。给同一个编码器连接 VAE。
3. 使用普通 `Text (Multiline)` 节点填写简短编辑要求，直接连接 Qwen 编码器的提示词输入。可启用选框的定位线以帮助模型理解位置；选框节点第四个输出中的自动区域提示词可按需要使用。
4. 同一个 Qwen 编码器的正向条件和潜变量，分别连接 `QwenRegionMaskedLatent.positive` 与 `latent_template`。选框节点第三个输出接 `QwenRegionMaskedLatent.region`。
5. 将蒙版锁定节点第一个输出接采样器的 latent 输入，采样后用 VAE 解码。
6. 原图接 `QwenRegionComposite.original`，解码图接 `edited_crop`；**回贴节点的 `region` 必须接蒙版锁定节点的第三个输出「受保护的回贴信息」**。

第 6 步使采样范围和回贴范围一致，避免把未采样的边缘单元贴回后，在人物脚底或其他边界产生杂纹。

示例提示词：

> 只在图一紫色定位框内加入图二的成年女性，参考长相、发型和服装。完整全身站立，右手拿饮料杯举到胸前，按街景生成透视和光照，双脚接地，保留其余街景。

## 框选与融合参数

- `context_pixels`：框外提供给模型观察的上下文像素。参考裁剪可能大于选框，最终回贴仍限定在选框及其蒙版内。
- `position_guide`：在参考裁剪中绘制品红定位线。定位线位于编辑框外，最终回贴不保留它。
- `feather_pixels`：选框遮罩向内羽化。需要清晰边缘时设为 `0`。
- `boundary_lock_pixels`：蒙版锁定节点向框内保留的边界宽度。
- `blend_pixels`：实际可编辑范围内的回贴过渡宽度。需要零羽化时设为 `0`。
- `edit_mask`：可选精细蒙版，必须是与原图同尺寸的单张 MASK。白色编辑、黑色保留，与选框取交集。

给头发、手持物、双脚和接触阴影留出空间，避免紧贴框边。潜空间边界中未被完整选中的单元会保留原图，因此实际可编辑范围可能略小于矩形。

区域节点负责位置、蒙版和回贴。人物一致性、动作、光照和自然融合仍由编辑模型、参考图、提示词及采样参数决定。

## 对齐功能

`QwenRegionComposite` 支持关闭、手动以及基于主体或上下文特征的自动对齐。手动模式可用 `offset_x`、`offset_y` 和 `scale` 调整生成裁剪；位移单位为原图像素。

自动对齐只有在特征匹配可信时应用补偿，无法还原模型重画的五官或非刚性形变。新增内容本身不需要贴合原有人物时，通常保持对齐关闭。

## 兼容与验证范围

- `QwenRegionMaskedLatent` 限定 Qwen Image 2.1 的单张、64 通道图像潜变量。正向条件和 latent 必须来自同一 Qwen 编码器，且包含第一参考图的 `reference_latents`；其他模型架构或批量图像不在该节点支持范围内。
- 来源插件在 ComfyUI 0.39.2、前端 1.53.10 环境中运行。发布前已核验五个节点在当前 `/object_info` 中注册，并检查 Python / JavaScript 语法。实现沿用当前部署版，发布副本仅规范文件末尾空行。
- 本次 Git 发布没有重新进行 GPU 采样；此前相关工作流已有运行结果。不同模型、种子和图片的视觉效果需单独测试。

本仓库提供区域编辑插件源码与安装说明。模型权重和完整采样工作流需按你的 ComfyUI 环境准备。
