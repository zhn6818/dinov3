# DINOv3 模型使用指南

本指南详细介绍加载DINOv3模型后可以执行的所有任务和应用场景。

## 📋 目录

1. [基础特征提取](#1-基础特征提取)
2. [图像分类](#2-图像分类)
3. [语义分割](#3-语义分割)
4. [目标检测](#4-目标检测)
5. [深度估计](#5-深度估计)
6. [图文理解 (DINOTxt)](#6-图文理解-dinotxt)
7. [高级应用](#7-高级应用)
8. [模型加载方式](#8-模型加载方式)

---

## 1. 基础特征提取

### 1.1 提取图像特征向量

DINOv3的核心功能是提取高质量的图像特征，这些特征可以用于各种下游任务。

#### 使用PyTorch Hub加载backbone模型

```python
import torch
from PIL import Image
from torchvision.transforms import v2

# 加载DINOv3 backbone模型
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
model.eval()

# 准备图像
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

image = Image.open("your_image.jpg").convert("RGB")
img_tensor = transform(image).unsqueeze(0)

# 提取特征
with torch.no_grad():
    features = model.forward_features(img_tensor)
    
    # CLS token特征 (全局图像特征)
    cls_token = features["x_norm_clstoken"]  # [1, 768]
    
    # Patch tokens特征 (每个patch的特征)
    patch_tokens = features["x_norm_patchtokens"]  # [1, 196, 768] for 224x224
    
    # 平均池化得到全局特征
    global_feature = patch_tokens.mean(dim=1)  # [1, 768]
```

#### 可用的backbone模型

```python
# ViT模型
dinov3_vits16      # ViT-Small/16 (21M参数)
dinov3_vits16plus  # ViT-Small+/16 (29M参数)
dinov3_vitb16      # ViT-Base/16 (86M参数)
dinov3_vitl16      # ViT-Large/16 (300M参数)
dinov3_vitl16plus  # ViT-Large+/16
dinov3_vith16plus  # ViT-Huge+/16 (840M参数)
dinov3_vit7b16     # ViT-7B/16 (6.7B参数)

# ConvNeXt模型
dinov3_convnext_tiny
dinov3_convnext_small
dinov3_convnext_base
dinov3_convnext_large
```

### 1.2 特征应用场景

- **图像检索**：使用CLS token或平均池化的patch tokens作为图像描述符
- **相似度计算**：计算两张图像特征的余弦相似度
- **聚类分析**：对图像特征进行聚类
- **降维可视化**：使用PCA/t-SNE对特征降维可视化

---

## 2. 图像分类

### 2.1 使用预训练的分类头

DINOv3提供了在ImageNet-1k上训练的线性分类头。

```python
import torch

# 加载带分类头的完整模型
classifier = torch.hub.load(
    'facebookresearch/dinov3', 
    'dinov3_vit7b16_lc',  # Linear Classifier
    pretrained=True
)
classifier.eval()

# 准备图像
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

image = Image.open("your_image.jpg").convert("RGB")
img_tensor = transform(image).unsqueeze(0)

# 预测
with torch.no_grad():
    logits = classifier(img_tensor)  # [1, 1000]
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = logits.argmax(dim=1).item()
    
print(f"预测类别: {predicted_class}")
print(f"置信度: {probabilities[0, predicted_class].item():.4f}")
```

### 2.2 训练自定义分类器

```python
# 1. 加载backbone并冻结
backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
for param in backbone.parameters():
    param.requires_grad = False

# 2. 添加自定义分类头
import torch.nn as nn
classifier_head = nn.Linear(768, num_classes)  # num_classes是你的类别数

# 3. 训练分类头
model = nn.Sequential(backbone, classifier_head)
# ... 训练代码 ...
```

---

## 3. 语义分割

### 3.1 使用预训练的分割模型

DINOv3提供了在ADE20K数据集上训练的语义分割模型。

```python
import torch
from PIL import Image
from torchvision.transforms import v2
from dinov3.eval.segmentation.inference import make_inference
from functools import partial

# 加载分割模型
segmentor = torch.hub.load(
    'facebookresearch/dinov3',
    'dinov3_vit7b16_ms',  # Multi-scale Segmentor
    pretrained=True
)
segmentor.eval()

# 准备图像
def make_transform(resize_size=896):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

img_size = 896
image = Image.open("your_image.jpg").convert("RGB")
transform = make_transform(img_size)
img_tensor = transform(image).unsqueeze(0)

# 执行分割
with torch.no_grad():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        # 获取分割结果 (150个ADE20K类别)
        segmentation_map = make_inference(
            img_tensor,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(image.size[1], image.size[0]),  # 恢复到原始尺寸
            n_output_channels=150,  # ADE20K有150个类别
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)  # [1, 1, H, W]

# 可视化结果
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map[0, 0].cpu(), cmap="tab20")
plt.axis("off")
plt.show()
```

### 3.2 自定义分割任务

```python
# 1. 加载backbone
backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')

# 2. 添加分割解码器
from dinov3.eval.segmentation.models import build_segmentation_decoder
decoder = build_segmentation_decoder(
    backbone_model=backbone,
    decoder_type="m2f",  # 或 "linear"
    hidden_dim=256,
    num_classes=your_num_classes,
)

# 3. 训练分割模型
# ... 训练代码 ...
```

---

## 4. 目标检测

### 4.1 使用预训练的检测模型

DINOv3提供了在COCO2017上训练的目标检测模型。

```python
import torch
from PIL import Image
from torchvision.transforms import v2

# 加载检测模型
detector = torch.hub.load(
    'facebookresearch/dinov3',
    'dinov3_vit7b16_de',  # Detector
    pretrained=True
)
detector.eval()

# 准备图像
def make_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

image = Image.open("your_image.jpg").convert("RGB")
transform = make_transform()
img_tensor = transform(image).unsqueeze(0)

# 执行检测
with torch.no_grad():
    # 输入需要是list of tensors
    results = detector([img_tensor[0]])  # 返回list of dicts

# 结果格式
for result in results:
    boxes = result["boxes"]      # [N, 4] in XYXY format
    scores = result["scores"]    # [N]
    labels = result["labels"]   # [N] (COCO类别ID, 1-90)

# 可视化检测结果
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(image)

for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:  # 置信度阈值
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label}: {score:.2f}", 
                bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.5))

plt.show()
```

### 4.2 自定义检测任务

```python
# 1. 加载backbone
backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')

# 2. 添加检测头
from dinov3.eval.detection.models.detr import build_model
detector = build_model(
    backbone=backbone,
    num_classes=your_num_classes,
    # ... 其他配置 ...
)

# 3. 训练检测模型
# ... 训练代码 ...
```

---

## 5. 深度估计

### 5.1 使用预训练的深度估计模型

DINOv3提供了在SYNTHMIX数据集上训练的深度估计模型。

```python
import torch
from PIL import Image
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps

# 加载深度估计模型
depther = torch.hub.load(
    'facebookresearch/dinov3',
    'dinov3_vit7b16_dd',  # Depth Estimator
    pretrained=True
)
depther.eval()

# 准备图像
def make_transform(resize_size=1024):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

img_size = 1024
image = Image.open("your_image.jpg").convert("RGB")
transform = make_transform(img_size)
img_tensor = transform(image).unsqueeze(0)

# 执行深度估计
with torch.no_grad():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        depths = depther(img_tensor)  # [1, 1, H, W]

# 可视化深度图
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.title("原始图像")
plt.subplot(122)
plt.imshow(depths[0, 0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")
plt.title("深度图")
plt.colorbar()
plt.show()
```

### 5.2 自定义深度估计

```python
# 1. 加载backbone
backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')

# 2. 添加深度解码器
from dinov3.eval.depth.models import make_depther_from_config
from dinov3.eval.depth.models import DecoderConfig

config = DecoderConfig(
    min_depth=0.001,
    max_depth=100.0,
    backbone_out_layers=[4, 11, 17, 23],  # 不同层的特征
    n_output_channels=256,
    use_backbone_norm=True,
    use_batchnorm=True,
)

depther = make_depther_from_config(
    backbone=backbone,
    decoder_config=config,
)

# 3. 训练深度估计模型
# ... 训练代码 ...
```

---

## 6. 图文理解 (DINOTxt)

### 6.1 零样本图文理解

DINOv3提供了DINOTxt模型，支持零样本的图文理解任务。

```python
import torch
from PIL import Image
from torchvision.transforms import v2

# 加载DINOTxt模型和tokenizer
model, tokenizer = torch.hub.load(
    'facebookresearch/dinov3',
    'dinov3_vitl16_dinotxt_tet1280d20h24l',
    pretrained=True
)
model.eval()

# 准备图像
def make_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

image = Image.open("your_image.jpg").convert("RGB")
transform = make_transform()
img_tensor = transform(image).unsqueeze(0)

# 准备文本
texts = ["a cat", "a dog", "a car"]
text_tokens = tokenizer(texts)  # 编码文本

# 计算图文相似度
with torch.no_grad():
    # 获取图像特征
    image_features = model.encode_image(img_tensor)
    
    # 获取文本特征
    text_features = model.encode_text(text_tokens)
    
    # 计算相似度
    similarity = image_features @ text_features.T  # [1, num_texts]
    
    # 找到最匹配的文本
    best_match_idx = similarity.argmax().item()
    print(f"最匹配的文本: {texts[best_match_idx]}")
    print(f"相似度: {similarity[0, best_match_idx].item():.4f}")
```

### 6.2 零样本分割

使用DINOTxt进行零样本语义分割：

```python
# 参考notebook: notebooks/dinotxt_segmentation_inference.ipynb
# 可以基于文本描述进行语义分割，无需预定义类别
```

---

## 7. 高级应用

### 7.1 PCA特征可视化

可视化DINOv3的patch特征，用于理解模型学到的表示。

```python
# 参考notebook: notebooks/pca.ipynb
# 对patch tokens进行PCA降维，生成彩虹可视化
```

### 7.2 前景分割

基于DINOv3特征训练简单的前景分割模型。

```python
# 参考notebook: notebooks/foreground_segmentation.ipynb
# 使用线性分类器在patch特征上训练前景/背景分割
```

### 7.3 密集和稀疏匹配

使用DINOv3特征匹配两张图像中的对应区域。

```python
# 参考notebook: notebooks/dense_sparse_matching.ipynb
# 计算两张图像patch特征的相似度，找到对应区域
```

### 7.4 分割跟踪

在视频中使用DINOv3特征进行分割跟踪。

```python
# 参考notebook: notebooks/segmentation_tracking.ipynb
# 基于非参数方法，使用DINOv3特征跟踪视频中的对象
```

---

## 8. 模型加载方式

### 8.1 使用PyTorch Hub (推荐)

```python
import torch

# 加载backbone
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')

# 加载带任务头的模型
classifier = torch.hub.load('facebookresearch/dinov3', 'dinov3_vit7b16_lc')
segmentor = torch.hub.load('facebookresearch/dinov3', 'dinov3_vit7b16_ms')
detector = torch.hub.load('facebookresearch/dinov3', 'dinov3_vit7b16_de')
depther = torch.hub.load('facebookresearch/dinov3', 'dinov3_vit7b16_dd')
```

### 8.2 使用本地路径

```python
# 如果已经下载了权重文件
model = torch.hub.load(
    'facebookresearch/dinov3', 
    'dinov3_vitb16',
    weights='path/to/weights.pth'
)
```

### 8.3 使用Hugging Face Transformers

```python
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

# 加载模型
pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name)

# 处理图像
image = load_image("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# 提取特征
with torch.inference_mode():
    outputs = model(**inputs)
    features = outputs.last_hidden_state
```

### 8.4 使用timm库

```python
import timm

# 加载DINOv3模型
model = timm.create_model(
    'dinov3_vitb16',
    pretrained=True,
    num_classes=0,  # 只加载backbone
)
```

---

## 9. 完整示例：图像检索系统

```python
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
import numpy as np
from pathlib import Path

# 1. 加载模型
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
model.eval()

# 2. 准备图像变换
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 3. 提取图像库的特征
image_dir = Path("image_database")
image_features = {}
image_paths = []

for img_path in image_dir.glob("*.jpg"):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        # 使用CLS token作为图像描述符
        feature = features["x_norm_clstoken"].squeeze(0)
        image_features[img_path] = feature
        image_paths.append(img_path)

# 4. 查询相似图像
query_image = Image.open("query.jpg").convert("RGB")
query_tensor = transform(query_image).unsqueeze(0)

with torch.no_grad():
    query_features = model.forward_features(query_tensor)
    query_feature = query_features["x_norm_clstoken"].squeeze(0)

# 5. 计算相似度
similarities = []
for img_path, feature in image_features.items():
    similarity = F.cosine_similarity(
        query_feature.unsqueeze(0),
        feature.unsqueeze(0)
    ).item()
    similarities.append((img_path, similarity))

# 6. 排序并返回最相似的图像
similarities.sort(key=lambda x: x[1], reverse=True)
top_k = 5
print(f"Top {top_k} 最相似的图像:")
for i, (path, sim) in enumerate(similarities[:top_k]):
    print(f"{i+1}. {path.name}: {sim:.4f}")
```

---

## 10. 总结

DINOv3可以用于以下任务：

| 任务类型 | 模型 | 数据集 | 应用场景 |
|---------|------|--------|---------|
| **特征提取** | Backbone | - | 图像检索、相似度计算、降维可视化 |
| **图像分类** | Linear Classifier | ImageNet-1k | 图像分类、自定义分类器训练 |
| **语义分割** | Segmentor (M2F) | ADE20K | 语义分割、自定义分割任务 |
| **目标检测** | Detector | COCO2017 | 目标检测、自定义检测任务 |
| **深度估计** | Depther (DPT) | SYNTHMIX | 单目深度估计、3D重建 |
| **图文理解** | DINOTxt | LVTD-2300M | 零样本图文匹配、零样本分割 |
| **高级应用** | Backbone | - | PCA可视化、前景分割、图像匹配、视频跟踪 |

### 选择建议

- **需要通用特征**：使用backbone模型 (`dinov3_vitb16`等)
- **图像分类**：使用分类器 (`dinov3_vit7b16_lc`)
- **分割任务**：使用分割器 (`dinov3_vit7b16_ms`)
- **检测任务**：使用检测器 (`dinov3_vit7b16_de`)
- **深度估计**：使用深度估计器 (`dinov3_vit7b16_dd`)
- **图文理解**：使用DINOTxt (`dinov3_vitl16_dinotxt_tet1280d20h24l`)

### 性能与资源

- **小模型** (ViT-S): 21M参数，速度快，适合实时应用
- **中等模型** (ViT-B): 86M参数，平衡性能和速度
- **大模型** (ViT-L): 300M参数，性能更好
- **超大模型** (ViT-7B): 6.7B参数，最佳性能，需要更多资源

---

## 参考资料

- [官方README](README.md)
- [模型卡片](MODEL_CARD.md)
- [数据集说明](DATASETS.md)
- [Notebooks示例](notebooks/)

更多详细信息请参考官方文档和notebooks示例。

