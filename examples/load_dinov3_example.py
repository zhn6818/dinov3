#!/usr/bin/env python3
"""
DINOv3模型加载和使用示例

本脚本演示如何加载DINOv3模型并执行各种任务。
"""

import os
import torch
from PIL import Image
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np


def make_image_transform(resize_size=224):
    """创建图像预处理变换"""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def example_1_feature_extraction(weights_path=None, use_huggingface=False):
    """示例1: 基础特征提取
    
    Args:
        weights_path: 本地权重文件路径（可选）
        use_huggingface: 是否使用 Hugging Face 加载模型（可选）
    """
    print("\n" + "="*50)
    print("示例1: 基础特征提取")
    print("="*50)
    
    # 加载backbone模型
    print("正在加载DINOv3 ViT-B/16模型...")
    
    if use_huggingface:
        # 使用 Hugging Face 加载（推荐，如果网络可以访问 Hugging Face）
        try:
            from transformers import AutoModel
            print("使用 Hugging Face 加载模型...")
            model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
            model.eval()
            print("模型加载完成!")
        except Exception as e:
            print(f"Hugging Face 加载失败: {e}")
            print("尝试使用 PyTorch Hub...")
            use_huggingface = False
    
    if not use_huggingface:
        if weights_path and os.path.exists(weights_path):
            # 使用本地权重文件
            print(f"从本地路径加载权重: {weights_path}")
            model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16', 
                                  source='local', weights=weights_path)
        else:
            # 使用 PyTorch Hub（需要网络）
            try:
                model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
            except Exception as e:
                print(f"网络下载失败: {e}")
                print("\n解决方案：")
                print("1. 手动下载权重文件：")
                print("   wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m.pth")
                print("   然后使用: weights_path='./dinov3_vitb16_pretrain_lvd1689m.pth'")
                print("\n2. 或使用 Hugging Face: use_huggingface=True")
                print("\n3. 或设置代理/镜像源")
                raise
        model.eval()
        print("模型加载完成!")
    
    # 准备图像 (这里使用一个示例URL，实际使用时替换为你的图像路径)
    print("\n准备图像...")
    try:
        import requests
        from io import BytesIO
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url, stream=True)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"图像尺寸: {image.size}")
    except:
        print("无法下载示例图像，请提供本地图像路径")
        return
    
    # 预处理
    transform = make_image_transform(224)
    img_tensor = transform(image).unsqueeze(0)
    print(f"输入tensor形状: {img_tensor.shape}")
    
    # 提取特征
    print("\n提取特征...")
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        
        # CLS token (全局图像特征)
        cls_token = features["x_norm_clstoken"]
        print(f"CLS token形状: {cls_token.shape}")
        print(f"CLS token特征维度: {cls_token.shape[-1]}")
        
        # Patch tokens (每个patch的特征)
        patch_tokens = features["x_norm_patchtokens"]
        print(f"Patch tokens形状: {patch_tokens.shape}")
        print(f"Patch数量: {patch_tokens.shape[1]}")
        
        # 平均池化得到全局特征
        global_feature = patch_tokens.mean(dim=1)
        print(f"全局特征形状: {global_feature.shape}")
    
    print("\n特征提取完成!")
    return model, features


def example_2_image_classification():
    """示例2: 图像分类"""
    print("\n" + "="*50)
    print("示例2: 图像分类 (需要下载分类头权重)")
    print("="*50)
    print("注意: 此示例需要访问Meta的模型权重下载页面")
    print("请参考README.md获取权重下载链接")
    
    # 加载分类器 (需要权重文件)
    # classifier = torch.hub.load(
    #     'facebookresearch/dinov3',
    #     'dinov3_vit7b16_lc',
    #     pretrained=True
    # )
    # classifier.eval()
    # 
    # # 准备图像
    # image = Image.open("your_image.jpg").convert("RGB")
    # transform = make_image_transform(224)
    # img_tensor = transform(image).unsqueeze(0)
    # 
    # # 预测
    # with torch.no_grad():
    #     logits = classifier(img_tensor)
    #     probabilities = torch.softmax(logits, dim=1)
    #     predicted_class = logits.argmax(dim=1).item()
    #     confidence = probabilities[0, predicted_class].item()
    # 
    # print(f"预测类别: {predicted_class}")
    # print(f"置信度: {confidence:.4f}")
    
    print("跳过此示例 (需要权重文件)")


def example_3_custom_classifier():
    """示例3: 创建自定义分类器"""
    print("\n" + "="*50)
    print("示例3: 创建自定义分类器")
    print("="*50)
    
    # 加载backbone
    print("加载DINOv3 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
    
    # 冻结backbone参数
    for param in backbone.parameters():
        param.requires_grad = False
    
    # 创建自定义分类头
    num_classes = 10  # 假设有10个类别
    embed_dim = 768   # ViT-B的embed维度
    
    classifier_head = torch.nn.Linear(embed_dim, num_classes)
    
    # 组合模型
    class CustomClassifier(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            features = self.backbone.forward_features(x)
            cls_token = features["x_norm_clstoken"]
            return self.head(cls_token)
    
    model = CustomClassifier(backbone, classifier_head)
    print(f"自定义分类器创建完成!")
    print(f"可训练参数: {sum(p.numel() for p in model.head.parameters() if p.requires_grad)}")
    print(f"冻结参数: {sum(p.numel() for p in model.backbone.parameters())}")
    
    return model


def example_4_feature_similarity():
    """示例4: 计算图像相似度"""
    print("\n" + "="*50)
    print("示例4: 计算图像相似度")
    print("="*50)
    
    # 加载模型
    model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
    model.eval()
    
    # 准备两张图像
    try:
        import requests
        from io import BytesIO
        
        url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        url2 = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 同一张图
        
        response1 = requests.get(url1, stream=True)
        image1 = Image.open(BytesIO(response1.content)).convert("RGB")
        
        response2 = requests.get(url2, stream=True)
        image2 = Image.open(BytesIO(response2.content)).convert("RGB")
        
        transform = make_image_transform(224)
        img1_tensor = transform(image1).unsqueeze(0)
        img2_tensor = transform(image2).unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            features1 = model.forward_features(img1_tensor)
            features2 = model.forward_features(img2_tensor)
            
            feat1 = features1["x_norm_clstoken"].squeeze(0)
            feat2 = features2["x_norm_clstoken"].squeeze(0)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                feat1.unsqueeze(0),
                feat2.unsqueeze(0)
            ).item()
            
            print(f"图像相似度: {similarity:.4f}")
            print(f"(同一张图像应该接近1.0)")
            
    except Exception as e:
        print(f"无法下载图像: {e}")
        print("请使用本地图像路径")


def example_5_list_available_models():
    """示例5: 列出所有可用的模型"""
    print("\n" + "="*50)
    print("示例5: 可用的DINOv3模型")
    print("="*50)
    
    models = {
        "Backbone模型": [
            "dinov3_vits16      - ViT-Small/16 (21M参数)",
            "dinov3_vits16plus  - ViT-Small+/16 (29M参数)",
            "dinov3_vitb16      - ViT-Base/16 (86M参数)",
            "dinov3_vitl16      - ViT-Large/16 (300M参数)",
            "dinov3_vitl16plus  - ViT-Large+/16",
            "dinov3_vith16plus  - ViT-Huge+/16 (840M参数)",
            "dinov3_vit7b16     - ViT-7B/16 (6.7B参数)",
            "dinov3_convnext_tiny",
            "dinov3_convnext_small",
            "dinov3_convnext_base",
            "dinov3_convnext_large",
        ],
        "任务特定模型": [
            "dinov3_vit7b16_lc  - 图像分类 (ImageNet-1k)",
            "dinov3_vit7b16_ms  - 语义分割 (ADE20K)",
            "dinov3_vit7b16_de  - 目标检测 (COCO2017)",
            "dinov3_vit7b16_dd  - 深度估计 (SYNTHMIX)",
            "dinov3_vitl16_dinotxt_tet1280d20h24l - 图文理解",
        ]
    }
    
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  - {model}")


def main():
    """主函数"""
    print("="*50)
    print("DINOv3 模型使用示例")
    print("="*50)
    
    # 检查CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 检查是否有本地权重文件或使用 Hugging Face
    weights_path = None
    use_huggingface = False
    
    # 方法1: 检查环境变量
    if os.getenv("DINOV3_WEIGHTS_PATH"):
        weights_path = os.getenv("DINOV3_WEIGHTS_PATH")
        print(f"使用环境变量指定的权重路径: {weights_path}")
    
    # 方法2: 检查常见路径
    common_paths = [
        "./dinov3_vitb16_pretrain_lvd1689m.pth",
        "./weights/dinov3_vitb16_pretrain_lvd1689m.pth",
        "~/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m.pth",
    ]
    for path in common_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            weights_path = expanded_path
            print(f"找到本地权重文件: {weights_path}")
            break
    
    # 方法3: 如果没有本地权重，尝试使用 Hugging Face
    if not weights_path:
        print("\n提示: 如果网络下载失败，可以：")
        print("1. 手动下载权重: wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m.pth")
        print("2. 使用 Hugging Face: 设置环境变量 USE_HUGGINGFACE=1")
        if os.getenv("USE_HUGGINGFACE", "0") == "1":
            use_huggingface = True
            print("使用 Hugging Face 加载模型...")
    
    # 运行示例
    try:
        example_1_feature_extraction(weights_path=weights_path, use_huggingface=use_huggingface)
        example_2_image_classification()
        example_3_custom_classifier()
        example_4_feature_similarity()
        example_5_list_available_models()
        
        print("\n" + "="*50)
        print("所有示例运行完成!")
        print("="*50)
        print("\n更多信息请参考:")
        print("  - DINOV3_USAGE_GUIDE.md (详细使用指南)")
        print("  - README.md (官方文档)")
        print("  - notebooks/ (Jupyter notebook示例)")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

