#!/usr/bin/env python3
"""
使用 Hugging Face 加载 DINOv3 模型的简单示例

这是最简单的使用方式，无需申请权限
"""

import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO

def main():
    print("="*60)
    print("DINOv3 模型使用示例 (Hugging Face)")
    print("="*60)
    
    # 1. 加载模型（首次运行会自动下载）
    print("\n[1/4] 加载模型...")
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    model = AutoModel.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    print("✓ 模型加载完成!")
    
    # 2. 准备图像
    print("\n[2/4] 准备图像...")
    try:
        # 下载示例图像
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url, stream=True)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"✓ 图像加载成功: {image.size}")
    except Exception as e:
        print(f"⚠️  无法下载示例图像: {e}")
        print("请提供本地图像路径")
        return
    
    # 3. 预处理图像
    print("\n[3/4] 预处理图像...")
    inputs = processor(images=image, return_tensors="pt")
    print(f"✓ 输入形状: {inputs['pixel_values'].shape}")
    
    # 4. 提取特征
    print("\n[4/4] 提取特征...")
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取 CLS token（全局特征）
        features = outputs.last_hidden_state
        cls_token = features[:, 0, :]  # 第一个 token 是 CLS token
        print(f"✓ 特征提取完成!")
        print(f"  - 特征形状: {features.shape}")
        print(f"  - CLS token 形状: {cls_token.shape}")
        print(f"  - 特征维度: {cls_token.shape[-1]}")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print("\n模型已缓存，下次使用会更快")
    print(f"缓存位置: ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")


if __name__ == "__main__":
    main()

