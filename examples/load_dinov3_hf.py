#!/usr/bin/env python3
"""
从本地文件夹加载 DINOv3 模型的简单示例

模型应该已经通过 download_dinov3_hf.py 下载到本地
"""

import os
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO

def find_model_directory(base_path):
    """
    自动查找模型目录
    
    Args:
        base_path: 基础路径（可能是模型目录或包含模型目录的父目录）
    
    Returns:
        找到的模型目录路径，如果找不到则返回None
    """
    # 如果直接是模型目录（包含 config.json）
    if os.path.isfile(os.path.join(base_path, "config.json")):
        return base_path
    
    # 尝试查找子目录中的模型
    # ModelScope 通常的结构: models/facebook/dinov3-vitb16-pretrain-lvd1689m/
    for root, dirs, files in os.walk(base_path):
        if "config.json" in files:
            return root
    
    return None

def main(model_path=None):
    """
    从本地路径加载 DINOv3 模型并提取特征
    
    Args:
        model_path: 模型本地路径，如果为None则使用默认路径 ./models
    """
    print("="*60)
    print("DINOv3 模型使用示例 (本地加载)")
    print("="*60)
    
    # 1. 确定模型路径
    if model_path is None:
        model_path = "./models"
    
    # 检查基础路径是否存在
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 路径不存在: {model_path}")
        print("\n请确保:")
        print("  1. 模型已通过 download_dinov3_hf.py 下载")
        print("  2. 或者使用 --model_path 参数指定正确的模型路径")
        print(f"\n例如: python load_dinov3_hf.py --model_path {model_path}")
        return
    
    # 自动查找模型目录
    actual_model_path = find_model_directory(model_path)
    if actual_model_path is None:
        print(f"\n❌ 错误: 在 {model_path} 中找不到模型目录")
        print("\n请确保目录中包含 config.json 文件")
        print("ModelScope 下载的模型通常在: models/facebook/dinov3-vitb16-pretrain-lvd1689m/")
        return
    
    if actual_model_path != model_path:
        print(f"📁 自动找到模型目录: {os.path.relpath(actual_model_path, model_path)}")
    
    # 2. 加载模型（从本地路径）
    print(f"\n[1/4] 从本地路径加载模型...")
    print(f"模型路径: {os.path.abspath(actual_model_path)}")
    try:
        model = AutoModel.from_pretrained(actual_model_path, local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(actual_model_path, local_files_only=True)
        model.eval()
        print("✓ 模型加载完成!")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("\n可能的原因:")
        print("  1. 模型文件不完整，请重新下载")
        print("  2. 模型路径不正确")
        print("  3. 缺少必要的配置文件（config.json, model.safetensors 等）")
        print(f"\n检查目录内容:")
        if os.path.exists(actual_model_path):
            files = os.listdir(actual_model_path)
            print(f"  目录中的文件: {', '.join(files[:10])}")
        return
    
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
    print(f"\n模型路径: {os.path.abspath(actual_model_path)}")
    print("模型已加载，可以直接使用")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从本地路径加载 DINOv3 模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models",
        help="模型本地路径（默认: ./models）"
    )
    
    args = parser.parse_args()
    main(model_path=args.model_path)

