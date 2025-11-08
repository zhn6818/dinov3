#!/usr/bin/env python3
"""
使用 Hugging Face 下载 DINOv3 模型

这是一个简单的脚本，用于下载 DINOv3 模型权重（无需申请权限）
"""

import torch
from transformers import AutoModel, AutoImageProcessor

def download_dinov3_model(model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"):
    """
    下载 DINOv3 模型
    
    Args:
        model_name: Hugging Face 模型名称
                   可选值:
                   - facebook/dinov3-vits16-pretrain-lvd1689m (ViT-S/16, 21M)
                   - facebook/dinov3-vits16plus-pretrain-lvd1689m (ViT-S+/16, 29M)
                   - facebook/dinov3-vitb16-pretrain-lvd1689m (ViT-B/16, 86M) [默认]
                   - facebook/dinov3-vitl16-pretrain-lvd1689m (ViT-L/16, 300M)
                   - facebook/dinov3-vith16plus-pretrain-lvd1689m (ViT-H+/16, 840M)
    """
    print("="*60)
    print("DINOv3 模型下载工具 (使用 Hugging Face)")
    print("="*60)
    print(f"\n模型: {model_name}")
    print("开始下载...")
    
    try:
        # 下载模型（会自动缓存到 ~/.cache/huggingface/）
        print("\n正在下载模型权重...")
        model = AutoModel.from_pretrained(model_name)
        print("✓ 模型下载完成!")
        
        # 下载图像处理器
        print("\n正在下载图像处理器...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        print("✓ 图像处理器下载完成!")
        
        # 显示模型信息
        print("\n" + "="*60)
        print("模型信息:")
        print("="*60)
        print(f"模型类型: {type(model).__name__}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"缓存位置: {model.config._name_or_path}")
        
        # 测试模型
        print("\n测试模型加载...")
        model.eval()
        print("✓ 模型加载成功!")
        
        print("\n" + "="*60)
        print("下载完成！模型已缓存，可以直接使用")
        print("="*60)
        print("\n使用方法:")
        print(f"  from transformers import AutoModel")
        print(f"  model = AutoModel.from_pretrained('{model_name}')")
        
        return model, processor
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 需要安装 transformers: pip install transformers")
        print("3. 磁盘空间不足")
        raise


def main():
    """主函数"""
    import sys
    
    # 可用的模型列表
    available_models = {
        "s16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "s16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "h16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    }
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        model_key = sys.argv[1].lower()
        if model_key in available_models:
            model_name = available_models[model_key]
        else:
            # 如果直接提供了完整的模型名称
            model_name = sys.argv[1]
    else:
        # 默认使用 ViT-B/16
        model_name = available_models["b16"]
        print("提示: 可以使用命令行参数指定模型")
        print("例如: python download_dinov3_hf.py b16")
        print("\n可用模型:")
        for key, name in available_models.items():
            print(f"  {key}: {name}")
        print()
    
    # 下载模型
    download_dinov3_model(model_name)


if __name__ == "__main__":
    main()

