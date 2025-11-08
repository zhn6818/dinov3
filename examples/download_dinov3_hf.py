#!/usr/bin/env python3
"""
使用 ModelScope 下载 DINOv3 模型

这是一个简单的脚本，用于下载 DINOv3 模型权重（使用 ModelScope，无需申请权限）
"""

import torch
from modelscope import snapshot_download
from transformers import AutoModel, AutoImageProcessor

def download_dinov3_model(model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", local_dir=None):
    """
    使用 ModelScope 下载 DINOv3 模型
    
    Args:
        model_name: ModelScope 模型名称
                   可选值:
                   - facebook/dinov3-vits16-pretrain-lvd1689m (ViT-S/16, 21M)
                   - facebook/dinov3-vits16plus-pretrain-lvd1689m (ViT-S+/16, 29M)
                   - facebook/dinov3-vitb16-pretrain-lvd1689m (ViT-B/16, 86M) [默认]
                   - facebook/dinov3-vitl16-pretrain-lvd1689m (ViT-L/16, 300M)
                   - facebook/dinov3-vith16plus-pretrain-lvd1689m (ViT-H+/16, 840M)
        local_dir: 本地保存目录，如果为None则使用默认缓存目录
    """
    print("="*60)
    print("DINOv3 模型下载工具 (使用 ModelScope)")
    print("="*60)
    print(f"\n模型: {model_name}")
    
    if local_dir:
        print(f"保存目录: {local_dir}")
    else:
        print("使用默认缓存目录: ~/.cache/modelscope/hub/")
    
    print("\n开始下载...")
    
    try:
        # 使用 ModelScope 下载模型
        print("\n正在下载模型文件...")
        model_dir = snapshot_download(model_name, cache_dir=local_dir)
        print(f"✓ 模型下载完成!")
        print(f"模型保存位置: {model_dir}")
        
        # 从本地目录加载模型
        print("\n正在加载模型...")
        model = AutoModel.from_pretrained(model_dir)
        print("✓ 模型加载完成!")
        
        # 加载图像处理器
        print("\n正在加载图像处理器...")
        processor = AutoImageProcessor.from_pretrained(model_dir)
        print("✓ 图像处理器加载完成!")
        
        # 显示模型信息
        print("\n" + "="*60)
        print("模型信息:")
        print("="*60)
        print(f"模型类型: {type(model).__name__}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"模型路径: {model_dir}")
        
        # 测试模型
        print("\n测试模型...")
        model.eval()
        print("✓ 模型测试成功!")
        
        print("\n" + "="*60)
        print("下载完成！模型已保存，可以直接使用")
        print("="*60)
        print("\n使用方法:")
        print(f"  from transformers import AutoModel, AutoImageProcessor")
        print(f"  model = AutoModel.from_pretrained('{model_dir}')")
        print(f"  processor = AutoImageProcessor.from_pretrained('{model_dir}')")
        
        return model, processor, model_dir
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("\n请先安装 ModelScope:")
        print("  pip install modelscope")
        raise
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 需要安装 modelscope: pip install modelscope")
        print("3. 需要安装 transformers: pip install transformers")
        print("4. 磁盘空间不足")
        raise


def main():
    """主函数"""
    import sys
    import argparse
    
    # 可用的模型列表
    available_models = {
        "s16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "s16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "h16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    }
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用 ModelScope 下载 DINOv3 模型")
    parser.add_argument(
        "model",
        nargs="?",
        default="b16",
        help="模型标识符 (s16, s16plus, b16, l16, h16plus) 或完整的模型名称"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="本地保存目录（可选，默认使用缓存目录）"
    )
    
    args = parser.parse_args()
    
    # 确定模型名称
    model_key = args.model.lower()
    if model_key in available_models:
        model_name = available_models[model_key]
    else:
        # 如果直接提供了完整的模型名称
        model_name = args.model
    
    # 显示帮助信息（如果没有提供模型参数）
    if len(sys.argv) == 1:
        print("提示: 可以使用命令行参数指定模型")
        print("例如: python download_dinov3_hf.py b16")
        print("例如: python download_dinov3_hf.py b16 --local_dir ./models")
        print("\n可用模型:")
        for key, name in available_models.items():
            print(f"  {key}: {name}")
        print()
    
    # 下载模型
    download_dinov3_model(model_name, local_dir=args.local_dir)


if __name__ == "__main__":
    main()

