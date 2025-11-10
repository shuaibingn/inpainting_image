"""
测试环境安装是否正确

运行此脚本以验证所有依赖是否正确安装。

使用方法：
python test_installation.py
"""

import sys


def test_imports():
    """测试所有必要的库是否可以导入"""
    print("=" * 60)
    print("测试依赖库导入...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'diffusers': 'Diffusers',
        'transformers': 'Transformers',
        'accelerate': 'Accelerate',
        'pycocotools': 'COCO Tools',
        'tqdm': 'tqdm',
        'cv2': 'OpenCV',
        'einops': 'Einops',
    }
    
    failed_imports = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - 已安装")
        except ImportError:
            print(f"✗ {name:20s} - 未安装")
            failed_imports.append(name)
    
    print()
    
    if failed_imports:
        print(f"❌ 有 {len(failed_imports)} 个包未安装:")
        for pkg in failed_imports:
            print(f"   - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    else:
        print("✓ 所有依赖已正确安装")
        return True


def test_pytorch():
    """测试 PyTorch 配置"""
    print("\n" + "=" * 60)
    print("测试 PyTorch 配置...")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 测试 GPU 计算
        try:
            x = torch.randn(3, 3).cuda()
            y = x @ x.T
            print("✓ GPU 计算测试通过")
        except Exception as e:
            print(f"✗ GPU 计算测试失败: {e}")
            return False
    else:
        print("⚠️  警告: CUDA 不可用，将使用 CPU 训练（速度会很慢）")
    
    return True


def test_diffusers():
    """测试 Diffusers 库"""
    print("\n" + "=" * 60)
    print("测试 Diffusers 库...")
    print("=" * 60)
    
    try:
        from diffusers import UNet2DConditionModel, DDPMScheduler
        print("✓ Diffusers 核心模块导入成功")
        
        # 测试创建模型（不下载权重）
        print("测试模型初始化...")
        config = {
            "sample_size": 64,
            "in_channels": 4,
            "out_channels": 4,
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
            "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            "block_out_channels": [320, 640],
            "layers_per_block": 2,
            "cross_attention_dim": 768,
        }
        
        unet = UNet2DConditionModel(**config)
        print(f"✓ UNet 初始化成功 (参数量: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M)")
        
        return True
    except Exception as e:
        print(f"✗ Diffusers 测试失败: {e}")
        return False


def test_project_files():
    """测试项目文件是否完整"""
    print("\n" + "=" * 60)
    print("测试项目文件...")
    print("=" * 60)
    
    import os
    
    required_files = [
        'data.py',
        'model.py',
        'train.py',
        'infer.py',
        'requirements.txt',
        'README.md',
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - 未找到")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 有 {len(missing_files)} 个文件缺失")
        return False
    else:
        print("\n✓ 所有项目文件完整")
        return True


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建...")
    print("=" * 60)
    
    try:
        from model import ZeroConv, ControlNetConditioningEmbedding
        import torch
        
        print("测试 ZeroConv...")
        zero_conv = ZeroConv(64, 64)
        x = torch.randn(1, 64, 32, 32)
        y = zero_conv(x)
        assert y.shape == x.shape
        assert torch.allclose(y, torch.zeros_like(y))  # 初始化为零
        print("✓ ZeroConv 测试通过")
        
        print("测试 ControlNetConditioningEmbedding...")
        cond_emb = ControlNetConditioningEmbedding()
        x = torch.randn(1, 4, 512, 512)
        y = cond_emb(x)
        print(f"✓ 条件编码测试通过 (输入: {x.shape} -> 输出: {y.shape})")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("ControlNet Inpainting 环境测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("依赖导入", test_imports()))
    results.append(("PyTorch 配置", test_pytorch()))
    results.append(("Diffusers 库", test_diffusers()))
    results.append(("项目文件", test_project_files()))
    results.append(("模型创建", test_model_creation()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！环境配置正确。")
        print("\n下一步:")
        print("1. 准备 COCO 数据集")
        print("2. 运行 python train.py 开始训练")
        print("3. 查看 README.md 了解详细使用方法")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
        print("\n建议:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 确保 PyTorch 和 CUDA 配置正确")
        print("3. 查看 README.md 了解环境要求")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

