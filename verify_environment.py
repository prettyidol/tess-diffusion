#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境依赖验证脚本
在安装 environment.yaml 后运行此脚本，验证所有依赖版本是否正确
"""

import sys


def check_import_and_version(module_name, expected_version=None, package_name=None):
    """检查模块导入和版本"""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        if expected_version:
            if version.startswith(expected_version):
                print(f"✅ {module_name}: {version} (期望: {expected_version})")
                return True
            else:
                print(f"⚠️ {module_name}: {version} (期望: {expected_version})")
                return False
        else:
            print(f"✅ {module_name}: {version}")
            return True
    except ImportError as e:
        print(f"❌ {module_name}: 导入失败 - {e}")
        return False


def main():
    print("=" * 80)
    print("TESS Diffusion 环境依赖验证")
    print("=" * 80)
    
    all_passed = True
    
    # 核心依赖检查
    print("\n[1/8] 核心 ML 框架")
    print("-" * 80)
    all_passed &= check_import_and_version("torch", "2.2.0")
    all_passed &= check_import_and_version("torchvision", "0.17.0")
    all_passed &= check_import_and_version("torchaudio", "2.2.0")
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️ CUDA 不可用 (CPU模式)")
    except Exception as e:
        print(f"❌ CUDA 检查失败: {e}")
        all_passed = False
    
    print("\n[2/8] Transformers 生态")
    print("-" * 80)
    all_passed &= check_import_and_version("transformers", "4.33.3")
    all_passed &= check_import_and_version("datasets", "2.14.6")
    all_passed &= check_import_and_version("diffusers", "0.27.2")
    all_passed &= check_import_and_version("accelerate", "0.23.0")
    all_passed &= check_import_and_version("tokenizers", "0.13.3")
    
    print("\n[3/8] 科学计算")
    print("-" * 80)
    all_passed &= check_import_and_version("numpy", "1.23.5")
    all_passed &= check_import_and_version("scipy", "1.9.3")
    all_passed &= check_import_and_version("sklearn", "1.2.2")
    
    print("\n[4/8] NLP 工具")
    print("-" * 80)
    all_passed &= check_import_and_version("sacrebleu", "2.5.1")
    all_passed &= check_import_and_version("nltk", "3.9.2")
    all_passed &= check_import_and_version("sentencepiece", "0.2.1")
    
    print("\n[5/8] 数据处理")
    print("-" * 80)
    all_passed &= check_import_and_version("pyarrow", "10.0.1")
    
    print("\n[6/8] 监控工具")
    print("-" * 80)
    all_passed &= check_import_and_version("tensorboard", "2.17.0")
    all_passed &= check_import_and_version("IPython", "8.4.0")
    
    print("\n[7/8] HuggingFace Hub")
    print("-" * 80)
    all_passed &= check_import_and_version("huggingface_hub", "0.23.5")
    
    print("\n[8/8] 关键 API 测试")
    print("-" * 80)
    
    # 测试 AdamW 导入（新版本）
    try:
        from torch.optim import AdamW
        print("✅ torch.optim.AdamW 导入成功")
    except ImportError as e:
        print(f"❌ torch.optim.AdamW 导入失败: {e}")
        all_passed = False
    
    # 测试 Transformers AutoModel
    try:
        from transformers import AutoTokenizer, AutoModel
        print("✅ transformers.AutoTokenizer 和 AutoModel 导入成功")
    except ImportError as e:
        print(f"❌ transformers 基础模块导入失败: {e}")
        all_passed = False
    
    # 测试 Diffusers
    try:
        from diffusers import DDPMScheduler
        print("✅ diffusers.DDPMScheduler 导入成功")
    except ImportError as e:
        print(f"❌ diffusers 导入失败: {e}")
        all_passed = False
    
    # 测试 SDLM 自定义模块（如果已安装）
    try:
        from sdlm.models import RobertaDiffusionConfig, RobertaForDiffusionLM
        print("✅ SDLM 自定义模型导入成功 (已安装项目)")
    except ImportError:
        print("ℹ️ SDLM 自定义模块未安装 (运行 'pip install -e .' 安装)")
    
    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有环境依赖检查通过！")
        print("\n环境准备就绪，可以开始训练:")
        print("  python run_mlm.py configs/tess_gpu_oneline_sc.json")
    else:
        print("⚠️ 某些依赖版本不匹配或缺失")
        print("\n建议:")
        print("  1. 检查 environment.yaml 是否正确")
        print("  2. 重新创建环境: conda env create -f environment.yaml --force")
        print("  3. 激活环境: conda activate sdlm")
    
    print("=" * 80)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
