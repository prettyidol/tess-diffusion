#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容性验证脚本 - 检查代码修复是否正确应用
运行此脚本以验证所有兼容性修复
"""

import os
import sys
from pathlib import Path


def check_file_contains(file_path, search_string, description):
    """检查文件是否包含指定字符串"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description}")
                return False
    except Exception as e:
        print(f"❌ {description} - 错误: {e}")
        return False


def check_file_not_contains(file_path, search_string, description):
    """检查文件是否不包含指定字符串"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string not in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description}")
                return False
    except Exception as e:
        print(f"❌ {description} - 错误: {e}")
        return False


def main():
    print("=" * 80)
    print("TESS Diffusion 兼容性验证")
    print("=" * 80)
    
    base_path = Path(__file__).parent
    all_passed = True
    
    # 检查1: AdamW 优化器修复
    print("\n[1/2] 检查 AdamW 优化器修复")
    print("-" * 80)
    
    trainer_path = base_path / "sdlm" / "trainer.py"
    
    # 应该使用 torch.optim.AdamW
    if check_file_contains(
        trainer_path,
        "from torch.optim import AdamW",
        "使用 torch.optim.AdamW (正确)"
    ):
        pass
    else:
        all_passed = False
    
    # 不应该使用 transformers.AdamW
    if check_file_not_contains(
        trainer_path,
        "from transformers import AdamW",
        "已移除 transformers.AdamW (正确)"
    ):
        pass
    else:
        all_passed = False
    
    # 检查2: torch.float32 拼写修复
    print("\n[2/2] 检查 torch.float32 拼写修复")
    print("-" * 80)
    
    scheduler_path = base_path / "sdlm" / "schedulers" / "scheduling_simplex_ddpm.py"
    
    # 应该使用 torch.float32
    if check_file_contains(
        scheduler_path,
        "dtype=torch.float32",
        "使用正确的 torch.float32"
    ):
        pass
    else:
        all_passed = False
    
    # 不应该有拼写错误 torch.torch.float32
    if check_file_not_contains(
        scheduler_path,
        "torch.torch.float32",
        "已修复 torch.torch.float32 拼写错误"
    ):
        pass
    else:
        all_passed = False
    
    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有兼容性检查通过！代码已准备好与 environment.yaml 中的依赖配合使用")
        print("\n下一步:")
        print("1. 安装 Miniconda3")
        print("2. 创建环境: conda env create -f environment.yaml")
        print("3. 激活环境: conda activate sdlm")
        print("4. 安装本项目: pip install -e .")
        print("5. 扩展 tokenizer: python extend_tokenizer_vocab.py \\")
        print("   --train_file tess_train1_oneline.txt \\")
        print("   --base_model roberta-base \\")
        print("   --output_dir extended_tokenizer")
        print("6. 开始训练: python run_mlm.py configs/tess_gpu_oneline_sc.json")
        return 0
    else:
        print("❌ 某些检查失败，请查看上述错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
