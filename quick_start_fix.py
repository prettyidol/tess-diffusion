#!/usr/bin/env python3
"""
快速开始脚本 - 自动执行所有修复步骤
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, check=True):
    """运行命令并显示状态"""
    print(f"\n{'='*80}")
    print(f"步骤: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*80)
    
    result = subprocess.run(cmd, check=check)
    
    if result.returncode == 0:
        print(f"✓ {description} 完成")
    else:
        print(f"✗ {description} 失败")
        if check:
            sys.exit(1)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="TESS修复和优化 - 快速开始")
    
    parser.add_argument("--train_file", type=str, default="tess_train1_oneline.txt",
                        help="训练数据文件")
    parser.add_argument("--base_model", type=str, default="roberta-base",
                        help="基础模型")
    parser.add_argument("--extended_tokenizer_dir", type=str, default="extended_tokenizer",
                        help="扩展tokenizer保存目录")
    parser.add_argument("--skip_tokenizer", action="store_true",
                        help="跳过tokenizer扩展(如已完成)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="跳过验证步骤")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="已有checkpoint路径(用于验证)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("TESS Diffusion - 自动修复和优化")
    print("="*80)
    print(f"训练文件: {args.train_file}")
    print(f"基础模型: {args.base_model}")
    print(f"扩展Tokenizer目录: {args.extended_tokenizer_dir}")
    print("="*80)
    
    # 步骤1: 扩展Tokenizer
    if not args.skip_tokenizer:
        if Path(args.extended_tokenizer_dir).exists():
            print(f"\n⚠️ 目录 {args.extended_tokenizer_dir} 已存在")
            response = input("是否覆盖? (y/n): ")
            if response.lower() != 'y':
                print("跳过tokenizer扩展")
                args.skip_tokenizer = True
        
        if not args.skip_tokenizer:
            cmd = [
                sys.executable,
                "extend_tokenizer_vocab.py",
                "--train_file", args.train_file,
                "--base_model", args.base_model,
                "--output_dir", args.extended_tokenizer_dir,
            ]
            
            if not run_command(cmd, "扩展Tokenizer词汇表", check=True):
                print("\n✗ Tokenizer扩展失败,请检查错误信息")
                sys.exit(1)
    else:
        print(f"\n跳过Tokenizer扩展 (使用现有: {args.extended_tokenizer_dir})")
    
    # 步骤2: 验证配置
    if not args.skip_validation:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # 使用扩展的tokenizer作为checkpoint进行基础验证
            checkpoint_path = args.extended_tokenizer_dir
        
        print(f"\n验证Checkpoint: {checkpoint_path}")
        
        cmd = [
            sys.executable,
            "validate_config.py",
            "--checkpoint", checkpoint_path,
            "--config", "configs/tess_gpu_oneline_sc.json",
            "--train_file", args.train_file,
            "--check_tokenization",
            "--num_sample_entities", "100",
        ]
        
        # 验证可以失败,仅作为参考
        run_command(cmd, "验证配置", check=False)
    
    # 步骤3: 显示后续步骤
    print("\n" + "="*80)
    print("✓ 自动修复完成!")
    print("="*80)
    print("\n后续步骤:")
    print("\n1. 更新训练配置:")
    print(f"   编辑 configs/tess_gpu_oneline_sc.json")
    print(f"   设置: \"tokenizer_name\": \"{args.extended_tokenizer_dir}\"")
    
    print("\n2. 启动训练:")
    print(f"   python run_mlm.py \\")
    print(f"       --model_name_or_path {args.base_model} \\")
    print(f"       --tokenizer_name {args.extended_tokenizer_dir} \\")
    print(f"       --train_file {args.train_file} \\")
    print(f"       --output_dir outputs/tess_fixed \\")
    print(f"       --per_device_train_batch_size 8 \\")
    print(f"       --num_train_epochs 3 \\")
    print(f"       --fp16")
    
    print("\n3. 评测:")
    print(f"   python run_optimized_eval.py \\")
    print(f"       --checkpoint outputs/tess_fixed/checkpoint-XXXX \\")
    print(f"       --quick  # 快速测试")
    
    print("\n" + "="*80)
    print("详细说明请参考: FIXES_AND_IMPROVEMENTS.md")
    print("="*80)


if __name__ == "__main__":
    main()
