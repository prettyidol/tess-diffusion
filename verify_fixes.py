#!/usr/bin/env python3
"""
系统修复验证脚本 - 检查所有关键修复是否已正确应用
"""

import os
import sys
import json
from pathlib import Path


def check_file_contains(file_path, search_strings):
    """检查文件是否包含指定的字符串"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {}
        for desc, search_str in search_strings.items():
            results[desc] = search_str in content
        
        return results
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")
        return None


def main():
    print("=" * 80)
    print("TESS Diffusion 系统修复验证")
    print("=" * 80)
    
    checks = {
        "修复1: KGQuadCollator导入": {
            "file": "run_mlm.py",
            "checks": {
                "KGQuadCollator导入": "from sdlm.data.kg_quad_collator import KGQuadCollator",
                "KGQuadCollatorForEval导入": "from sdlm.data.kg_quad_collator import KGQuadCollator, KGQuadCollatorForEval",
                "创建collator函数": "def create_data_collator(mode):",
                "条件选择": 'if data_args.conditional_generation and data_args.conditional_generation in ['
            }
        },
        "修复2: eval参数更新": {
            "file": "eval_kg_ranking.py",
            "checks": {
                "tess_t_eval默认值=60": 'ap.add_argument("--tess_t_eval", type=int, default=60',
                "tess_num_steps默认值=500": 'ap.add_argument("--tess_num_steps", type=int, default=500',
                "neg_k默认值=128": 'ap.add_argument("--neg_k", type=int, default=128',
            }
        },
    }
    
    all_passed = True
    
    for check_name, check_info in checks.items():
        print(f"\n{'='*80}")
        print(f"检查: {check_name}")
        print(f"文件: {check_info['file']}")
        print('='*80)
        
        file_path = Path(check_info['file'])
        
        if not file_path.exists():
            print(f"❌ 文件不存在: {check_info['file']}")
            all_passed = False
            continue
        
        results = check_file_contains(str(file_path), check_info['checks'])
        
        if results is None:
            all_passed = False
            continue
        
        for desc, found in results.items():
            if found:
                print(f"  ✅ {desc}")
            else:
                print(f"  ❌ {desc}")
                all_passed = False
    
    # 检查collator文件
    print(f"\n{'='*80}")
    print("检查: KGQuadCollator文件")
    print('='*80)
    
    collator_path = Path("sdlm/data/kg_quad_collator.py")
    if collator_path.exists():
        print(f"  ✅ 文件存在: {collator_path}")
        
        results = check_file_contains(str(collator_path), {
            "KGQuadCollator类": "class KGQuadCollator:",
            "KGQuadCollatorForEval类": "class KGQuadCollatorForEval:",
            "__call__方法": "def __call__(self, features: List[Dict[str, Any]])",
        })
        
        for desc, found in results.items():
            if found:
                print(f"  ✅ {desc}")
            else:
                print(f"  ❌ {desc}")
                all_passed = False
    else:
        print(f"  ❌ 文件不存在: {collator_path}")
        all_passed = False
    
    # 检查配置文件
    print(f"\n{'='*80}")
    print("检查: 训练配置")
    print('='*80)
    
    config_path = Path("configs/tess_gpu_oneline_sc.json")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config_checks = {
                "simplex_value=5": config.get("simplex_value") == 5,
                "num_diffusion_steps=500": config.get("num_diffusion_steps") == 500,
                "num_inference_diffusion_steps=100": config.get("num_inference_diffusion_steps") == 100,
                "self_condition=logits_addition": config.get("self_condition") == "logits_addition",
                "beta_schedule=squaredcos_improved_ddpm": config.get("beta_schedule") == "squaredcos_improved_ddpm",
            }
            
            for desc, result in config_checks.items():
                if result:
                    print(f"  ✅ {desc}")
                else:
                    print(f"  ⚠️ {desc} (可选)")
            
        except Exception as e:
            print(f"  ❌ 配置文件格式错误: {e}")
            all_passed = False
    else:
        print(f"  ❌ 配置文件不存在: {config_path}")
        all_passed = False
    
    # 检查脚本文件
    print(f"\n{'='*80}")
    print("检查: 辅助脚本")
    print('='*80)
    
    scripts = [
        "extend_tokenizer_vocab.py",
        "validate_config.py",
        "run_optimized_eval.py",
        "quick_start_fix.py",
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}")
            all_passed = False
    
    # 检查文档
    print(f"\n{'='*80}")
    print("检查: 文档文件")
    print('='*80)
    
    docs = [
        "FIXES_AND_IMPROVEMENTS.md",
        "SUMMARY_OF_FIXES.md",
        "COLAB_TRAINING_GUIDE.md",
        "SYSTEM_CHECK_REPORT.md",
    ]
    
    for doc in docs:
        if Path(doc).exists():
            print(f"  ✅ {doc}")
        else:
            print(f"  ⚠️ {doc} (可选)")
    
    # 总结
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ 所有关键修复已应用!")
        print("="*80)
        print("\n后续步骤:")
        print("1. 扩展tokenizer: python extend_tokenizer_vocab.py ...")
        print("2. 训练模型: python run_mlm.py configs/tess_gpu_oneline_sc.json")
        print("3. 评测: python run_optimized_eval.py --checkpoint outputs/checkpoint-XXXX --quick")
        print("="*80)
        return 0
    else:
        print("❌ 发现未完成的修复!")
        print("请检查上述❌标记的项目")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
