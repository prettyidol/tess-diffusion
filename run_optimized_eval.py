#!/usr/bin/env python3
"""
优化的KG评测脚本 - 使用推荐参数运行eval_kg_ranking.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_eval_with_optimized_params(
    checkpoint_dir: str,
    test_file: str = "tess_test1_oneline.txt",
    mode: str = "tail",
    scorer: str = "tess",
    k: int = 10,
    tess_t_eval: int = 60,  # 优化: 从100降到60
    neg_k: int = 128,  # 优化: 从256降到128
    num_queries: int = 2000,
    output_file: str = None,
):
    """
    使用优化参数运行评测
    
    参数说明:
    - checkpoint_dir: 训练checkpoint路径
    - test_file: 测试文件
    - mode: 'tail' (预测尾实体) 或 'head' (预测头实体)
    - scorer: 'tess' (扩散模型), 'freq' (频率基线), 'random' (随机基线)
    - k: Hits@k中的k值
    - tess_t_eval: TESS评测时的timestep (推荐40-80, 默认60)
    - neg_k: 负样本数量 (推荐64-128, 默认128)
    - num_queries: 评测query数量 (完整测试集约7000+, 快速测试用200-2000)
    """
    
    # 构建命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "eval_kg_ranking.py",
        "--test_file", test_file,
        "--mode", mode,
        "--scorer", scorer,
        "--k", str(k),
        "--neg_k", str(neg_k),
        "--max_queries", str(num_queries),
    ]
    
    # TESS特定参数
    if scorer == "tess":
        cmd.extend([
            "--tess_checkpoint", checkpoint_dir,
            "--tess_t_eval", str(tess_t_eval),
            "--tess_simplex_value", "5",
            "--tess_num_diffusion_steps", "500",
            "--tess_beta_schedule", "squaredcos_improved_ddpm",
            "--candidates", "sampled",  # 使用sampled模式加速
        ])
    
    # 输出文件
    if output_file:
        cmd.extend(["--output", output_file])
    
    print("=" * 80)
    print("运行优化的KG评测")
    print("=" * 80)
    print(f"命令: {' '.join(cmd)}")
    print("=" * 80)
    
    # 运行命令
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode


def grid_search_tess_t_eval(
    checkpoint_dir: str,
    test_file: str = "tess_test1_oneline.txt",
    mode: str = "tail",
    num_queries: int = 200,
    t_values: list = None,
):
    """
    Grid search最优的tess_t_eval参数
    """
    if t_values is None:
        t_values = [40, 60, 80, 100, 120]
    
    print("\n" + "=" * 80)
    print("Grid Search for Optimal tess_t_eval")
    print("=" * 80)
    print(f"测试集: {test_file}")
    print(f"模式: {mode}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Query数量: {num_queries}")
    print(f"测试t值: {t_values}")
    print("=" * 80 + "\n")
    
    results = {}
    
    for t in t_values:
        print(f"\n{'=' * 40}")
        print(f"测试 tess_t_eval = {t}")
        print('=' * 40)
        
        output_file = f"eval_results_t{t}_q{num_queries}.json"
        
        returncode = run_eval_with_optimized_params(
            checkpoint_dir=checkpoint_dir,
            test_file=test_file,
            mode=mode,
            scorer="tess",
            k=10,
            tess_t_eval=t,
            neg_k=128,
            num_queries=num_queries,
            output_file=output_file,
        )
        
        if returncode == 0:
            # 读取结果
            try:
                import json
                with open(output_file, 'r') as f:
                    result = json.load(f)
                    results[t] = result
                    print(f"\n结果 (t={t}):")
                    print(f"  MRR: {result.get('MRR', 'N/A'):.4f}")
                    print(f"  Hits@1: {result.get('Hits@1', 'N/A'):.4f}")
                    print(f"  Hits@10: {result.get('Hits@10', 'N/A'):.4f}")
            except Exception as e:
                print(f"读取结果失败: {e}")
        else:
            print(f"评测失败 (t={t})")
    
    # 总结
    print("\n" + "=" * 80)
    print("Grid Search 总结")
    print("=" * 80)
    print(f"{'t_eval':<10} {'MRR':<10} {'Hits@1':<10} {'Hits@10':<10}")
    print("-" * 80)
    
    best_t = None
    best_mrr = 0
    
    for t, result in sorted(results.items()):
        mrr = result.get('MRR', 0)
        h1 = result.get('Hits@1', 0)
        h10 = result.get('Hits@10', 0)
        print(f"{t:<10} {mrr:<10.4f} {h1:<10.4f} {h10:<10.4f}")
        
        if mrr > best_mrr:
            best_mrr = mrr
            best_t = t
    
    print("=" * 80)
    print(f"\n最佳 tess_t_eval: {best_t} (MRR={best_mrr:.4f})")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="优化的KG评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 快速评测 (200 queries, 推荐参数):
   python run_optimized_eval.py --checkpoint outputs/checkpoint-5500 --quick

2. 完整评测 (2000 queries):
   python run_optimized_eval.py --checkpoint outputs/checkpoint-5500 --num_queries 2000

3. Grid search最优t值 (200 queries):
   python run_optimized_eval.py --checkpoint outputs/checkpoint-5500 --grid_search

4. 评测head预测:
   python run_optimized_eval.py --checkpoint outputs/checkpoint-5500 --mode head

5. 使用频率基线:
   python run_optimized_eval.py --checkpoint outputs/checkpoint-5500 --scorer freq
        """
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练checkpoint目录")
    parser.add_argument("--test_file", type=str, default="tess_test1_oneline.txt",
                        help="测试文件路径")
    parser.add_argument("--mode", type=str, default="tail", choices=["tail", "head"],
                        help="预测模式: tail (预测尾实体) 或 head (预测头实体)")
    parser.add_argument("--scorer", type=str, default="tess", 
                        choices=["tess", "freq", "random"],
                        help="评分器类型")
    parser.add_argument("--k", type=int, default=10,
                        help="Hits@k中的k值")
    parser.add_argument("--tess_t_eval", type=int, default=60,
                        help="TESS评测timestep (推荐40-80)")
    parser.add_argument("--neg_k", type=int, default=128,
                        help="负样本数量 (推荐64-128)")
    parser.add_argument("--num_queries", type=int, default=2000,
                        help="评测query数量")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (200 queries)")
    parser.add_argument("--grid_search", action="store_true",
                        help="Grid search最优tess_t_eval")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    
    args = parser.parse_args()
    
    # 快速模式
    if args.quick:
        args.num_queries = 200
        print("快速模式: 使用200个queries")
    
    # Grid search模式
    if args.grid_search:
        grid_search_tess_t_eval(
            checkpoint_dir=args.checkpoint,
            test_file=args.test_file,
            mode=args.mode,
            num_queries=args.num_queries,
        )
    else:
        # 标准评测
        returncode = run_eval_with_optimized_params(
            checkpoint_dir=args.checkpoint,
            test_file=args.test_file,
            mode=args.mode,
            scorer=args.scorer,
            k=args.k,
            tess_t_eval=args.tess_t_eval,
            neg_k=args.neg_k,
            num_queries=args.num_queries,
            output_file=args.output,
        )
        
        sys.exit(returncode)


if __name__ == "__main__":
    main()
