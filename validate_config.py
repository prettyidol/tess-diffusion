#!/usr/bin/env python3
"""
验证脚本 - 检查训练和评测配置的一致性
包括: 实体tokenization、self-conditioning、模型配置等
"""

import argparse
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
import torch


def check_entity_tokenization(tokenizer, sample_entities: list):
    """
    检查实体是否被正确tokenize(不分词)
    """
    print("\n" + "=" * 80)
    print("检查实体 Tokenization")
    print("=" * 80)
    
    split_entities = []
    ok_entities = []
    
    for entity in sample_entities:
        tokens = tokenizer.tokenize(entity)
        if len(tokens) > 1:
            split_entities.append((entity, tokens))
        else:
            ok_entities.append((entity, tokens))
    
    print(f"\n总实体数: {len(sample_entities)}")
    print(f"✓ 不分词实体: {len(ok_entities)}")
    print(f"✗ 分词实体: {len(split_entities)}")
    
    if ok_entities:
        print(f"\n示例 (不分词实体, 前5个):")
        for entity, tokens in ok_entities[:5]:
            print(f"  '{entity}' → {tokens}")
    
    if split_entities:
        print(f"\n⚠️ 警告: 发现{len(split_entities)}个实体被分词!")
        print(f"示例 (分词实体, 前10个):")
        for entity, tokens in split_entities[:10]:
            print(f"  '{entity}' → {tokens} ({len(tokens)} tokens)")
        print(f"\n建议: 运行 extend_tokenizer_vocab.py 扩展词汇表")
    
    return len(split_entities) == 0


def check_model_config(checkpoint_dir: str):
    """
    检查模型配置
    """
    print("\n" + "=" * 80)
    print("检查模型配置")
    print("=" * 80)
    
    try:
        config = AutoConfig.from_pretrained(checkpoint_dir)
        
        print(f"\n模型类型: {config.model_type}")
        print(f"词汇表大小: {config.vocab_size}")
        print(f"隐藏层大小: {config.hidden_size}")
        print(f"注意力头数: {config.num_attention_heads}")
        print(f"层数: {config.num_hidden_layers}")
        print(f"最大位置编码: {config.max_position_embeddings}")
        
        # 检查self-conditioning配置
        if hasattr(config, 'self_condition'):
            print(f"\n✓ Self-conditioning: {config.self_condition}")
            if hasattr(config, 'self_condition_zeros_after_softmax'):
                print(f"  - zeros_after_softmax: {config.self_condition_zeros_after_softmax}")
            if hasattr(config, 'self_condition_mlp_projection'):
                print(f"  - mlp_projection: {config.self_condition_mlp_projection}")
        else:
            print(f"\n⚠️ Self-conditioning: 未配置")
        
        # 检查simplex配置
        if hasattr(config, 'simplex_value'):
            print(f"\n✓ Simplex value: {config.simplex_value}")
        
        # 检查分类器free guidance
        if hasattr(config, 'classifier_free_simplex_inputs'):
            print(f"✓ Classifier-free guidance: {config.classifier_free_simplex_inputs}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 加载配置失败: {e}")
        return False


def check_training_config(config_path: str):
    """
    检查训练配置文件
    """
    print("\n" + "=" * 80)
    print("检查训练配置")
    print("=" * 80)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n配置文件: {config_path}")
        
        # 关键参数
        key_params = [
            'model_name_or_path',
            'tokenizer_name',
            'simplex_value',
            'num_diffusion_steps',
            'num_inference_diffusion_steps',
            'beta_schedule',
            'self_condition',
            'self_condition_zeros_after_softmax',
            'max_seq_length',
            'per_device_train_batch_size',
            'learning_rate',
            'fp16',
        ]
        
        print("\n关键参数:")
        for param in key_params:
            value = config.get(param, "未设置")
            print(f"  {param}: {value}")
        
        # 检查潜在问题
        warnings = []
        
        if config.get('tokenizer_name') in [None, 'roberta-base']:
            warnings.append("tokenizer_name未指定扩展词汇表路径")
        
        if config.get('self_condition') is None:
            warnings.append("self_condition未启用")
        
        if config.get('num_diffusion_steps', 0) < 100:
            warnings.append(f"num_diffusion_steps过小: {config.get('num_diffusion_steps')}")
        
        if config.get('simplex_value', 0) < 1:
            warnings.append(f"simplex_value过小: {config.get('simplex_value')}")
        
        if warnings:
            print("\n⚠️ 警告:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("\n✓ 配置检查通过")
        
        return len(warnings) == 0
        
    except Exception as e:
        print(f"\n✗ 加载配置失败: {e}")
        return False


def check_checkpoint_completeness(checkpoint_dir: str):
    """
    检查checkpoint是否完整
    """
    print("\n" + "=" * 80)
    print("检查 Checkpoint 完整性")
    print("=" * 80)
    
    checkpoint_path = Path(checkpoint_dir)
    
    required_files = [
        'config.json',
        'pytorch_model.bin',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
    ]
    
    optional_files = [
        'trainer_state.json',
        'optimizer.pt',
        'scheduler.pt',
        'training_args.bin',
    ]
    
    print(f"\nCheckpoint 目录: {checkpoint_path}")
    
    missing_required = []
    found_optional = []
    missing_optional = []
    
    # 检查必需文件
    print("\n必需文件:")
    for file in required_files:
        file_path = checkpoint_path / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing_required.append(file)
    
    # 检查可选文件
    print("\n可选文件:")
    for file in optional_files:
        file_path = checkpoint_path / file
        if file_path.exists():
            print(f"  ✓ {file}")
            found_optional.append(file)
        else:
            print(f"  - {file} (不存在)")
            missing_optional.append(file)
    
    if missing_required:
        print(f"\n✗ 缺少{len(missing_required)}个必需文件!")
        return False
    else:
        print(f"\n✓ 所有必需文件存在")
        
        if 'optimizer.pt' in missing_optional or 'scheduler.pt' in missing_optional:
            print(f"ℹ️  提示: 缺少optimizer/scheduler状态,无法完全恢复训练")
        
        return True


def extract_sample_entities(train_file: str, n: int = 100):
    """
    从训练文件中提取样本实体
    """
    entities = set()
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1000:  # 只读前1000行
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # 格式: "h\tr\tt\ttime ||| h2\tr2\tt2\ttime2 ||| ..."
                quads = line.split(' ||| ')
                
                for quad in quads:
                    parts = quad.split('\t')
                    if len(parts) == 4:
                        head, relation, tail, time = parts
                        entities.add(head.strip())
                        entities.add(tail.strip())
                    
                    if len(entities) >= n:
                        break
                
                if len(entities) >= n:
                    break
        
        return list(entities)[:n]
        
    except Exception as e:
        print(f"⚠️ 读取训练文件失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="验证TESS训练和评测配置")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint目录路径")
    parser.add_argument("--config", type=str, default="configs/tess_gpu_oneline_sc.json",
                        help="训练配置文件路径")
    parser.add_argument("--train_file", type=str, default="tess_train1_oneline.txt",
                        help="训练数据文件")
    parser.add_argument("--check_tokenization", action="store_true",
                        help="检查实体tokenization")
    parser.add_argument("--num_sample_entities", type=int, default=100,
                        help="检查tokenization时的样本实体数量")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESS 配置验证")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"训练配置: {args.config}")
    print(f"训练文件: {args.train_file}")
    print("=" * 80)
    
    all_passed = True
    
    # 1. 检查checkpoint完整性
    if not check_checkpoint_completeness(args.checkpoint):
        all_passed = False
    
    # 2. 检查模型配置
    if not check_model_config(args.checkpoint):
        all_passed = False
    
    # 3. 检查训练配置
    if Path(args.config).exists():
        if not check_training_config(args.config):
            all_passed = False
    else:
        print(f"\n⚠️ 训练配置文件不存在: {args.config}")
        all_passed = False
    
    # 4. 检查实体tokenization (可选)
    if args.check_tokenization:
        print(f"\n从训练文件提取{args.num_sample_entities}个样本实体...")
        sample_entities = extract_sample_entities(args.train_file, args.num_sample_entities)
        
        if sample_entities:
            print(f"提取了{len(sample_entities)}个实体")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
            
            if not check_entity_tokenization(tokenizer, sample_entities):
                all_passed = False
        else:
            print("⚠️ 未能提取样本实体")
    
    # 总结
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有检查通过")
        print("=" * 80)
        sys.exit(0)
    else:
        print("✗ 发现问题,请检查上述警告")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
