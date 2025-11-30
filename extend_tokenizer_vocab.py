#!/usr/bin/env python3
"""
从KG训练数据中提取所有实体和关系,扩展tokenizer词汇表
避免实体被分词为子词,提高KG completion性能
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer


def extract_kg_elements_from_oneline(file_path):
    """从oneline格式KG文件提取实体和关系"""
    entities = set()
    relations = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 每行格式: "h\tr\tt\ttime ||| h2\tr2\tt2\ttime2 ||| ..."
            quads = line.split(' ||| ')
            
            for quad in quads:
                parts = quad.split('\t')
                if len(parts) != 4:
                    print(f"警告: 行 {line_idx+1} 四元组格式错误: {quad}")
                    continue
                
                head, relation, tail, time = parts
                entities.add(head.strip())
                entities.add(tail.strip())
                relations.add(relation.strip())
    
    return entities, relations


def check_tokenizer_splitting(tokenizer, elements):
    """检查哪些元素会被tokenizer分词"""
    split_elements = {}
    
    for elem in elements:
        tokens = tokenizer.tokenize(elem)
        if len(tokens) > 1:
            split_elements[elem] = tokens
    
    return split_elements


def extend_tokenizer(base_model, entities, relations, output_dir, special_prefix="[KG]"):
    """扩展tokenizer词汇表,添加实体和关系作为单个token"""
    
    # 加载基础tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(f"原始词汇表大小: {len(tokenizer)}")
    
    # 检查哪些元素需要添加
    all_elements = list(entities) + list(relations)
    split_elements = check_tokenizer_splitting(tokenizer, all_elements)
    
    print(f"\n发现 {len(split_elements)}/{len(all_elements)} 个元素会被分词:")
    print(f"  - 实体: {len([e for e in split_elements if e in entities])}")
    print(f"  - 关系: {len([r for r in split_elements if r in relations])}")
    
    # 准备新增词汇
    new_tokens = []
    
    # 方案1: 直接添加原始文本
    for elem in split_elements:
        if elem not in tokenizer.get_vocab():
            new_tokens.append(elem)
    
    # 方案2: 添加带特殊前缀的版本 (可选,用于明确标识)
    # for elem in split_elements:
    #     prefixed = f"{special_prefix}{elem}"
    #     if prefixed not in tokenizer.get_vocab():
    #         new_tokens.append(prefixed)
    
    print(f"\n将添加 {len(new_tokens)} 个新token到词汇表")
    
    # 添加新token
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    print(f"成功添加 {num_added} 个token")
    print(f"扩展后词汇表大小: {len(tokenizer)}")
    
    # 保存扩展后的tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    print(f"\n已保存扩展tokenizer到: {output_path}")
    
    # 保存统计信息
    stats = {
        "original_vocab_size": len(tokenizer) - num_added,
        "added_tokens": num_added,
        "final_vocab_size": len(tokenizer),
        "total_entities": len(entities),
        "total_relations": len(relations),
        "split_entities": len([e for e in split_elements if e in entities]),
        "split_relations": len([r for r in split_elements if r in relations]),
        "sample_split_entities": list(split_elements.items())[:20]
    }
    
    stats_path = output_path / "vocab_extension_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"已保存统计信息到: {stats_path}")
    
    # 验证扩展效果
    print("\n验证扩展效果:")
    test_samples = list(split_elements.keys())[:5]
    for sample in test_samples:
        tokens_before = split_elements[sample]
        tokens_after = tokenizer.tokenize(sample)
        print(f"  '{sample}':")
        print(f"    扩展前: {tokens_before} ({len(tokens_before)} tokens)")
        print(f"    扩展后: {tokens_after} ({len(tokens_after)} tokens)")
    
    return tokenizer, stats


def main():
    parser = argparse.ArgumentParser(description="扩展tokenizer词汇表以支持KG实体和关系")
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件路径 (oneline格式)")
    parser.add_argument("--base_model", type=str, default="roberta-base",
                        help="基础模型名称")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存扩展tokenizer的目录")
    parser.add_argument("--special_prefix", type=str, default="",
                        help="为KG元素添加特殊前缀 (可选)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KG Tokenizer 词汇表扩展")
    print("=" * 60)
    print(f"训练文件: {args.train_file}")
    print(f"基础模型: {args.base_model}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 提取实体和关系
    print("\n[1/3] 从训练数据提取实体和关系...")
    entities, relations = extract_kg_elements_from_oneline(args.train_file)
    print(f"提取完成:")
    print(f"  - 唯一实体数: {len(entities)}")
    print(f"  - 唯一关系数: {len(relations)}")
    
    # 扩展tokenizer
    print(f"\n[2/3] 扩展tokenizer词汇表...")
    tokenizer, stats = extend_tokenizer(
        args.base_model, 
        entities, 
        relations, 
        args.output_dir,
        args.special_prefix
    )
    
    # 生成使用说明
    print(f"\n[3/3] 生成使用说明...")
    readme = f"""# 扩展Tokenizer使用说明

## 词汇表扩展统计

- **原始词汇表大小**: {stats['original_vocab_size']}
- **新增token数量**: {stats['added_tokens']}
- **最终词汇表大小**: {stats['final_vocab_size']}
- **实体总数**: {stats['total_entities']}
- **关系总数**: {stats['total_relations']}
- **需分词的实体数**: {stats['split_entities']}
- **需分词的关系数**: {stats['split_relations']}

## 在训练中使用

修改 `run_mlm.py` 或配置文件:

```python
# 方法1: 直接指定扩展后的tokenizer路径
tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")

# 方法2: 在配置文件中设置
{{"tokenizer_name": "{args.output_dir}"}}
```

## 模型embedding层调整

由于词汇表扩大,需要调整模型embedding层:

```python
model = AutoModelForMaskedLM.from_pretrained("{args.base_model}")
model.resize_token_embeddings(len(tokenizer))

# 新增token的embedding将随机初始化
# 建议训练时对新token使用较大学习率
```

## 验证

运行以下代码验证实体不再被分词:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")

# 测试实体tokenization
test_entities = {stats['sample_split_entities'][:3] if stats['sample_split_entities'] else []}
for entity, old_tokens in test_entities:
    new_tokens = tokenizer.tokenize(entity)
    print(f"{{entity}}: {{new_tokens}}")
```
"""
    
    readme_path = Path(args.output_dir) / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"已保存使用说明到: {readme_path}")
    
    print("\n" + "=" * 60)
    print("词汇表扩展完成!")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"1. 在训练脚本中使用扩展tokenizer: --tokenizer_name {args.output_dir}")
    print(f"2. 调整模型embedding层: model.resize_token_embeddings(len(tokenizer))")
    print(f"3. 重新训练模型")


if __name__ == "__main__":
    main()
