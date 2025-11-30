"""
KG专用数据collator,用于时序知识图谱补全任务
支持:
1. 实体边界保护 - 不在实体token内部进行mask
2. 负采样 - corrupt head/tail进行对比学习
3. 时间感知mask - 考虑时间戳信息
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
import torch
import random
from collections import defaultdict


@dataclass
class KGQuadCollator:
    """
    KG四元组专用collator,用于时序知识图谱补全
    
    数据格式假设:
    - input_ids: tokenized的KG quads序列
    - quad_boundaries: 每个quad的边界 [(start, end), ...]
    - entity_positions: 实体在序列中的位置 [(start, end, type), ...] type: 'head'/'tail'
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mode: str = "train",  # train/eval
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        # KG特定参数
        mask_entity_prob: float = 0.15,  # mask实体的概率
        mask_relation_prob: float = 0.10,  # mask关系的概率
        mask_time_prob: float = 0.05,  # mask时间的概率
        negative_sample_ratio: float = 0.3,  # 负采样比例
        corrupt_head_prob: float = 0.5,  # corrupt head vs tail的概率
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        self.mask_entity_prob = mask_entity_prob
        self.mask_relation_prob = mask_relation_prob
        self.mask_time_prob = mask_time_prob
        self.negative_sample_ratio = negative_sample_ratio
        self.corrupt_head_prob = corrupt_head_prob
        
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
    
    def _parse_quad_structure(self, input_ids: List[int], tokenizer) -> Dict:
        """
        解析tokenized序列中的quad结构
        假设格式: h_tokens r_tokens t_tokens time_tokens ||| ...
        
        返回:
        - entity_mask: [seq_len] bool数组,标记哪些位置是实体token
        - quad_boundaries: [(start, end), ...] 每个quad的边界
        - element_positions: [{'type': 'head'/'relation'/'tail'/'time', 'start': int, 'end': int}, ...]
        """
        # 解码回文本以定位分隔符
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # 简化实现: 创建基础的entity_mask
        # 假设每个quad格式固定: "h\tr\tt\ttime"
        # 实际需要根据tokenizer的行为精确定位
        
        seq_len = len(input_ids)
        entity_mask = [0] * seq_len
        element_positions = []
        
        # 尝试通过特殊token定位quad边界
        sep_token = "|||"
        tab_token = "\t"
        
        tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]
        
        # 简单启发式: 标记所有非特殊token位置
        special_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        }
        
        # 更精确的方法需要在数据预处理时保存边界信息
        # 这里提供基础实现
        current_pos = 0
        quad_count = 0
        
        # 粗略估计: 每个quad约占固定长度
        # 实际应该在数据加载时提供边界标注
        
        return {
            "entity_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "quad_boundaries": [],
            "element_positions": element_positions,
        }
    
    def _create_kg_span_mask(self, features: List[Dict[str, Any]]) -> torch.Tensor:
        """
        创建KG感知的span mask
        - 保护实体边界,只mask完整的实体
        - 按概率mask head/relation/tail/time
        """
        batch_size = len(features)
        max_len = max(len(f["input_ids"]) for f in features)
        
        span_masks = []
        
        for feature in features:
            seq_len = len(feature["input_ids"])
            span_mask = [False] * seq_len
            
            # 如果提供了entity_mask,使用它来保护实体边界
            if "entity_mask" in feature:
                entity_positions = feature["entity_mask"]
                
                # 随机选择要mask的实体
                if self.mode == "train" and self.rng.random() < self.mask_entity_prob:
                    # 简化: 随机mask一些位置,但避开特殊token
                    for i in range(seq_len):
                        if entity_positions[i] == 1 and self.rng.random() < 0.3:
                            span_mask[i] = True
            else:
                # 降级方案: 使用标准span masking
                # 但避开开头和结尾的特殊token
                mask_start = 1 if seq_len > 2 else 0
                mask_end = seq_len - 1 if seq_len > 2 else seq_len
                
                if mask_end > mask_start:
                    num_to_mask = int((mask_end - mask_start) * self.mask_entity_prob)
                    if num_to_mask > 0:
                        mask_positions = self.rng.choice(
                            range(mask_start, mask_end),
                            size=min(num_to_mask, mask_end - mask_start),
                            replace=False
                        )
                        for pos in mask_positions:
                            span_mask[int(pos)] = True
            
            # Pad to max_len
            span_mask = span_mask + [False] * (max_len - seq_len)
            span_masks.append(span_mask)
        
        return torch.tensor(span_masks, dtype=torch.bool)
    
    def _add_negative_samples(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        添加负样本 (corrupt head or tail)
        仅在训练时使用
        """
        if self.mode != "train" or self.negative_sample_ratio <= 0:
            return features
        
        # 简化实现: 暂不添加负样本
        # 完整实现需要:
        # 1. 构建实体集合
        # 2. 随机替换head或tail
        # 3. 添加is_negative标签
        
        return features
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理一个batch的数据
        
        输入features格式:
        - input_ids: List[int]
        - entity_mask (可选): List[int], 1表示实体token, 0表示其他
        
        输出batch格式:
        - input_ids: [batch, seq_len]
        - span_mask: [batch, seq_len] - 标记哪些位置需要预测
        - entity_mask: [batch, seq_len] - 标记实体位置,用于noise scheduler
        """
        
        # 移除attention_mask (diffusion模型不需要)
        for f in features:
            f.pop("attention_mask", None)
        
        # Tokenizer padding
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # 移除attention_mask
        if "attention_mask" in batch:
            del batch["attention_mask"]
        
        # 创建KG感知的span mask
        span_mask = self._create_kg_span_mask(features)
        batch["span_mask"] = span_mask
        
        # 处理entity_mask
        if len(features) > 0 and "entity_mask" in features[0]:
            max_len = batch["input_ids"].shape[1]
            padded_entity_masks = []
            
            for f in features:
                entity_mask = f.get("entity_mask", [])
                
                # 转换为列表
                if isinstance(entity_mask, torch.Tensor):
                    entity_mask = entity_mask.tolist()
                elif isinstance(entity_mask, np.ndarray):
                    entity_mask = entity_mask.tolist()
                
                # Pad到max_len
                entity_mask = (entity_mask + [0] * max(0, max_len - len(entity_mask)))[:max_len]
                padded_entity_masks.append(entity_mask)
            
            batch["entity_mask"] = torch.tensor(padded_entity_masks, dtype=torch.long)
        else:
            # 如果没有提供entity_mask,创建全0的mask (不保护任何位置)
            batch["entity_mask"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        
        return batch


@dataclass 
class KGQuadCollatorForEval:
    """
    评估时使用的简化collator
    不进行mask和负采样,只做padding
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 移除attention_mask
        for f in features:
            f.pop("attention_mask", None)
        
        # Padding
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        if "attention_mask" in batch:
            del batch["attention_mask"]
        
        # 对于eval,创建一个覆盖后半段的span_mask (用于prefix LM风格评估)
        seq_len = batch["input_ids"].shape[1]
        batch_size = batch["input_ids"].shape[0]
        
        # 简单策略: mask后50%
        span_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        mid_point = seq_len // 2
        span_mask[:, mid_point:] = True
        batch["span_mask"] = span_mask
        
        # Entity mask (如果提供)
        if len(features) > 0 and "entity_mask" in features[0]:
            max_len = seq_len
            padded_entity_masks = []
            
            for f in features:
                entity_mask = f.get("entity_mask", [])
                if isinstance(entity_mask, torch.Tensor):
                    entity_mask = entity_mask.tolist()
                elif isinstance(entity_mask, np.ndarray):
                    entity_mask = entity_mask.tolist()
                
                entity_mask = (entity_mask + [0] * max(0, max_len - len(entity_mask)))[:max_len]
                padded_entity_masks.append(entity_mask)
            
            batch["entity_mask"] = torch.tensor(padded_entity_masks, dtype=torch.long)
        else:
            batch["entity_mask"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        
        return batch
