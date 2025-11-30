# Google Colab T4 GPU 训练完整指南

## 重要说明: 训练模式解析

### ✅ 当前训练模式: **无监督序列到序列**

**关键理解**:

TESS Diffusion的训练是**无监督的MLM(Masked Language Modeling)**风格,**不区分head/tail**:

1. **训练阶段**:
   - 输入: 完整的KG序列 `"h\tr\tt\ttime ||| h2\tr2\tt2\ttime2"`
   - 目标: 随机mask一些token,学习重建整个序列
   - **不是**仅预测tail,而是学习整个KG的语言表示
   - Collator随机选择位置mask(可以是head、relation、tail或time)

2. **评测阶段**:
   - 可选择模式: `--mode tail` (预测尾实体) 或 `--mode head` (预测头实体)
   - 给定 `(h, r)` 查询,在候选集中排序找最可能的tail
   - 或给定 `(r, t)` 查询,在候选集中排序找最可能的head

**训练不区分head/tail,评测时才选择预测方向**。

---

## 训练时间预估 (T4 GPU)

### 数据规模分析
假设ICEWS数据集:
- 训练集: ~50,000 quads
- 验证集: ~5,000 quads  
- 测试集: ~7,000 quads
- 序列长度: 平均150 tokens

### 时间估算

**单个epoch时间**:
```
训练步数 = 50,000 / batch_size
         = 50,000 / 8 = 6,250 steps

单步时间(T4 FP16) ≈ 0.8-1.2 秒
总时间 = 6,250 × 1.0 秒 ≈ 104 分钟 ≈ 1.7 小时
```

**3个epoch总时间**:
```
训练: 1.7 × 3 = 5.1 小时
验证: 每次约10分钟 × 6次 = 1小时
总计: 约 6-7 小时
```

**加速建议**:
- 使用 `gradient_accumulation_steps=2`,减小batch size到4: ~同样时间
- 减少验证频率 `eval_steps=1000`: 节省30分钟
- 使用1 epoch快速验证: ~2小时

---

## Google Colab 训练步骤

### 步骤1: 环境设置 (5-10分钟)

```python
# ============ Cell 1: 检查GPU ============
!nvidia-smi

# ============ Cell 2: 挂载Google Drive ============
from google.colab import drive
drive.mount('/content/drive')

# 设置工作目录(假设你把代码上传到了这里)
%cd /content/drive/MyDrive/tess-diffusion

# ============ Cell 3: 创建conda环境 ============
# 安装miniconda
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local

# 初始化conda
import sys
sys.path.append('/usr/local/lib/python3.10/site-packages')

# ============ Cell 4: 安装依赖 ============
# 方案A: 使用conda (推荐)
!conda env create -f environment.yaml
!conda init bash

# 方案B: 使用pip (更快)
!pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
!pip install transformers==4.25.1 diffusers==0.7.2 datasets==2.14.6 accelerate==0.12.0
!pip install scipy scikit-learn nltk sacrebleu evaluate bert_score tensorboard

# 验证安装
!python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

### 步骤2: 数据准备 (2-3分钟)

```python
# ============ Cell 5: 检查数据文件 ============
!ls -lh tess_train1_oneline.txt tess_valid1_oneline.txt tess_test1_oneline.txt

# 查看数据格式
!head -n 3 tess_train1_oneline.txt

# ============ Cell 6: 扩展Tokenizer词汇表 ============
!python extend_tokenizer_vocab.py \
    --train_file tess_train1_oneline.txt \
    --base_model roberta-base \
    --output_dir extended_tokenizer

# 验证扩展效果
!python validate_config.py \
    --checkpoint extended_tokenizer \
    --config configs/tess_gpu_oneline_sc.json \
    --train_file tess_train1_oneline.txt \
    --check_tokenization \
    --num_sample_entities 50
```

**预期输出**:
```
原始词汇表大小: 50265
将添加 XXXX 个新token
扩展后词汇表大小: 50265 + XXXX
✓ 实体不再被分词
```

---

### 步骤3: 修改配置文件 (1分钟)

```python
# ============ Cell 7: 更新配置 ============
import json

# 读取配置
with open('configs/tess_gpu_oneline_sc.json', 'r') as f:
    config = json.load(f)

# 修改关键参数
config['tokenizer_name'] = 'extended_tokenizer'  # 使用扩展tokenizer
config['output_dir'] = '/content/drive/MyDrive/tess_outputs'  # 输出到Drive
config['per_device_train_batch_size'] = 8  # T4适配
config['per_device_eval_batch_size'] = 8
config['num_train_epochs'] = 3
config['save_steps'] = 500
config['eval_steps'] = 500
config['logging_steps'] = 50
config['fp16'] = True
config['time_save_interval_seconds'] = 1800  # 每30分钟备份
config['gdrive_backup_dir'] = '/content/drive/MyDrive/tess_backups'
config['backup_keep_last'] = 2

# 保存修改
with open('configs/tess_gpu_oneline_sc_colab.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ 配置已更新:")
print(json.dumps(config, indent=2))
```

---

### 步骤4: 启动训练 (6-7小时)

```python
# ============ Cell 8: 完整训练 (3 epochs) ============
!python run_mlm.py configs/tess_gpu_oneline_sc_colab.json

# ============ 或者使用命令行参数 ============
!python run_mlm.py \
    --model_name_or_path roberta-base \
    --tokenizer_name extended_tokenizer \
    --train_file tess_train1_oneline.txt \
    --validation_file tess_valid1_oneline.txt \
    --output_dir /content/drive/MyDrive/tess_outputs \
    --line_by_line True \
    --max_seq_length 256 \
    --pad_to_max_length True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --fp16 True \
    --simplex_value 5 \
    --num_diffusion_steps 500 \
    --num_inference_diffusion_steps 100 \
    --beta_schedule squaredcos_improved_ddpm \
    --self_condition logits_addition \
    --self_condition_zeros_after_softmax True \
    --overwrite_output_dir True
```

**训练日志示例**:
```
Step 50/6250:  loss=3.245  lr=9.8e-5  speed=1.1s/step
Step 100/6250: loss=2.987  lr=9.6e-5  speed=1.0s/step
Step 500/6250: loss=2.156  lr=9.0e-5  speed=1.0s/step
Saving checkpoint to checkpoint-500...
Running evaluation...
  eval_loss: 2.234
Step 1000/6250: loss=1.876  lr=8.4e-5  speed=1.0s/step
...
Epoch 1/3 完成, 总用时: 1.7小时
```

---

### 步骤5: 快速验证训练效果 (可选 - 2小时)

如果想快速验证修复是否有效,可先训练1个epoch:

```python
# ============ Cell 9: 快速训练 (1 epoch) ============
!python run_mlm.py \
    --model_name_or_path roberta-base \
    --tokenizer_name extended_tokenizer \
    --train_file tess_train1_oneline.txt \
    --validation_file tess_valid1_oneline.txt \
    --output_dir /content/drive/MyDrive/tess_outputs_quick \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --fp16 True \
    --simplex_value 5 \
    --num_diffusion_steps 500 \
    --self_condition logits_addition \
    --max_seq_length 256 \
    --line_by_line True
```

预计时间: **1.7-2小时**

---

### 步骤6: 评测 (40-50分钟)

```python
# ============ Cell 10: 快速评测 (200 queries) ============
!python run_optimized_eval.py \
    --checkpoint /content/drive/MyDrive/tess_outputs/checkpoint-6000 \
    --test_file tess_test1_oneline.txt \
    --mode tail \
    --quick

# ============ Cell 11: Grid Search 最优参数 ============
!python run_optimized_eval.py \
    --checkpoint /content/drive/MyDrive/tess_outputs/checkpoint-6000 \
    --grid_search \
    --num_queries 500

# ============ Cell 12: 完整评测 (tail预测) ============
!python run_optimized_eval.py \
    --checkpoint /content/drive/MyDrive/tess_outputs/checkpoint-6000 \
    --mode tail \
    --num_queries 2000 \
    --tess_t_eval 60 \
    --neg_k 128 \
    --output eval_tail_results.json

# ============ Cell 13: 完整评测 (head预测) ============
!python run_optimized_eval.py \
    --checkpoint /content/drive/MyDrive/tess_outputs/checkpoint-6000 \
    --mode head \
    --num_queries 2000 \
    --tess_t_eval 60 \
    --neg_k 128 \
    --output eval_head_results.json
```

**预期结果对比**:

| 模式 | 修复前MRR | 预期修复后MRR | 提升 |
|------|-----------|--------------|------|
| Tail预测 | 16.7% | **35-45%** | +110-170% |
| Head预测 | ~15% | **30-40%** | +100-160% |

---

### 步骤7: 监控训练 (在训练时运行)

```python
# ============ Cell 14: 启动TensorBoard ============
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/tess_outputs

# ============ Cell 15: 查看训练进度 ============
!tail -n 20 /content/drive/MyDrive/tess_outputs/trainer_log.txt

# ============ Cell 16: 检查checkpoints ============
!ls -lh /content/drive/MyDrive/tess_outputs/checkpoint-*/
```

---

## 时间安排建议

### 方案A: 完整训练 (推荐用于最终结果)
```
1. 环境设置: 10分钟
2. 数据准备: 3分钟
3. 训练3 epochs: 6-7小时
4. 评测: 50分钟
总计: 约8小时
```

### 方案B: 快速验证 (推荐首次尝试)
```
1. 环境设置: 10分钟
2. 数据准备: 3分钟
3. 训练1 epoch: 2小时
4. 快速评测(200 queries): 10分钟
总计: 约2.5小时
```

### 方案C: 分批训练 (适合Colab时间限制)
```
第一次运行:
- 训练2 epochs: 4小时
- 保存checkpoint

第二次运行:
- 从checkpoint恢复
- 继续训练1 epoch: 2小时
- 评测: 50分钟
```

---

## Colab资源限制应对

### 问题1: 12小时运行限制
**解决**:
```python
# 设置自动保存
config['save_steps'] = 500  # 每500步保存
config['time_save_interval_seconds'] = 1800  # 每30分钟快照

# 断点恢复
!python run_mlm.py \
    --resume_from_checkpoint /content/drive/MyDrive/tess_outputs/checkpoint-3000 \
    configs/tess_gpu_oneline_sc_colab.json
```

### 问题2: GPU断开
**解决**: 所有输出都保存到Google Drive,可随时恢复

### 问题3: 内存不足
**解决**:
```python
config['per_device_train_batch_size'] = 4  # 降到4
config['gradient_accumulation_steps'] = 2  # 累积2步
config['max_seq_length'] = 200  # 缩短序列
```

---

## 训练完成验证清单

训练结束后检查:

- [ ] 训练loss降到 < 2.0
- [ ] 验证loss稳定在 2.0-2.5
- [ ] checkpoint文件完整(pytorch_model.bin存在)
- [ ] Tail评测MRR > 30%
- [ ] Head评测MRR > 25%
- [ ] Hits@10 > 50%

---

## 常见问题排查

### 问题: loss不下降
**检查**:
1. Tokenizer是否使用extended_tokenizer
2. Learning rate是否过小(应为1e-4)
3. 数据是否正确加载

### 问题: 评测结果仍然低
**排查**:
1. 运行grid search找最优tess_t_eval
2. 检查实体tokenization: `validate_config.py --check_tokenization`
3. 确认使用正确的checkpoint

### 问题: 训练太慢
**优化**:
1. 确认fp16已启用
2. 减少eval_steps到1000
3. 使用gradient_accumulation_steps

---

## 预期最终性能

基于修复,预期在ICEWS数据集上:

**Tail预测**:
- MRR: 35-45% (原16.7%)
- Hits@1: 20-30% (原7.6%)
- Hits@10: 55-65% (原34.7%)

**Head预测**:
- MRR: 30-40%
- Hits@1: 18-28%
- Hits@10: 50-60%

**训练时间**: T4 GPU 约6-7小时(3 epochs)

---

## 保存结果

```python
# ============ Cell 17: 整理结果 ============
# 创建结果目录
!mkdir -p /content/drive/MyDrive/tess_final_results

# 复制最佳checkpoint
!cp -r /content/drive/MyDrive/tess_outputs/checkpoint-6000 \
       /content/drive/MyDrive/tess_final_results/best_checkpoint

# 复制评测结果
!cp eval_*.json /content/drive/MyDrive/tess_final_results/

# 生成报告
!python -c "
import json
results = {
    'tail': json.load(open('eval_tail_results.json')),
    'head': json.load(open('eval_head_results.json'))
}
print('=== 最终评测结果 ===')
print(f'Tail MRR: {results[\"tail\"][\"MRR\"]:.4f}')
print(f'Tail Hits@10: {results[\"tail\"][\"Hits@10\"]:.4f}')
print(f'Head MRR: {results[\"head\"][\"MRR\"]:.4f}')
print(f'Head Hits@10: {results[\"head\"][\"Hits@10\"]:.4f}')
"
```

---

**开始训练前建议**: 先运行方案B(快速验证2.5小时),确认修复有效后再运行完整训练。
