# 🚀 Checkpoint 保存配置 - 快速参考

## ✅ 已完成的配置更新

### 新增参数

```json
"save_total_limit": 5
```

**作用**: 最多保留5个checkpoint，自动删除最旧的

---

## 📦 Checkpoint 保存效果

### 训练过程示例

```
训练开始...
├─ Step 500   → 保存 checkpoint-500/   (包含完整模型)
├─ Step 1000  → 保存 checkpoint-1000/  (包含完整模型)
├─ Step 1500  → 保存 checkpoint-1500/  (包含完整模型)
├─ Step 2000  → 保存 checkpoint-2000/  (包含完整模型)
├─ Step 2500  → 保存 checkpoint-2500/  (包含完整模型)
└─ Step 3000  → 保存 checkpoint-3000/ + 删除 checkpoint-500/ ✅
```

### 每个 checkpoint 包含的文件

```
checkpoint-500/
├── config.json           ✅ 模型配置
├── pytorch_model.bin     ✅ 模型权重 (~500 MB)
├── optimizer.pt          ✅ 优化器状态 (~1 GB)
├── scheduler.pt          ✅ 学习率调度器
├── trainer_state.json    ✅ 训练进度
├── training_args.bin     ✅ 训练参数
└── rng_state.pth         ✅ 随机数状态
```

**总大小**: ~1.5 GB/checkpoint × 5 = **~7.5 GB**

---

## 🔐 防丢失保护机制

### 1️⃣ 每500步保存 checkpoint

```json
"save_steps": 500
```

**保护**: 最多损失500步进度 (~15-20分钟)

### 2️⃣ 每5分钟额外快照

```json
"time_save_interval_seconds": 300
```

**保护**: 时间触发的额外备份

### 3️⃣ 自动限制数量

```json
"save_total_limit": 5
```

**保护**: 防止磁盘空间耗尽

---

## 🎯 Google Colab 使用流程

### 场景1: 正常训练 (无中断)

```bash
# 直接开始训练
python run_mlm.py configs/tess_gpu_oneline_sc.json

# 训练过程会自动:
# ✅ 每500步保存checkpoint
# ✅ 保持最新5个checkpoint
# ✅ 自动删除旧checkpoint
```

### 场景2: Colab 断线后恢复

```bash
# 1. 查看现有checkpoint
ls -lh outputs/checkpoint-*

# 输出示例:
# checkpoint-1000/
# checkpoint-1500/
# checkpoint-2000/

# 2. 从最新checkpoint恢复
python run_mlm.py configs/tess_gpu_oneline_sc.json \
    --resume_from_checkpoint outputs/checkpoint-2000

# ✅ 从第2000步继续训练
# ✅ 优化器状态完全恢复
# ✅ 学习率调度器继续
```

### 场景3: 启用 Google Drive 备份 (推荐)

```python
# 在 Colab Notebook 中:

# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 修改配置启用备份
import json
with open('configs/tess_gpu_oneline_sc.json', 'r') as f:
    config = json.load(f)

config['gdrive_backup_dir'] = '/content/drive/MyDrive/tess_backups'

with open('configs/tess_gpu_oneline_sc.json', 'w') as f:
    json.dump(config, f, indent=2)

# 3. 开始训练
!python run_mlm.py configs/tess_gpu_oneline_sc.json
```

**效果**: 每个checkpoint同时备份到Google Drive ✅

---

## 📊 时间估算 (T4 GPU)

### 1 Epoch (~10,000 样本)

```
总步数: 625 步
训练时间: ~2 小时
保存checkpoint: checkpoint-500, checkpoint-625
磁盘使用: ~3 GB
```

### 3 Epochs

```
总步数: 1,875 步
训练时间: ~6-7 小时
保存checkpoint: 
  - checkpoint-500   (Epoch 1, 27%)
  - checkpoint-1000  (Epoch 2, 53%)
  - checkpoint-1500  (Epoch 3, 80%)
  - checkpoint-1875  (完成, 100%)
磁盘使用: ~6 GB
```

---

## 🔍 常用命令

### 查看所有 checkpoint

```bash
ls -lh outputs/ | grep checkpoint
```

### 查看 checkpoint 详情

```bash
# 查看某个checkpoint包含的文件
ls -lh outputs/checkpoint-1500/

# 查看训练状态
cat outputs/checkpoint-1500/trainer_state.json
```

### 手动清理 (如果需要)

```bash
# 只保留最新3个
ls -dt outputs/checkpoint-* | tail -n +4 | xargs rm -rf

# 删除特定checkpoint
rm -rf outputs/checkpoint-500
```

### 从特定步数恢复

```bash
# 从checkpoint-1000继续训练
python run_mlm.py configs/tess_gpu_oneline_sc.json \
    --resume_from_checkpoint outputs/checkpoint-1000
```

---

## ⚠️ 重要提示

### 1. 磁盘空间检查

```bash
# 检查可用空间
df -h /content

# Colab 默认: ~100 GB
# 5个checkpoint: ~7.5 GB
# 剩余空间: ~92 GB ✅
```

### 2. 训练完成后

最终模型保存在:
```
outputs/
├── config.json            # 最终配置
├── pytorch_model.bin      # 最终权重
└── checkpoint-N/          # 最后一步的完整状态
```

### 3. 如果空间不足

修改配置减少保留数量:
```json
"save_total_limit": 3  // 从5改为3
```

---

## ✅ 配置验证清单

- ✅ `save_strategy`: "steps" - 按步数保存
- ✅ `save_steps`: 500 - 每500步保存
- ✅ `save_total_limit`: 5 - 最多5个checkpoint
- ✅ `evaluation_strategy`: "steps" - 同时评测
- ✅ `eval_steps`: 500 - 每500步评测
- ✅ `time_save_interval_seconds`: 300 - 额外时间备份

**所有设置已就绪，可以开始训练！** 🎉

---

## 📝 训练日志示例

```
Training...
Step 500/1875  [=======>.......] 27%  Loss: 2.34
  ✅ Saving checkpoint to outputs/checkpoint-500
  ✅ Evaluation: MRR=0.28

Step 1000/1875 [=============>.] 53%  Loss: 1.89
  ✅ Saving checkpoint to outputs/checkpoint-1000
  ✅ Evaluation: MRR=0.32

Step 1500/1875 [==================>.] 80%  Loss: 1.56
  ✅ Saving checkpoint to outputs/checkpoint-1500
  ✅ Evaluation: MRR=0.36

Step 1875/1875 [====================] 100%  Loss: 1.42
  ✅ Saving checkpoint to outputs/checkpoint-1875
  ✅ Training complete!
  ✅ Final model saved to outputs/
```

