# 📁 Checkpoint 保存策略说明

## ✅ 已配置的 Checkpoint 设置

### 关键参数

```json
{
  "save_strategy": "steps",        // 按步数保存
  "save_steps": 500,                // 每500步保存一次
  "save_total_limit": 5,            // 最多保留5个checkpoint
  "evaluation_strategy": "steps",   // 每500步也进行一次评测
  "eval_steps": 500
}
```

---

## 📦 Checkpoint 文件结构

每个 checkpoint 会自动包含以下完整文件：

```
outputs/
├── checkpoint-500/
│   ├── config.json                    # 模型配置
│   ├── pytorch_model.bin              # 模型权重
│   ├── optimizer.pt                   # 优化器状态
│   ├── scheduler.pt                   # 学习率调度器状态
│   ├── trainer_state.json             # 训练状态
│   ├── training_args.bin              # 训练参数
│   └── rng_state.pth                  # 随机数状态
├── checkpoint-1000/
│   └── (同上)
├── checkpoint-1500/
│   └── (同上)
├── checkpoint-2000/
│   └── (同上)
└── checkpoint-2500/
    └── (同上)
```

---

## 🔄 自动清理机制

### 工作原理

1. **保存新checkpoint时**:
   - 每500步创建新的 `checkpoint-N/` 文件夹
   - 保存完整的模型和训练状态

2. **超过限制时自动删除**:
   - 当checkpoint数量 > 5 时
   - 自动删除**最早**的checkpoint
   - 例如: 保存checkpoint-3000时，会删除checkpoint-500

### 示例流程

```
步骤    | 保存的Checkpoint                                  | 操作
--------|--------------------------------------------------|--------
500     | checkpoint-500                                   | 创建
1000    | checkpoint-500, checkpoint-1000                  | 创建
1500    | checkpoint-500, 1000, 1500                       | 创建
2000    | checkpoint-500, 1000, 1500, 2000                 | 创建
2500    | checkpoint-500, 1000, 1500, 2000, 2500           | 创建
3000    | checkpoint-1000, 1500, 2000, 2500, 3000          | 删除500
3500    | checkpoint-1500, 2000, 2500, 3000, 3500          | 删除1000
```

---

## 💾 存储空间估算

### 单个 Checkpoint 大小

对于 roberta-base + TESS diffusion:

| 文件 | 大小 (约) |
|------|----------|
| pytorch_model.bin | ~500 MB |
| optimizer.pt | ~1 GB |
| scheduler.pt | ~10 KB |
| config.json | ~1 KB |
| 其他文件 | ~10 MB |
| **总计** | **~1.5 GB** |

### 总存储需求

```
5 个 checkpoint × 1.5 GB = ~7.5 GB
```

**Google Colab 磁盘**: 默认 ~100 GB，完全足够 ✅

---

## 🚀 训练时间估算 (T4 GPU)

### 1 Epoch 训练

假设训练数据 ~10,000 条，batch_size=16:

```
总步数 = 10,000 / 16 ≈ 625 steps

保存的 checkpoint:
- checkpoint-500  (第500步，约80%进度)
- checkpoint-625  (训练结束，100%进度)

预计时间: ~2 小时
```

### 3 Epochs 训练

```
总步数 = 625 × 3 ≈ 1,875 steps

保存的 checkpoint:
- checkpoint-500   (Epoch 1, 27%)
- checkpoint-1000  (Epoch 2, 53%)
- checkpoint-1500  (Epoch 3, 80%)
- checkpoint-1875  (训练结束, 100%)

预计时间: ~6-7 小时
```

---

## 🔐 防止数据丢失的多重保护

### 1. 定期 Checkpoint (每500步)

```json
"save_steps": 500,
"save_total_limit": 5
```

**保护**: 即使Colab中断，最多损失500步 (~15-20分钟)

### 2. 时间触发备份 (每5分钟)

```json
"time_save_interval_seconds": 300
```

**保护**: 每5分钟额外保存一次轻量级快照

### 3. Google Drive 备份 (可选)

```json
"gdrive_backup_dir": "/content/drive/MyDrive/tess_backups"
```

**使用方法**: 在Colab中挂载Google Drive后启用

---

## 📋 使用说明

### 在 Google Colab 中启用 Google Drive 备份

1. **挂载 Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **修改配置** (可选):
   ```python
   import json
   
   # 读取配置
   with open('configs/tess_gpu_oneline_sc.json', 'r') as f:
       config = json.load(f)
   
   # 启用 Google Drive 备份
   config['gdrive_backup_dir'] = '/content/drive/MyDrive/tess_backups'
   config['backup_keep_last'] = 3
   
   # 保存配置
   with open('configs/tess_gpu_oneline_sc.json', 'w') as f:
       json.dump(config, f, indent=2)
   ```

3. **开始训练**:
   ```bash
   python run_mlm.py configs/tess_gpu_oneline_sc.json
   ```

---

## 🔍 查看和管理 Checkpoint

### 查看所有 checkpoint

```bash
# 列出所有checkpoint
ls -lh outputs/checkpoint-*

# 或使用Python
import os
checkpoints = sorted([d for d in os.listdir('outputs') if d.startswith('checkpoint-')])
print(f"保存的checkpoint: {checkpoints}")
print(f"总数: {len(checkpoints)}")
```

### 从特定 checkpoint 恢复训练

```bash
python run_mlm.py configs/tess_gpu_oneline_sc.json \
    --resume_from_checkpoint outputs/checkpoint-1500
```

### 手动删除旧 checkpoint (如果需要)

```bash
# 删除特定checkpoint
rm -rf outputs/checkpoint-500

# 只保留最新的3个
ls -dt outputs/checkpoint-* | tail -n +4 | xargs rm -rf
```

---

## ⚠️ 重要提示

### 1. Colab 断线重连

如果 Colab 断线，重新运行后:

```python
# 自动从最新checkpoint恢复
python run_mlm.py configs/tess_gpu_oneline_sc.json \
    --resume_from_checkpoint outputs/checkpoint-1500
```

### 2. 磁盘空间监控

定期检查磁盘使用:

```bash
df -h /content
```

如果空间不足，可以:
- 减少 `save_total_limit` 从 5 到 3
- 删除不需要的checkpoint
- 压缩并移动到Google Drive

### 3. 训练完成后

```bash
# 最终模型会保存在根目录
outputs/
├── checkpoint-1875/          # 最后一个step
├── config.json               # 最终模型配置
├── pytorch_model.bin         # 最终模型权重
└── trainer_state.json        # 训练状态
```

---

## 📊 配置对比

| 参数 | 之前 | 现在 | 说明 |
|------|------|------|------|
| `save_steps` | 500 | 500 | 保持不变 |
| `save_total_limit` | ❌ 未设置 | ✅ 5 | **新增**: 最多5个checkpoint |
| `eval_steps` | 500 | 500 | 保持不变 |

---

## ✅ 总结

**新配置优势**:

1. ✅ **每500步自动保存** - 防止进度丢失
2. ✅ **最多保留5个checkpoint** - 节省磁盘空间 (~7.5 GB)
3. ✅ **自动删除旧checkpoint** - 无需手动管理
4. ✅ **包含完整模型文件** - 可随时恢复训练
5. ✅ **支持Google Drive备份** - 额外安全保障

**适用场景**:

- ✅ Google Colab 免费版 (可能随时断线)
- ✅ 长时间训练 (3-10 小时)
- ✅ 磁盘空间有限 (~100 GB)
- ✅ 需要中途评测和监控

**完全就绪，可以开始训练！** 🚀

