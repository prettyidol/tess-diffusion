# 🚀 TESS 训练快速开始指南

## ✅ 代码已修复完成！

所有与 environment.yaml 的兼容性问题已解决：
- ✅ AdamW 优化器迁移到 torch.optim
- ✅ torch.float32 拼写错误已修复
- ✅ 所有依赖版本兼容确认

## 📦 立即开始 - 3步走

### 1️⃣ 安装环境（约10分钟）
```bash
# 打开 Anaconda Prompt
cd d:\idol01\homework\paper_code\tess-diffusion

# 创建环境（自动安装所有依赖）
conda env create -f environment.yaml

# 激活环境
conda activate sdlm

# 安装项目
pip install -e .

# 验证环境
python verify_environment.py
```

### 2️⃣ 准备数据（约2分钟）
```bash
# 扩展 tokenizer（重要！避免实体分词）
python extend_tokenizer_vocab.py ^
    --train_file tess_train1_oneline.txt ^
    --base_model roberta-base ^
    --output_dir extended_tokenizer
```

### 3️⃣ 开始训练（4-6小时）
```bash
# 修改配置使用扩展的 tokenizer
# 编辑 configs/tess_gpu_oneline_sc.json
# 将 "tokenizer_name": null 改为 "tokenizer_name": "extended_tokenizer"

# 启动训练
python run_mlm.py configs/tess_gpu_oneline_sc.json

# 在另一个终端监控（可选）
tensorboard --logdir outputs/tess_training
```

## 🔧 自定义配置（可选）

### 减少内存占用
如果 GPU 内存不足，编辑 `configs/tess_gpu_oneline_sc.json`:
```json
{
  "per_device_train_batch_size": 8,      // 从 16 减少到 8
  "gradient_accumulation_steps": 2,      // 添加此行
  "max_seq_length": 128                  // 从 256 减少到 128
}
```

### 快速测试训练
创建快速测试配置 `configs/tess_quick_test.json`:
```json
{
  "model_name_or_path": "roberta-base",
  "tokenizer_name": "extended_tokenizer",
  "train_file": "tess_train1_oneline.txt",
  "validation_file": "tess_valid1_oneline.txt",
  "output_dir": "outputs/quick_test",
  "max_steps": 100,                      // 只训练 100 步测试
  "per_device_train_batch_size": 8,
  "simplex_value": 5,
  "num_diffusion_steps": 100,            // 减少扩散步数
  "fp16": true
}
```

## 📊 监控训练

### 检查 Loss
```bash
# 查看训练日志
cat outputs/tess_training/trainer_state.json

# 或使用 TensorBoard
tensorboard --logdir outputs/tess_training
# 浏览器访问: http://localhost:6006
```

### 中途评估
```bash
# 在训练过程中评估最新检查点
python eval_kg_ranking.py ^
    --test_file tess_test1_oneline.txt ^
    --mode tail ^
    --k 1 3 10 ^
    --checkpoint outputs/tess_training/checkpoint-500
```

## ⚠️ 常见问题速查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CUDA out of memory | GPU 内存不足 | 降低 batch_size 到 4 或 8 |
| 实体被分词 | tokenizer 未扩展 | 运行 extend_tokenizer_vocab.py |
| Loss 不下降 | 学习率太高 | 改为 5e-5 或 1e-5 |
| 训练很慢 | CPU 模式 | 检查 torch.cuda.is_available() |
| Import Error | 环境未激活 | conda activate sdlm |

## 📝 完整文档

- **COMPATIBILITY_ANALYSIS.md** - 详细兼容性分析
- **TRAINING_SETUP_SUMMARY.md** - 完整安装和训练指南
- **verify_compatibility.py** - 代码修复验证脚本
- **verify_environment.py** - 环境依赖验证脚本

## 🎯 预期结果

### 训练完成后
- 模型检查点: `outputs/tess_training/checkpoint-XXXX/`
- TensorBoard 日志: `outputs/tess_training/runs/`
- 训练状态: `outputs/tess_training/trainer_state.json`

### 评估指标
- MRR > 0.3
- Hits@10 > 0.6
- 训练 Loss 从 ~8 降到 ~2-3

## 💡 提示

1. **首次运行**: 建议先用 `tess_quick_test.json` 测试 100 步，确保环境正常
2. **GPU 检查**: 运行前确认 `torch.cuda.is_available()` 返回 `True`
3. **定期保存**: 配置已设置每 500 步保存，最多保留 5 个检查点
4. **实验记录**: TensorBoard 会自动记录所有训练指标

---

**准备好了吗？运行第一个命令开始吧！** 🚀

```bash
conda env create -f environment.yaml
```
