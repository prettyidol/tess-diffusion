# TESS 训练环境准备总结

## ✅ 已完成的工作

### 1. 依赖兼容性分析
已创建详细的兼容性分析文档：`COMPATIBILITY_ANALYSIS.md`

**主要发现**:
- ✅ Python 3.9 - 兼容
- ✅ PyTorch 2.2.0 - 兼容
- ✅ Transformers 4.33.3 - 兼容（需修复 AdamW 导入）
- ✅ Diffusers 0.27.2 - 兼容（需修复拼写错误）
- ✅ Datasets 2.14.6 - 兼容
- ✅ 其他依赖 - 全部兼容

### 2. 关键代码修复

#### 修复 1: AdamW 优化器迁移
**文件**: `sdlm/trainer.py`
**问题**: Transformers 4.33.3 已废弃 `from transformers import AdamW`
**修复**: 改用 `from torch.optim import AdamW`
**状态**: ✅ 已修复并验证

```python
# 第 48 行 - 已修复
from torch.optim import AdamW

# 第 707 行 - 无需修改（调用方式相同）
self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
```

#### 修复 2: Torch 数据类型拼写
**文件**: `sdlm/schedulers/scheduling_simplex_ddpm.py`
**问题**: `torch.torch.float32` 拼写错误
**修复**: 改为 `torch.float32`
**状态**: ✅ 已修复并验证

```python
# 第 66 行 - 已修复
return betas, torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)
```

### 3. 验证脚本

#### 脚本 1: verify_compatibility.py
**用途**: 验证代码修复是否正确应用
**运行**: `python verify_compatibility.py`
**结果**: ✅ 所有检查通过

#### 脚本 2: verify_environment.py
**用途**: 验证环境依赖版本是否正确安装
**运行**: `python verify_environment.py` (需要先安装环境)
**用途**: 安装环境后使用

## 📋 训练环境安装步骤

### 步骤 1: 安装 Miniconda3
下载并安装 Miniconda3 for Windows:
```bash
# 下载链接
https://docs.conda.io/en/latest/miniconda.html

# 安装后，打开 Anaconda Prompt
```

### 步骤 2: 创建虚拟环境
```bash
cd d:\idol01\homework\paper_code\tess-diffusion

# 使用 environment.yaml 创建环境
conda env create -f environment.yaml

# 这将创建名为 "sdlm" 的环境，包含所有依赖
```

### 步骤 3: 激活环境
```bash
conda activate sdlm
```

### 步骤 4: 安装项目
```bash
pip install -e .
```

### 步骤 5: 验证环境
```bash
# 运行环境验证脚本
python verify_environment.py

# 应该看到所有 ✅ 标记
```

### 步骤 6: 验证 CUDA（如果使用 GPU）
```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## 🚀 训练流程

### 1. 扩展 Tokenizer（重要！）
```bash
python extend_tokenizer_vocab.py \
    --train_file tess_train1_oneline.txt \
    --base_model roberta-base \
    --output_dir extended_tokenizer
```

这将:
- 从训练数据提取所有实体和关系
- 将它们添加到 tokenizer 词汇表（避免分词）
- 保存扩展后的 tokenizer 到 `extended_tokenizer/`

### 2. 更新配置文件
确保 `configs/tess_gpu_oneline_sc.json` 中的路径正确：

```json
{
  "model_name_or_path": "roberta-base",
  "tokenizer_name": "extended_tokenizer",  // 使用扩展后的 tokenizer
  "train_file": "tess_train1_oneline.txt",
  "validation_file": "tess_valid1_oneline.txt",
  "output_dir": "outputs/tess_training",  // 训练输出目录
  ...
}
```

### 3. 开始训练
```bash
python run_mlm.py configs/tess_gpu_oneline_sc.json
```

### 4. 监控训练
```bash
# 在另一个终端启动 TensorBoard
tensorboard --logdir outputs/tess_training
```

### 5. 评估模型
```bash
python eval_kg_ranking.py \
    --test_file tess_test1_oneline.txt \
    --mode tail \
    --k 1 3 10 \
    --checkpoint outputs/tess_training/checkpoint-XXXX
```

## ⚙️ 配置说明

### 训练参数（tess_gpu_oneline_sc.json）
```json
{
  // 模型参数
  "max_seq_length": 256,                    // 序列最大长度
  "per_device_train_batch_size": 16,       // 训练批次大小
  "per_device_eval_batch_size": 16,        // 评估批次大小
  "learning_rate": 1e-4,                    // 学习率
  
  // Diffusion 参数
  "simplex_value": 5,                       // Simplex 缩放值
  "num_diffusion_steps": 500,               // 训练扩散步数
  "num_inference_diffusion_steps": 100,     // 推理扩散步数
  "beta_schedule": "squaredcos_improved_ddpm",
  
  // Self-conditioning
  "self_condition": "logits_addition",      // 启用 self-conditioning
  "self_condition_zeros_after_softmax": true,
  
  // 训练策略
  "save_steps": 500,                        // 每 500 步保存
  "save_total_limit": 5,                    // 最多保留 5 个检查点
  "fp16": true                              // 使用混合精度
}
```

## 🔍 故障排查

### 问题 1: CUDA 不可用
**症状**: `torch.cuda.is_available()` 返回 `False`
**解决**:
1. 确认 GPU 驱动已安装
2. 检查 CUDA 版本（需要 11.8）
3. 重新安装 PyTorch with CUDA: `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`

### 问题 2: 内存不足
**症状**: `RuntimeError: CUDA out of memory`
**解决**:
1. 降低 batch size: `"per_device_train_batch_size": 8`
2. 启用梯度累积: `"gradient_accumulation_steps": 2`
3. 降低序列长度: `"max_seq_length": 128`

### 问题 3: 实体被分词
**症状**: 评估指标很差，实体未被识别
**解决**:
1. 确认已运行 `extend_tokenizer_vocab.py`
2. 确认配置中使用了扩展的 tokenizer: `"tokenizer_name": "extended_tokenizer"`
3. 运行验证: `python validate_config.py --checkpoint outputs/tess_training/checkpoint-XXX`

### 问题 4: 训练不收敛
**症状**: Loss 不下降或波动很大
**解决**:
1. 检查学习率: 尝试 `1e-5` 或 `5e-5`
2. 检查数据: 确认 `tess_train1_oneline.txt` 格式正确
3. 启用梯度裁剪: `"max_grad_norm": 1.0`

## 📊 预期结果

### 训练指标
- **Loss**: 应该从 ~8-10 逐渐降到 ~2-3
- **训练时间**: 约 4-6 小时（单 GPU，取决于硬件）
- **内存占用**: ~12-16 GB VRAM（batch_size=16）

### 评估指标（KG Ranking）
- **MR**: Mean Rank - 越低越好（目标 < 100）
- **MRR**: Mean Reciprocal Rank - 越高越好（目标 > 0.3）
- **Hits@1**: 越高越好（目标 > 0.2）
- **Hits@3**: 越高越好（目标 > 0.4）
- **Hits@10**: 越高越好（目标 > 0.6）

## 📝 检查清单

在开始训练前，确认以下项目：

- [x] ✅ 代码修复已应用（AdamW, torch.float32）
- [x] ✅ 兼容性验证通过 (`python verify_compatibility.py`)
- [ ] 环境已安装 (`conda env create -f environment.yaml`)
- [ ] 环境验证通过 (`python verify_environment.py`)
- [ ] 项目已安装 (`pip install -e .`)
- [ ] Tokenizer 已扩展 (`python extend_tokenizer_vocab.py ...`)
- [ ] 配置文件已更新（路径正确）
- [ ] GPU 可用（如果使用 GPU）
- [ ] 数据文件存在（tess_train1_oneline.txt, tess_valid1_oneline.txt）

## 🎯 下一步

代码已经准备好与 environment.yaml 中的依赖配合使用：

1. **立即可以做**: 安装环境并开始训练
2. **文档可用**: COMPATIBILITY_ANALYSIS.md 详细说明所有兼容性信息
3. **验证工具**: verify_compatibility.py 和 verify_environment.py 可用于检查

**现在可以安全地按照上述步骤安装环境并开始训练！**
