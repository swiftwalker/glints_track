# 完整训练系统使用指南

## 🎯 新的 Checkpoint 系统

训练系统已完全重构，现在支持：
- ✅ 完整的 checkpoint 保存（模型 + 优化器 + 训练参数）
- ✅ 恢复训练功能
- ✅ 结构化输出目录
- ✅ 详细的训练日志（无 tqdm 进度信息）
- ✅ 训练指标的 JSON 记录

## 📁 输出目录结构

```
runs/
└── train_20251106_143022/          # 训练输出目录
    ├── training.log                # 文本日志（完整训练记录）
    ├── metrics.json                # 训练指标（JSON 格式）
    ├── training_args.json          # 训练参数配置
    ├── best_model.pth              # 最佳模型（完整 checkpoint）
    ├── best_model_weights.pt       # 最佳模型权重（仅权重，兼容旧代码）
    ├── latest_checkpoint.pth       # 最新 checkpoint（用于恢复训练）
    └── checkpoints/                # 定期保存的 checkpoint
        ├── checkpoint_epoch_010.pth
        ├── checkpoint_epoch_020.pth
        └── checkpoint_epoch_030.pth
```

## 🚀 基础使用

### 1. 新建训练（自动创建输出目录）

```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --epochs 50 \
    --batch 16 \
    --lr 1e-3
```

输出目录将自动创建为 `runs/train_YYYYMMDD_HHMMSS/`

### 2. 指定输出目录

```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal_experiment \
    --epochs 50 \
    --loss focal
```

### 3. 恢复训练

```bash
# 方法 1: 从 latest_checkpoint 恢复
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal_experiment \
    --resume runs/focal_experiment/latest_checkpoint.pth \
    --epochs 100

# 方法 2: 从特定 checkpoint 恢复
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal_experiment \
    --resume runs/focal_experiment/checkpoints/checkpoint_epoch_020.pth \
    --epochs 100
```

## 📊 日志和指标

### training.log（文本日志）

```
[2025-11-06 14:30:22] 训练日志 - 开始时间: 2025-11-06 14:30:22
[2025-11-06 14:30:22] ================================================================================
[2025-11-06 14:30:22] 开始训练
[2025-11-06 14:30:22] 输出目录: runs/train_20251106_143022
[2025-11-06 14:30:22] 数据集: train_256.h5
...
[2025-11-06 14:35:15] Epoch 001 (耗时: 45.32s, LR: 1.00e-03):
[2025-11-06 14:35:15]   Train: total=0.2345 focal=0.1234 bce=0.0000 dice=0.0000 div=0.0111
[2025-11-06 14:35:15]   Val:   total=0.1987 focal=0.1045 bce=0.0000 dice=0.0000 div=0.0092
[2025-11-06 14:35:15]   ✅ 保存最佳模型 (Val Loss: 0.1987)
```

### metrics.json（结构化指标）

```json
[
  {
    "epoch": 1,
    "train": {
      "total": 0.2345,
      "focal": 0.1234,
      "bce": 0.0,
      "dice": 0.0,
      "div": 0.0111
    },
    "val": {
      "total": 0.1987,
      "focal": 0.1045,
      "bce": 0.0,
      "dice": 0.0,
      "div": 0.0092
    },
    "lr": 0.001,
    "epoch_time": 45.32
  }
]
```

## ⚙️ 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `runs/train_YYYYMMDD_HHMMSS` | 输出目录 |
| `--resume` | None | 恢复训练的 checkpoint 路径 |
| `--save_freq` | 10 | 每 N 个 epoch 保存一次 checkpoint |
| `--keep_last_n` | 3 | 保留最近 N 个 checkpoint（未实现） |

## 📝 完整参数列表

### 数据和输出
- `--h5`: HDF5 数据集路径（必需）
- `--output_dir`: 输出目录
- `--resume`: 恢复训练的 checkpoint 路径

### 训练参数
- `--epochs`: 训练轮数（默认 50）
- `--batch`: Batch size（默认 8）
- `--lr`: 学习率（默认 1e-3）
- `--val_split`: 验证集比例（默认 0.1）
- `--gpu`: 指定 GPU 设备（如 '0', '1', '0,1' 等，不指定则自动选择）

### 数据加载优化
- `--num_workers`: DataLoader 工作进程数（默认 4）
- `--preload`: 预加载数据到内存（默认启用）
- `--no_preload`: 禁用预加载
- `--shared_memory`: 使用共享内存

### Checkpoint 相关
- `--save_freq`: 保存频率（默认每 10 epoch）
- `--no_save_optimizer`: 不保存优化器状态（减小文件大小约 50%，但无法完美恢复训练）
  - 默认：保存优化器状态（推荐，支持完美恢复训练）

### Loss 函数
- `--loss`: Loss 类型（focal/bce/dice/hybrid）
- `--alpha`, `--gamma`: Focal Loss 参数
- `--lam_focal`, `--lam_bce`, `--lam_dice`: 损失权重
- `--div_weight`, `--div_mode`: 相似度惩罚
- `--lam_agg`, `--agg_mode`: 聚合参数

## 🎬 实际使用场景

### 场景 1: 快速实验

```bash
# 快速测试，自动保存到 runs/
python train_glint_unet.py --h5 train_256.h5 --epochs 10 --batch 32
```

### 场景 2: 正式训练

```bash
# 指定输出目录，便于管理
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/focal_baseline \
    --epochs 100 \
    --batch 16 \
    --lr 1e-3 \
    --loss focal \
    --save_freq 5
```

### 场景 3: 训练中断后恢复

```bash
# 从中断处继续训练
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/focal_baseline \
    --resume experiments/focal_baseline/latest_checkpoint.pth \
    --epochs 200
```

### 场景 4: 指定 GPU 训练

```bash
# 使用 GPU 0
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/gpu0_exp \
    --gpu 0 \
    --epochs 50

# 使用 GPU 1
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/gpu1_exp \
    --gpu 1 \
    --epochs 50

# 同时在多个 GPU 上运行不同实验
python train_glint_unet.py --h5 train_256.h5 --output_dir exp1 --gpu 0 --epochs 50 &
python train_glint_unet.py --h5 train_256.h5 --output_dir exp2 --gpu 1 --epochs 50 &
wait
```

### 场景 5: 多实验对比

```bash
# 实验 1: Focal Loss
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/exp1_focal \
    --loss focal \
    --epochs 50 &

# 实验 2: Hybrid Loss
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/exp2_hybrid \
    --loss hybrid \
    --epochs 50 &

# 实验 3: BCE Loss
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/exp3_bce \
    --loss bce \
    --epochs 50 &

wait
```

### 场景 6: 不保存优化器状态（减小文件大小）

```bash
# 适合：只需要模型权重，不需要恢复训练的场景
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/inference_only \
    --no_save_optimizer \
    --epochs 100

# 文件大小对比（示例）：
# 保存优化器:     checkpoint.pth ~200MB
# 不保存优化器:   checkpoint.pth ~100MB (节省约50%)
```

### 场景 7: 学习率调优

```bash
for lr in 1e-2 5e-3 1e-3 5e-4 1e-4; do
    python train_glint_unet.py \
        --h5 train_256.h5 \
        --output_dir experiments/lr_tuning/lr_${lr} \
        --lr ${lr} \
        --epochs 30 \
        --shared_memory &
done
wait
```

## 🔍 查看训练结果

### 查看日志

```bash
# 实时查看训练日志
tail -f runs/train_20251106_143022/training.log

# 查看完整日志
cat runs/train_20251106_143022/training.log

# 搜索最佳性能
grep "保存最佳模型" runs/train_20251106_143022/training.log
```

### 分析指标

```python
import json
import matplotlib.pyplot as plt

# 读取指标
with open('runs/train_20251106_143022/metrics.json', 'r') as f:
    metrics = json.load(f)

# 绘制 loss 曲线
epochs = [m['epoch'] for m in metrics]
train_loss = [m['train']['total'] for m in metrics]
val_loss = [m['val']['total'] for m in metrics]

plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, val_loss, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## 🔄 与旧版本的兼容性

旧参数仍然支持（已弃用）：
- `--model_path`: 使用 `--output_dir` 替代
- `--save`: 使用 `--output_dir` 替代

旧模型加载方式：
```python
# 方法 1: 仅加载权重（兼容）
model.load_state_dict(torch.load('runs/xxx/best_model_weights.pt'))

# 方法 2: 加载完整 checkpoint（推荐）
checkpoint = torch.load('runs/xxx/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 💡 最佳实践

1. **命名输出目录**：使用有意义的名称
   ```bash
   --output_dir experiments/focal_lr1e3_batch16
   ```

2. **定期保存**：重要实验设置较小的 `--save_freq`
   ```bash
   --save_freq 5
   ```

3. **使用共享内存**：多实验并行时
   ```bash
   --shared_memory
   ```

4. **恢复训练**：意外中断后继续
   ```bash
   --resume path/to/latest_checkpoint.pth
   ```

5. **查看日志**：定期检查训练进度
   ```bash
   tail -f runs/xxx/training.log
   ```

## 🆘 常见问题

**Q: 如何从 checkpoint 提取模型权重？**
```python
checkpoint = torch.load('runs/xxx/best_model.pth')
torch.save(checkpoint['model_state_dict'], 'model_weights_only.pt')
```

**Q: 如何更改恢复训练的学习率？**
```bash
# checkpoint 会恢复优化器状态，但可以手动修改
python train_glint_unet.py \
    --resume runs/xxx/latest_checkpoint.pth \
    --lr 1e-4  # 新的学习率会覆盖
```

**Q: 输出目录已存在怎么办？**
- 系统会继续使用该目录
- 新的日志会追加到文件
- 建议使用不同的输出目录或清理旧文件

**Q: 如何清理旧的 checkpoint？**
```bash
# 只保留最佳模型和最新 checkpoint
cd runs/xxx/checkpoints
ls -t | tail -n +4 | xargs rm  # 保留最新 3 个
```

**Q: 不保存优化器状态有什么影响？**
- ✅ 优点：文件大小减少约 50%
- ❌ 缺点：无法完美恢复训练（学习率调度、momentum 等会丢失）
- 💡 建议：如果只需要模型推理，使用 `--no_save_optimizer`
- 💡 建议：如果需要恢复训练，保持默认（保存优化器状态）
