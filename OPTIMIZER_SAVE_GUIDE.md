# 优化器状态保存选项说明

## ✨ 新功能

添加了 `--no_save_optimizer` 参数，允许控制是否在 checkpoint 中保存优化器状态。

**默认行为**：保存优化器状态（推荐，支持完美恢复训练）

## 🎯 使用场景

### 场景 1: 需要恢复训练（默认行为）

```bash
# 保存优化器状态（默认行为，无需额外参数）
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/exp1 \
    --epochs 100
```

**优点**：
- ✅ 可以完美恢复训练（包括 momentum、学习率调度等）
- ✅ 适合长时间训练、可能中断的场景

**缺点**：
- ❌ 文件较大（约 2 倍大小）

### 场景 2: 只需要模型权重

```bash
# 不保存优化器状态
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/exp2 \
    --no_save_optimizer \
    --epochs 100
```

**优点**：
- ✅ 文件大小减少约 50%
- ✅ 节省磁盘空间
- ✅ 适合只需要推理的模型

**缺点**：
- ❌ 无法完美恢复训练状态
- ❌ 使用 `--resume` 时会从头初始化优化器

## 📊 文件大小对比

以 UNet 模型为例：

| 配置 | Checkpoint 大小 | 说明 |
|-----|----------------|------|
| 默认（保存优化器） | ~200 MB | 模型权重 (~100MB) + 优化器状态 (~100MB) |
| `--no_save_optimizer` | ~100 MB | 仅模型权重 (~100MB) |

**节省空间**：对于 300 个 epoch，save_freq=50 的训练：
- 默认：6 个 checkpoint × 200MB = 1.2 GB
- 不保存优化器：6 个 checkpoint × 100MB = 600 MB
- **节省：600 MB (50%)**

## 🔄 恢复训练的影响

### 保存优化器状态（推荐）

```bash
# 训练（默认保存优化器状态）
python train_glint_unet.py --h5 train_256.h5 --output_dir exp --epochs 100

# 恢复训练（完美继续）
python train_glint_unet.py --h5 train_256.h5 --output_dir exp \
    --resume exp/latest_checkpoint.pth --epochs 200
```

**恢复内容**：
- ✅ 模型权重
- ✅ 优化器状态（momentum 缓冲区）
- ✅ 学习率
- ✅ Epoch 计数
- ✅ 训练指标

### 不保存优化器状态

```bash
# 训练
python train_glint_unet.py --h5 train_256.h5 --output_dir exp \
    --no_save_optimizer --epochs 100

# 恢复训练（部分恢复）
python train_glint_unet.py --h5 train_256.h5 --output_dir exp \
    --resume exp/latest_checkpoint.pth --epochs 200
```

**恢复内容**：
- ✅ 模型权重
- ❌ 优化器状态（会重新初始化）
- ⚠️  学习率（使用命令行指定的值）
- ✅ Epoch 计数
- ✅ 训练指标

**影响**：
- 优化器的 momentum 缓冲区丢失，可能导致训练不稳定
- 需要手动调整学习率以适应恢复训练

## 💡 推荐使用方式

### 情况 A: 正式训练（推荐保存优化器）

```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/final_model \
    --epochs 300 \
    --save_freq 10
```

**原因**：
- 训练时间长，可能中断
- 需要完美恢复训练状态
- 磁盘空间充足

### 情况 B: 快速实验（可不保存优化器）

```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/quick_test \
    --no_save_optimizer \
    --epochs 50
```

**原因**：
- 训练时间短，不太可能中断
- 主要关注最终模型权重
- 节省磁盘空间

### 情况 C: 批量实验（可不保存优化器）

```bash
for loss in focal bce hybrid; do
    python train_glint_unet.py \
        --h5 train_256.h5 \
        --output_dir exp_${loss} \
        --no_save_optimizer \
        --loss ${loss} \
        --epochs 50 &
done
wait
```

**原因**：
- 多个实验并行，磁盘压力大
- 主要对比最终性能，不需要恢复
- 节省大量磁盘空间

## 🔍 检查 Checkpoint 内容

```python
import torch

# 加载 checkpoint
checkpoint = torch.load('runs/exp1/best_model.pth')

# 查看包含的键
print("Checkpoint 包含的键:", checkpoint.keys())
# 输出: dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'train_args', 'metrics', 'timestamp'])

# 检查是否包含优化器状态
if 'optimizer_state_dict' in checkpoint:
    print("✅ 包含优化器状态")
else:
    print("❌ 不包含优化器状态")
```

## 📝 技术实现

### CheckpointManager 类

```python
class CheckpointManager:
    def __init__(self, output_dir, save_optimizer=True):
        self.save_optimizer = save_optimizer
        # ...
    
    def save_checkpoint(self, epoch, model, optimizer, ...):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 只有在 save_optimizer=True 时才保存
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
```

### 加载时的兼容性

```python
def load_checkpoint(self, checkpoint_path, model, optimizer=None, device='cuda'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果 checkpoint 中有优化器状态且传入了 optimizer，则加载
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## ⚙️ 默认行为

- **默认值**：保存优化器状态（无需任何参数）
- **原因**：保证训练可恢复性，这是最安全的选择
- **禁用**：使用 `--no_save_optimizer` 明确禁用以减小文件大小

## ✅ 总结

| 特性 | 默认（保存优化器） | `--no_save_optimizer` |
|------|-----------------|---------------------|
| 文件大小 | 大 (~200MB) | 小 (~100MB) |
| 恢复训练 | 完美恢复 | 部分恢复 |
| 磁盘占用 | 高 | 低 |
| 推荐场景 | 正式训练、长时间训练 | 快速实验、只需推理 |

**建议**：
- 🎯 如果不确定，使用默认设置（保存优化器，无需额外参数）
- 💾 如果磁盘空间紧张，添加 `--no_save_optimizer`
- 🔄 如果需要恢复训练，使用默认设置（保存优化器状态）
