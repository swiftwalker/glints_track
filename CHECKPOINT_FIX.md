# Checkpoint 保存频率 Bug 修复

## 🐛 问题描述

**症状**：即使设置了 `--save_freq 50`，checkpoints 目录中仍然每个 epoch 都会保存一个文件。

**原因**：`save_checkpoint()` 方法无论 `is_best` 参数是什么值，都会先保存到 `checkpoints/` 目录，然后如果是最佳模型再额外保存到根目录。

## ✅ 修复方案

修改 `CheckpointManager.save_checkpoint()` 方法的逻辑：

### 修复前
```python
# 总是先保存到 checkpoints/ 目录
checkpoint_path = os.path.join(self.checkpoints_dir, filename)
torch.save(checkpoint, checkpoint_path)

# 如果是最佳模型，额外保存一份
if is_best:
    best_path = os.path.join(self.output_dir, "best_model.pth")
    torch.save(checkpoint, best_path)
```

### 修复后
```python
# 如果是最佳模型，保存到根目录
if is_best:
    best_path = os.path.join(self.output_dir, "best_model.pth")
    torch.save(checkpoint, best_path)
    return best_path
else:
    # 定期保存到 checkpoints 目录
    checkpoint_path = os.path.join(self.checkpoints_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
```

## 📁 正确的保存行为

### 配置示例
```bash
--save_freq 50
--epochs 300
```

### 保存逻辑

| Epoch | 条件 | 保存位置 | 说明 |
|-------|------|---------|------|
| 1 | 首次训练 | `latest_checkpoint.pth` | 每个 epoch 都更新 |
| 1 | 首个最佳 | `best_model.pth` | 第一次总是最佳 |
| 50 | `50 % 50 == 0` | `checkpoints/checkpoint_epoch_050.pth` | 定期保存 |
| 75 | 发现更好模型 | `best_model.pth` | 覆盖之前的最佳 |
| 100 | `100 % 50 == 0` | `checkpoints/checkpoint_epoch_100.pth` | 定期保存 |
| 150 | `150 % 50 == 0` | `checkpoints/checkpoint_epoch_150.pth` | 定期保存 |
| 200 | `200 % 50 == 0` | `checkpoints/checkpoint_epoch_200.pth` | 定期保存 |
| 250 | `250 % 50 == 0` | `checkpoints/checkpoint_epoch_250.pth` | 定期保存 |
| 300 | `300 % 50 == 0` | `checkpoints/checkpoint_epoch_300.pth` | 定期保存 |
| 每个 epoch | - | `latest_checkpoint.pth` | 覆盖式更新 |

### 预期文件结构
```
runs/exp1/
├── training.log
├── metrics.json
├── training_args.json
├── best_model.pth              # 最佳模型（完整）
├── best_model_weights.pt       # 最佳模型权重（兼容）
├── latest_checkpoint.pth       # 最新状态（用于恢复训练）
└── checkpoints/
    ├── checkpoint_epoch_050.pth
    ├── checkpoint_epoch_100.pth
    ├── checkpoint_epoch_150.pth
    ├── checkpoint_epoch_200.pth
    ├── checkpoint_epoch_250.pth
    └── checkpoint_epoch_300.pth
```

**结果**：`checkpoints/` 目录只有 6 个文件（300 / 50 = 6），而不是 300 个！

## 🔍 代码审查

### train_glint_unet.py 第 595-614 行（训练循环）

```python
# 保存最优模型
is_best = va["total"] < best_loss
if is_best:
    best_loss = va["total"]
    checkpoint_mgr.save_checkpoint(
        epoch, model, optimizer, args,
        {'train': tr, 'val': va},
        is_best=True  # ✅ 保存到 best_model.pth
    )
    logger.log(f"  ✅ 保存最佳模型 (Val Loss: {best_loss:.4f})")

# 定期保存 checkpoint
if epoch % args.save_freq == 0:
    checkpoint_mgr.save_checkpoint(
        epoch, model, optimizer, args,
        {'train': tr, 'val': va},
        filename=f"checkpoint_epoch_{epoch:03d}.pth"  # ✅ 保存到 checkpoints/
    )
    logger.log(f"  💾 保存 checkpoint (Epoch {epoch})")

# 总是保存最新的 checkpoint
checkpoint_mgr.save_latest(epoch, model, optimizer, args, {'train': tr, 'val': va})
```

## ✅ 验证修复

运行训练后检查文件：

```bash
# 启动训练
bash train.sh

# 训练一段时间后检查 checkpoints 数量
ls -lh runs/exp1/checkpoints/ | wc -l

# 预期结果：文件数 = (当前 epoch / save_freq)
# 例如：epoch 100, save_freq 50 → 2 个文件
```

## 🎉 修复完成

- ✅ `is_best=True` → 只保存到根目录 `best_model.pth`
- ✅ `epoch % save_freq == 0` → 保存到 `checkpoints/checkpoint_epoch_XXX.pth`
- ✅ 每个 epoch → 覆盖式保存 `latest_checkpoint.pth`
- ✅ 修复了默认参数 `is_best=False`

**现在 checkpoint 保存完全按照 `save_freq` 参数工作！**
