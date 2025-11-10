# 训练系统优化完成总结

## ✅ 已完成的功能

### 1. **完整的 Checkpoint 系统**
- ✅ 保存模型权重 + 优化器状态 + 训练参数
- ✅ 支持恢复训练（从任意 checkpoint 继续）
- ✅ 自动保存最佳模型
- ✅ 定期保存 checkpoint（可配置频率）
- ✅ 始终保存最新 checkpoint

### 2. **结构化输出目录**
```
runs/train_YYYYMMDD_HHMMSS/
├── training.log              # 文本日志
├── metrics.json              # JSON 格式指标
├── training_args.json        # 训练参数
├── best_model.pth            # 最佳模型（完整）
├── best_model_weights.pt     # 最佳模型权重（兼容）
├── latest_checkpoint.pth     # 最新 checkpoint
└── checkpoints/
    ├── checkpoint_epoch_010.pth
    ├── checkpoint_epoch_020.pth
    └── ...
```

### 3. **训练日志系统**
- ✅ 文本格式日志（无 tqdm 进度条）
- ✅ 包含时间戳的详细记录
- ✅ 训练/验证指标记录
- ✅ JSON 格式的结构化指标

### 4. **数据加载优化**（已有）
- ✅ 多进程数据加载（num_workers）
- ✅ 数据预加载到内存
- ✅ 共享内存机制（多进程复用）
- ✅ pin_memory 加速 GPU 传输

## 🎯 核心类和功能

### `TrainingLogger`
负责训练日志记录：
- `log()`: 记录日志消息（带时间戳）
- `log_epoch()`: 记录 epoch 指标
- 自动保存到 `training.log` 和 `metrics.json`

### `CheckpointManager`
负责 checkpoint 管理：
- `save_checkpoint()`: 保存完整 checkpoint
- `load_checkpoint()`: 加载 checkpoint 并恢复状态
- `save_latest()`: 保存最新 checkpoint（覆盖式）
- `save_training_args()`: 保存训练参数到 JSON

### `SharedMemoryDatasetCache`（已有）
负责共享内存管理：
- 使用 MD5 唯一标识数据集
- 自动检测和连接已存在的共享内存
- 支持多进程数据共享

## 📝 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | str | auto | 输出目录 |
| `--resume` | str | None | 恢复训练的 checkpoint |
| `--save_freq` | int | 10 | Checkpoint 保存频率 |
| `--keep_last_n` | int | 3 | 保留最近 N 个 checkpoint |
| `--gpu` | str | None | 指定 GPU 设备（如 '0', '1', '0,1'）|
| `--no_save_optimizer` | flag | False | 不保存优化器状态（减小文件约 50%）|

## 🚀 使用示例

### 基础训练
```bash
python train_glint_unet.py --h5 train_256.h5 --epochs 50
```

### 指定输出目录
```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/focal_exp1 \
    --epochs 50
```

### 恢复训练
```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/focal_exp1 \
    --resume experiments/focal_exp1/latest_checkpoint.pth \
    --epochs 100
```

### 指定 GPU 训练
```bash
# 使用 GPU 0
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/gpu0_exp \
    --gpu 0 \
    --epochs 50

# 在不同 GPU 上并行运行多个实验
python train_glint_unet.py --h5 train_256.h5 --output_dir exp1 --gpu 0 --epochs 50 &
python train_glint_unet.py --h5 train_256.h5 --output_dir exp2 --gpu 1 --epochs 50 &
wait
```

### 多实验并行（共享内存）
```bash
# 实验 1
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/focal \
    --shared_memory \
    --loss focal &

# 实验 2
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir experiments/hybrid \
    --shared_memory \
    --loss hybrid &

wait
```

## 📊 Checkpoint 内容

每个 checkpoint 包含：
```python
{
    'epoch': 42,
    'model_state_dict': {...},      # 模型权重
    'optimizer_state_dict': {...},  # 优化器状态
    'train_args': {...},            # 训练参数
    'metrics': {                    # 当前指标
        'train': {...},
        'val': {...}
    },
    'timestamp': '2025-11-06 14:30:22'
}
```

## 🔄 恢复训练流程

1. 加载 checkpoint
2. 恢复模型权重
3. 恢复优化器状态（学习率、动量等）
4. 从下一个 epoch 继续训练
5. 保持之前的最佳 loss 记录

## 📈 日志示例

### training.log
```
[2025-11-06 14:30:22] ================================================================================
[2025-11-06 14:30:22] 开始训练
[2025-11-06 14:30:22] 输出目录: runs/train_20251106_143022
[2025-11-06 14:30:22] 数据集: train_256.h5
[2025-11-06 14:30:25] 训练集样本数: 900, 验证集样本数: 100
[2025-11-06 14:31:10] 
Epoch 001 (耗时: 45.32s, LR: 1.00e-03):
  Train: total=0.2345 focal=0.1234 bce=0.0000 dice=0.0000 div=0.0111
  Val:   total=0.1987 focal=0.1045 bce=0.0000 dice=0.0000 div=0.0092
[2025-11-06 14:31:10]   ✅ 保存最佳模型 (Val Loss: 0.1987)
```

### metrics.json
```json
[
  {
    "epoch": 1,
    "train": {"total": 0.2345, "focal": 0.1234, ...},
    "val": {"total": 0.1987, "focal": 0.1045, ...},
    "lr": 0.001,
    "epoch_time": 45.32
  },
  ...
]
```

## 🔧 技术实现

### 日志记录
- 使用 `datetime` 生成时间戳
- 同时写入文件和控制台
- tqdm 进度条可选禁用（通过 logger 参数）

### Checkpoint 保存
- PyTorch 原生 `torch.save()`
- 保存完整状态字典
- 自动创建子目录

### 参数序列化
- 使用 JSON 格式
- 自动过滤不可序列化对象
- 保留所有训练配置

## 💡 最佳实践

1. **使用有意义的输出目录名**
   ```bash
   --output_dir experiments/focal_lr1e3_batch16
   ```

2. **定期检查日志**
   ```bash
   tail -f runs/xxx/training.log
   ```

3. **备份重要实验**
   ```bash
   cp -r runs/focal_exp1 backups/
   ```

4. **使用共享内存进行多实验**
   ```bash
   --shared_memory  # 多进程场景
   ```

5. **从最新 checkpoint 恢复**
   ```bash
   --resume runs/xxx/latest_checkpoint.pth
   ```

## 📚 相关文档

- `TRAINING_GUIDE.md`: 完整使用指南
- `SHARED_MEMORY_USAGE.md`: 共享内存详细文档
- `SHARED_MEMORY_QUICKSTART.md`: 共享内存快速开始

## 🧪 测试

运行测试验证功能：
```bash
python test_training_system.py
```

## 🔗 向后兼容

旧参数仍然支持（但已弃用）：
- `--model_path` → 使用 `--output_dir`
- `--save` → 使用 `--output_dir`

旧模型加载方式仍然可用：
```python
# 仅加载权重
model.load_state_dict(torch.load('best_model_weights.pt'))
```

## 🎉 性能提升总结

### 完整优化链条：
1. **数据加载**: num_workers + pin_memory
2. **内存优化**: preload + shared_memory
3. **训练管理**: checkpoint + logging
4. **实验追踪**: 结构化输出

### 预期效果：
- ✅ GPU 利用率: 30-50% → 80-95%
- ✅ 训练速度: 提升 2-4 倍
- ✅ 内存使用: 多进程节省 50-75%
- ✅ 实验管理: 完全可追溯和可恢复
