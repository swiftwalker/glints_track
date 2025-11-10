# GPU 参数添加完成总结

## ✅ 已完成的修改

### 1. **train_glint_unet.py**
- ✅ 添加 `--gpu` 参数到参数解析器
- ✅ 在设备选择前设置 `CUDA_VISIBLE_DEVICES` 环境变量
- ✅ 添加 GPU 信息日志输出

### 2. **文档更新**
- ✅ `TRAINING_GUIDE.md` - 添加 GPU 参数说明和使用场景
- ✅ `OPTIMIZATION_SUMMARY.md` - 添加 GPU 参数到参数表和使用示例
- ✅ `GPU_USAGE.md` - 创建详细的 GPU 使用指南

### 3. **脚本更新**
- ✅ `train_with_shm.sh` - 添加 GPU 使用示例场景

### 4. **测试**
- ✅ `test_gpu_param.py` - 创建 GPU 参数功能测试脚本
- ✅ 验证单 GPU、多 GPU 和不指定 GPU 的场景

## 📝 代码修改详情

### train_glint_unet.py 第 441 行
```python
# 添加 GPU 参数
ap.add_argument("--gpu", type=str, default=None, help="指定 GPU 设备 (如 '0', '1', '0,1' 等)，不指定则自动选择")
```

### train_glint_unet.py 第 530-538 行
```python
# 设置 GPU 设备
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger.log(f"指定 GPU 设备: {args.gpu}")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.log(f"使用设备: {device}")
if device == "cuda":
    logger.log(f"GPU 设备数量: {torch.cuda.device_count()}")
    logger.log(f"当前 GPU: {torch.cuda.get_device_name(0)}")
```

## 🚀 使用方法

### 基础用法
```bash
# 使用 GPU 0
python train_glint_unet.py --h5 train_256.h5 --gpu 0 --epochs 50

# 使用 GPU 1
python train_glint_unet.py --h5 train_256.h5 --gpu 1 --epochs 50

# 使用多个 GPU
python train_glint_unet.py --h5 train_256.h5 --gpu 0,1 --epochs 50
```

### 多实验并行（推荐用法）
```bash
# 在不同 GPU 上同时运行多个实验
python train_glint_unet.py --h5 train_256.h5 --output_dir exp1 --gpu 0 --epochs 50 &
python train_glint_unet.py --h5 train_256.h5 --output_dir exp2 --gpu 1 --epochs 50 &
python train_glint_unet.py --h5 train_256.h5 --output_dir exp3 --gpu 2 --epochs 50 &
wait
```

### 结合共享内存
```bash
# 多 GPU + 共享内存 = 最优性能
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal \
    --gpu 0 \
    --shared_memory \
    --loss focal &

python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/hybrid \
    --gpu 1 \
    --shared_memory \
    --loss hybrid &

wait
```

## 🧪 测试结果

### 测试 1: 不指定 GPU
```bash
$ python test_gpu_param.py
GPU 设备数量: 8  ✅
```

### 测试 2: 指定单个 GPU
```bash
$ python test_gpu_param.py --gpu 2
指定 GPU 设备: 2
GPU 设备数量: 1  ✅
```

### 测试 3: 指定多个 GPU
```bash
$ python test_gpu_param.py --gpu 0,1
指定 GPU 设备: 0,1
GPU 设备数量: 2  ✅
```

## 📊 参数完整列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpu` | str | None | 指定 GPU 设备（如 '0', '1', '0,1'）|
| `--output_dir` | str | auto | 输出目录 |
| `--resume` | str | None | 恢复训练的 checkpoint |
| `--save_freq` | int | 10 | Checkpoint 保存频率 |
| `--shared_memory` | flag | False | 使用共享内存 |
| `--num_workers` | int | 4 | DataLoader 工作进程数 |

## 🎯 典型使用场景

### 场景 1: 单实验训练
适合：快速测试、模型调试
```bash
python train_glint_unet.py --h5 train_256.h5 --gpu 0 --epochs 10
```

### 场景 2: 多实验对比
适合：超参数搜索、Loss 函数对比
```bash
python train_glint_unet.py --h5 train_256.h5 --gpu 0 --loss focal --output_dir exp1 &
python train_glint_unet.py --h5 train_256.h5 --gpu 1 --loss hybrid --output_dir exp2 &
wait
```

### 场景 3: 长时间训练
适合：正式训练、完整实验
```bash
python train_glint_unet.py \
    --h5 train_256.h5 \
    --gpu 0 \
    --output_dir experiments/final_model \
    --epochs 200 \
    --save_freq 5 \
    --shared_memory
```

## ✨ 新功能亮点

1. **灵活的 GPU 选择**
   - 支持单 GPU、多 GPU 或自动选择
   - 语法简单：`--gpu 0` 或 `--gpu 0,1`

2. **完善的日志输出**
   - 自动记录使用的 GPU 设备
   - 显示 GPU 数量和型号信息

3. **多实验并行**
   - 轻松在不同 GPU 上运行多个实验
   - 结合 `&` 和 `wait` 实现并行管理

4. **与现有功能完美集成**
   - 兼容 checkpoint 系统
   - 兼容共享内存功能
   - 兼容所有训练参数

## 📚 相关文档

- `GPU_USAGE.md` - GPU 参数详细使用指南
- `TRAINING_GUIDE.md` - 完整训练系统指南
- `OPTIMIZATION_SUMMARY.md` - 优化功能总结
- `SHARED_MEMORY_USAGE.md` - 共享内存使用说明

## 🎉 完成状态

- ✅ 功能实现完成
- ✅ 测试验证通过
- ✅ 文档编写完成
- ✅ 示例脚本更新
- ✅ 与现有功能集成

**可以立即使用！**
