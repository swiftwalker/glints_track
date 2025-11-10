# GPU 参数使用说明

## 功能

新增 `--gpu` 参数，允许指定训练使用的 GPU 设备。

## 语法

```bash
python train_glint_unet.py --gpu <device_id> [其他参数...]
```

- `<device_id>`: GPU 设备 ID，可以是：
  - 单个 GPU：`0`, `1`, `2`, 等
  - 多个 GPU：`0,1`, `0,1,2`, 等
  - 不指定：自动使用所有可用 GPU

## 使用示例

### 1. 使用单个 GPU

```bash
# 使用 GPU 0
python train_glint_unet.py \
    --h5 train_256.h5 \
    --gpu 0 \
    --epochs 50

# 使用 GPU 2
python train_glint_unet.py \
    --h5 train_256.h5 \
    --gpu 2 \
    --epochs 50
```

### 2. 使用多个 GPU

```bash
# 使用 GPU 0 和 1
python train_glint_unet.py \
    --h5 train_256.h5 \
    --gpu 0,1 \
    --epochs 50
```

### 3. 多实验并行（不同 GPU）

```bash
# 在 GPU 0 上运行实验 1
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/exp1 \
    --gpu 0 \
    --epochs 50 &

# 在 GPU 1 上运行实验 2
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/exp2 \
    --gpu 1 \
    --epochs 50 &

# 在 GPU 2 上运行实验 3
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/exp3 \
    --gpu 2 \
    --epochs 50 &

# 等待所有任务完成
wait
```

### 4. 结合共享内存（多 GPU + 数据共享）

```bash
# 在不同 GPU 上并行训练，共享同一份数据集
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

## 技术细节

- 内部实现：通过设置 `CUDA_VISIBLE_DEVICES` 环境变量
- 设置时机：在数据加载和模型创建之前
- 日志输出：训练日志会记录使用的 GPU 设备信息

## 日志示例

```
[2025-11-06 14:30:22] 指定 GPU 设备: 2
[2025-11-06 14:30:22] 使用设备: cuda
[2025-11-06 14:30:22] GPU 设备数量: 1
[2025-11-06 14:30:22] 当前 GPU: Tesla V100-SXM2-16GB
```

## 注意事项

1. **不指定 `--gpu`**：默认使用所有可用的 GPU
2. **多 GPU 训练**：当前代码使用单 GPU 训练，指定多个 GPU (如 `0,1`) 时会使用第一个
3. **并行实验**：使用后台运行 (`&`) 和 `wait` 命令来管理多个并行任务
4. **GPU 可用性**：确保指定的 GPU ID 在系统中可用

## 测试

运行测试脚本验证功能：

```bash
# 测试不指定 GPU
python test_gpu_param.py

# 测试指定单个 GPU
python test_gpu_param.py --gpu 2

# 测试指定多个 GPU
python test_gpu_param.py --gpu 0,1
```
