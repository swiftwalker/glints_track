# 共享内存数据集缓存功能说明

## 📖 功能概述

共享内存数据集缓存允许多个训练进程共享同一份内存中的数据集，通过 MD5 校验唯一标识数据集。

### 优势：
- ✅ **内存节省**：多进程只需加载一次数据
- ✅ **快速启动**：后续进程直接连接已有共享内存
- ✅ **自动管理**：MD5 校验确保数据一致性
- ✅ **跨进程复用**：适合多模型并行训练、超参数搜索等场景

## 🚀 使用方法

### 基础用法

```bash
# 启用共享内存（默认也会预加载）
python train_glint_unet.py --h5 train_256.h5 --shared_memory

# 同时指定其他参数
python train_glint_unet.py \
    --h5 train_256.h5 \
    --shared_memory \
    --num_workers 6 \
    --batch 16 \
    --epochs 50
```

### 多进程并行训练

```bash
# 进程 1：focal loss
python train_glint_unet.py \
    --h5 train_256.h5 \
    --shared_memory \
    --loss focal \
    --model_path model_focal.pt &

# 进程 2：hybrid loss（复用进程 1 加载的数据）
python train_glint_unet.py \
    --h5 train_256.h5 \
    --shared_memory \
    --loss hybrid \
    --model_path model_hybrid.pt &

wait
```

### 清理共享内存

训练完成后清理共享内存：

```bash
# 方法 1：使用训练脚本
python train_glint_unet.py --h5 train_256.h5 --cleanup_shm

# 方法 2：使用管理工具
python shm_manager.py --cleanup --h5 train_256.h5

# 方法 3：清理所有共享内存
python shm_manager.py --cleanup-all
```

## 🔍 共享内存管理工具

`shm_manager.py` 提供了查看和管理共享内存的功能：

### 查看共享内存状态

```bash
# 列出所有 glint_* 共享内存
python shm_manager.py --list

# 检查特定文件的共享内存
python shm_manager.py --list --h5 train_256.h5
```

输出示例：
```
文件: train_256.h5
MD5: a1b2c3d4e5f6...

检查共享内存状态:
------------------------------------------------------------
✓ images       - glint_a1b2c3d4e5f6_images
  大小: 245.3 MB
✓ heatmaps     - glint_a1b2c3d4e5f6_heatmaps
  大小: 122.7 MB
```

### 清理共享内存

```bash
# 清理特定文件的共享内存
python shm_manager.py --cleanup --h5 train_256.h5

# 清理所有共享内存
python shm_manager.py --cleanup-all
```

## 🎯 工作原理

1. **MD5 校验**：计算 HDF5 文件的 MD5 值作为唯一标识
2. **共享内存命名**：`glint_{md5}_{dataset_name}`
3. **自动检测**：
   - 第一个进程：创建共享内存并加载数据
   - 后续进程：检测到已存在的共享内存，直接连接
4. **元数据存储**：在共享内存前 64 字节存储 shape 和 dtype 信息

## 📊 性能对比

| 场景 | 普通预加载 | 共享内存 | 内存节省 |
|------|-----------|---------|---------|
| 单进程 | 1x 内存 | 1x 内存 | 0% |
| 2 进程 | 2x 内存 | 1x 内存 | 50% |
| 4 进程 | 4x 内存 | 1x 内存 | 75% |

数据集大小：train_256.h5 ≈ 368 MB
- 4 进程普通预加载：~1.5 GB
- 4 进程共享内存：~368 MB

## ⚠️ 注意事项

### 1. 何时使用共享内存

**适合：**
- 多模型并行训练
- 超参数搜索
- 需要频繁重启训练
- 多个实验同时运行

**不适合：**
- 单进程训练（无额外优势）
- 数据集过大超过可用内存
- 数据需要频繁更新

### 2. 内存管理

- 共享内存在 Linux 上存储在 `/dev/shm`
- 进程结束后共享内存依然存在（需手动清理）
- 定期检查：`df -h /dev/shm`

### 3. 数据一致性

- MD5 校验确保文件一致性
- 如果 HDF5 文件更新，MD5 会改变，自动创建新的共享内存
- 旧的共享内存需要手动清理

### 4. 系统限制

查看系统共享内存限制：
```bash
# 最大共享内存大小
cat /proc/sys/kernel/shmmax

# 总共享内存限制
cat /proc/sys/kernel/shmall

# 当前使用情况
ipcs -m
```

如需调整限制，编辑 `/etc/sysctl.conf`：
```
kernel.shmmax = 17179869184  # 16 GB
kernel.shmall = 4194304      # 页数
```

## 🛠️ 故障排除

### 问题 1：共享内存已满

```bash
# 查看使用情况
df -h /dev/shm

# 清理旧的共享内存
python shm_manager.py --cleanup-all
```

### 问题 2：权限错误

```bash
# 检查 /dev/shm 权限
ls -la /dev/shm

# 如有需要，清理自己创建的共享内存
python shm_manager.py --cleanup-all
```

### 问题 3：连接已有共享内存失败

- 可能是元数据损坏
- 解决方法：清理并重新创建
```bash
python shm_manager.py --cleanup --h5 train_256.h5
```

## 📝 示例脚本

参考 `train_with_shm.sh` 了解完整使用示例。

## 🔗 相关参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--shared_memory` | False | 启用共享内存 |
| `--preload` | True | 预加载数据（共享内存需要此项） |
| `--cleanup_shm` | False | 清理共享内存后退出 |
| `--num_workers` | 4 | DataLoader 工作进程数 |

## 💡 最佳实践

1. **首次运行**：使用 `--shared_memory` 创建共享内存
2. **并行训练**：所有进程都加 `--shared_memory`
3. **完成后清理**：使用 `shm_manager.py --cleanup-all`
4. **定期检查**：`python shm_manager.py --list`
5. **内存监控**：`watch -n 1 "df -h /dev/shm"`
