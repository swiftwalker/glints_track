#!/bin/bash
# 训练脚本示例

# 场景 1: 基础训练（自动创建输出目录）
echo "=== 场景 1: 基础训练 ==="
python train_glint_unet.py \
    --h5 train_256.h5 \
    --epochs 50 \
    --batch 16 \
    --num_workers 6 \
    --lr 1e-3

# 场景 2: 指定输出目录
echo -e "\n=== 场景 2: 指定输出目录 ==="
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal_exp1 \
    --epochs 50 \
    --batch 16 \
    --loss focal

# 场景 3: 恢复训练
echo -e "\n=== 场景 3: 恢复训练 ==="
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal_exp1 \
    --resume runs/focal_exp1/latest_checkpoint.pth \
    --epochs 100

# 场景 4: 多进程并行训练（使用共享内存）
echo -e "\n=== 场景 4: 多进程并行训练 ==="

# 进程 1: focal loss
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/focal_exp \
    --epochs 50 \
    --batch 16 \
    --shared_memory \
    --loss focal &

# 进程 2: hybrid loss  
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/hybrid_exp \
    --epochs 50 \
    --batch 16 \
    --shared_memory \
    --loss hybrid &

wait

# 场景 5: 指定 GPU 训练
echo -e "\n=== 场景 5: 指定 GPU 训练 ==="

# 在 GPU 0 上训练
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/gpu0_exp \
    --gpu 0 \
    --epochs 50 \
    --batch 16

# 场景 6: 多 GPU 并行训练（不同实验在不同 GPU）
echo -e "\n=== 场景 6: 多 GPU 并行训练 ==="

# 在 GPU 0 上训练 focal
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/gpu0_focal \
    --gpu 0 \
    --epochs 50 \
    --loss focal \
    --shared_memory &

# 在 GPU 1 上训练 hybrid
python train_glint_unet.py \
    --h5 train_256.h5 \
    --output_dir runs/gpu1_hybrid \
    --gpu 1 \
    --epochs 50 \
    --loss hybrid \
    --shared_memory &

wait

# 场景 7: 超参数搜索
echo -e "\n=== 场景 7: 超参数搜索 ==="
for lr in 1e-3 5e-4 1e-4; do
    python train_glint_unet.py \
        --h5 train_256.h5 \
        --output_dir runs/lr_${lr} \
        --epochs 50 \
        --lr ${lr} \
        --shared_memory &
done
wait

# 清理共享内存
echo -e "\n=== 清理共享内存 ==="
python shm_manager.py --cleanup-all

echo "完成！"
