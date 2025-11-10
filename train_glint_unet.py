import torch, h5py, random, os, argparse, hashlib, json, time
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from multiprocessing import shared_memory
from unet_glint import UNet
from losses import build_loss
from tqdm import tqdm

# ------------------------------
# Checkpoint 和日志管理
# ------------------------------
class TrainingLogger:
    """训练日志记录器（不包含 tqdm 进度条信息）"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "training.log")
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        self.metrics_history = []
        
        # 创建日志文件并写入表头 
        with open(self.log_file, 'w') as f:
            f.write(f"训练日志 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message, print_console=True):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        # 打印到控制台
        if print_console:
            print(log_message)
    
    def log_epoch(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """记录 epoch 指标"""
        message = f"\nEpoch {epoch:03d} (耗时: {epoch_time:.2f}s, LR: {lr:.2e}):\n"
        message += f"  Train: total={train_metrics['total']:.4f} focal={train_metrics['focal']:.4f} "
        message += f"bce={train_metrics['bce']:.4f} dice={train_metrics['dice']:.4f} div={train_metrics['div']:.4f}\n"
        message += f"  Val:   total={val_metrics['total']:.4f} focal={val_metrics['focal']:.4f} "
        message += f"bce={val_metrics['bce']:.4f} dice={val_metrics['dice']:.4f} div={val_metrics['div']:.4f}"
        
        self.log(message)
        
        # 保存指标到 JSON
        self.metrics_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': lr,
            'epoch_time': epoch_time
        })
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class CheckpointManager:
    """Checkpoint 管理器"""
    def __init__(self, output_dir, save_optimizer=True):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        self.save_optimizer = save_optimizer
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch, model, optimizer, train_args, metrics, 
                       is_best=False, filename=None):
        """
        保存完整的 checkpoint
        
        Args:
            epoch: 当前 epoch
            model: 模型
            optimizer: 优化器
            train_args: 训练参数（argparse.Namespace 或 dict）
            metrics: 当前指标
            is_best: 是否是最佳模型
            filename: 自定义文件名
        """
        # 转换 train_args 为 dict
        if hasattr(train_args, '__dict__'):
            train_args_dict = vars(train_args)
        else:
            train_args_dict = train_args
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_args': train_args_dict,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 根据配置决定是否保存优化器状态
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 如果是最佳模型，保存到根目录
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            
            # 同时保存纯模型权重（兼容旧代码）
            model_only_path = os.path.join(self.output_dir, "best_model_weights.pt")
            torch.save(model.state_dict(), model_only_path)
            
            return best_path
        else:
            # 定期保存到 checkpoints 目录
            if filename is None:
                filename = f"checkpoint_epoch_{epoch:03d}.pth"
            
            checkpoint_path = os.path.join(self.checkpoints_dir, filename)
            torch.save(checkpoint, checkpoint_path)
            
            return checkpoint_path
    
    def save_latest(self, epoch, model, optimizer, train_args, metrics):
        """保存为 latest checkpoint（覆盖式）"""
        latest_path = os.path.join(self.output_dir, "latest_checkpoint.pth")
        
        if hasattr(train_args, '__dict__'):
            train_args_dict = vars(train_args)
        else:
            train_args_dict = train_args
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_args': train_args_dict,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 根据配置决定是否保存优化器状态
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, latest_path)
        return latest_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, device='cuda'):
        """
        加载 checkpoint
        
        Returns:
            dict: 包含 epoch, train_args, metrics 的字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'train_args': checkpoint.get('train_args', {}),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
    
    def get_latest_checkpoint(self):
        """获取最新的 checkpoint 路径"""
        latest_path = os.path.join(self.output_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def save_training_args(self, args):
        """保存训练参数到 JSON"""
        args_path = os.path.join(self.output_dir, "training_args.json")
        
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = args
        
        # 过滤掉不可序列化的对象
        serializable_args = {}
        for key, value in args_dict.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_args[key] = value
            else:
                serializable_args[key] = str(value)
        
        with open(args_path, 'w') as f:
            json.dump(serializable_args, f, indent=2)
        
        return args_path

# ------------------------------
# 共享内存数据集缓存管理
# ------------------------------
class SharedMemoryDatasetCache:
    """
    使用共享内存缓存数据集，支持多进程共享
    通过 MD5 校验唯一标识数据集
    """
    @staticmethod
    def compute_md5(file_path, chunk_size=8192):
        """计算文件 MD5"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    
    @staticmethod
    def get_shm_name(h5_path, dataset_name):
        """根据文件路径和数据集名称生成共享内存名称"""
        md5_hash = SharedMemoryDatasetCache.compute_md5(h5_path)
        return f"glint_{md5_hash}_{dataset_name}"
    
    @staticmethod
    def try_attach_or_create(h5_path, dataset_name, data_array=None):
        """
        尝试连接现有共享内存，如果不存在则创建
        
        Args:
            h5_path: HDF5 文件路径
            dataset_name: 数据集名称 (images/heatmaps)
            data_array: 如果需要创建，提供的数据数组
            
        Returns:
            (shm, np_array, is_new) - 共享内存对象、numpy 数组、是否新创建
        """
        shm_name = SharedMemoryDatasetCache.get_shm_name(h5_path, dataset_name)
        
        # 尝试连接已存在的共享内存
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            print(f"  ✓ 连接到现有共享内存: {shm_name}")
            
            # 需要知道形状和 dtype 才能创建 numpy 数组
            # 我们将形状和 dtype 信息存储在共享内存的前几个字节
            meta_size = 64  # 预留 64 字节存储元数据
            meta_bytes = bytes(shm.buf[:meta_size])
            
            # 解析元数据: ndim(4) + shape(8*ndim) + dtype_len(4) + dtype_str
            ndim = int.from_bytes(meta_bytes[0:4], 'little')
            shape = tuple(int.from_bytes(meta_bytes[4+i*8:4+(i+1)*8], 'little') 
                         for i in range(ndim))
            dtype_len = int.from_bytes(meta_bytes[4+ndim*8:8+ndim*8], 'little')
            dtype_str = meta_bytes[8+ndim*8:8+ndim*8+dtype_len].decode('utf-8')
            
            # 创建 numpy 数组视图
            np_array = np.ndarray(shape, dtype=np.dtype(dtype_str), 
                                 buffer=shm.buf[meta_size:])
            
            return shm, np_array, False
            
        except FileNotFoundError:
            # 共享内存不存在，创建新的
            if data_array is None:
                raise ValueError("需要提供 data_array 来创建新的共享内存")
            
            print(f"  ✓ 创建新的共享内存: {shm_name}")
            
            # 准备元数据
            shape = data_array.shape
            dtype_str = str(data_array.dtype)
            ndim = len(shape)
            
            meta_size = 64
            meta_bytes = bytearray(meta_size)
            meta_bytes[0:4] = ndim.to_bytes(4, 'little')
            for i, s in enumerate(shape):
                meta_bytes[4+i*8:4+(i+1)*8] = s.to_bytes(8, 'little')
            dtype_bytes = dtype_str.encode('utf-8')
            meta_bytes[4+ndim*8:8+ndim*8] = len(dtype_bytes).to_bytes(4, 'little')
            meta_bytes[8+ndim*8:8+ndim*8+len(dtype_bytes)] = dtype_bytes
            
            # 创建共享内存
            total_size = meta_size + data_array.nbytes
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_size)
            
            # 写入元数据
            shm.buf[:meta_size] = meta_bytes
            
            # 创建 numpy 数组视图并复制数据
            np_array = np.ndarray(shape, dtype=data_array.dtype, 
                                 buffer=shm.buf[meta_size:])
            np_array[:] = data_array[:]
            
            print(f"    形状: {shape}, dtype: {dtype_str}, 大小: {total_size / 1024**2:.1f} MB")
            
            return shm, np_array, True
    
    @staticmethod
    def cleanup(h5_path, dataset_names=['images', 'heatmaps']):
        """清理共享内存（在程序结束时调用）"""
        for dataset_name in dataset_names:
            shm_name = SharedMemoryDatasetCache.get_shm_name(h5_path, dataset_name)
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                print(f"  ✓ 清理共享内存: {shm_name}")
            except FileNotFoundError:
                pass

# ------------------------------
# Dataset
# ------------------------------
class GlintH5(Dataset):
    def __init__(self, path, preload=True, use_shared_memory=False):
        """
        Args:
            path: HDF5 文件路径
            preload: 是否预加载所有数据到内存
            use_shared_memory: 是否使用共享内存（支持多进程共享数据）
        """
        self.path = path
        self.shm_imgs = None
        self.shm_hms = None
        
        f = h5py.File(path, "r")
        
        if preload and use_shared_memory:
            # 使用共享内存加载数据
            print(f"使用共享内存加载数据集: {path}")
            print(f"  MD5: {SharedMemoryDatasetCache.compute_md5(path)[:16]}...")
            
            # 尝试连接或创建 images 共享内存
            imgs_data = f["images"][:]
            self.shm_imgs, self.imgs, is_new_imgs = \
                SharedMemoryDatasetCache.try_attach_or_create(path, "images", imgs_data)
            
            # 尝试连接或创建 heatmaps 共享内存
            hms_data = f["heatmaps"][:].astype("float32")
            self.shm_hms, self.hms, is_new_hms = \
                SharedMemoryDatasetCache.try_attach_or_create(path, "heatmaps", hms_data)
            
            f.close()
            
            if is_new_imgs or is_new_hms:
                print(f"  ✓ 数据已加载到共享内存，其他进程可直接复用")
            else:
                print(f"  ✓ 已复用现有共享内存中的数据")
                
        elif preload:
            # 普通内存加载（不共享）
            print(f"预加载数据集到内存: {path}")
            self.imgs = f["images"][:]
            self.hms = f["heatmaps"][:].astype("float32")
            f.close()
            print(f"  图像数据: {self.imgs.shape}, {self.imgs.dtype}")
            print(f"  热力图数据: {self.hms.shape}, {self.hms.dtype}")
        else:
            # 保持文件打开，按需读取（适用于大数据集）
            self.f = f
            self.imgs = f["images"]
            self.hms = f["heatmaps"]
    
    def __len__(self): 
        return self.imgs.shape[0]
        
    def __getitem__(self, i):
        img = torch.from_numpy(self.imgs[i]).float().unsqueeze(0) / 255.0
        hm = torch.from_numpy(self.hms[i] if isinstance(self.hms[i], np.ndarray) else self.hms[i][:].astype("float32"))
        return img, hm
    
    def __del__(self):
        """析构时关闭共享内存连接（但不 unlink）"""
        if self.shm_imgs is not None:
            try:
                self.shm_imgs.close()
            except:
                pass
        if self.shm_hms is not None:
            try:
                self.shm_hms.close()
            except:
                pass

# ------------------------------
# Training Loop
# ------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, logger=None):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for imgs, targets in tqdm(loader, desc="Train", ncols=90, disable=logger is not None):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = loss_fn(out, targets)      # ← 返回标量张量

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches

def validate(model, loader, loss_fn, device, logger=None):
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", ncols=90, disable=logger is not None):
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            loss = loss_fn(out, targets)
            total_loss += loss.item()

    return total_loss / n_batches

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    # 数据和输出相关
    ap.add_argument("--h5", required=True, help="HDF5 数据集路径")
    ap.add_argument("--output_dir", default=None, help="输出目录（保存 checkpoints、日志等）")
    ap.add_argument("--resume", default=None, help="恢复训练的 checkpoint 路径")
    
    # 训练参数
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--gpu", type=str, default=None, help="指定 GPU 设备 (如 '0', '1', '0,1' 等)，不指定则自动选择")
    
    # 数据加载优化
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作进程数（推荐 4-8）")
    ap.add_argument("--preload", action="store_true", default=True, help="预加载数据到内存")
    ap.add_argument("--no_preload", action="store_false", dest="preload", help="不预加载数据（大数据集使用）")
    ap.add_argument("--shared_memory", action="store_true", help="使用共享内存（多进程训练可复用数据）")
    ap.add_argument("--cleanup_shm", action="store_true", help="清理共享内存后退出")
    
    # Checkpoint 相关
    ap.add_argument("--save_freq", type=int, default=10, help="每 N 个 epoch 保存一次 checkpoint")
    ap.add_argument("--keep_last_n", type=int, default=3, help="保留最近 N 个 checkpoint")
    ap.add_argument("--no_save_optimizer", action="store_true", help="不保存优化器状态（减小文件大小）")
    
    # 兼容旧参数
    ap.add_argument("--save", default="checkpoints", help="[已弃用] 使用 --output_dir 替代")
    ap.add_argument("--model_path", default="best.pt", help="[已弃用] 使用 --output_dir 替代")
    ap.add_argument("--loss", default="focal", choices=["focal","bce","dice","hybrid"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--lam_focal", type=float, default=1.0)
    ap.add_argument("--lam_bce",   type=float, default=0)
    ap.add_argument("--lam_dice",  type=float, default=0)
    ap.add_argument("--div_weight", type=float, default=0.05, help="相似度惩罚系数")
    ap.add_argument("--div_mode", default="cosine", choices=["overlap", "cosine", "kl"], help="相似度惩罚模式")
    ap.add_argument("--lam_agg", type=float, default=0.2, help="聚合类无关项总权重")
    ap.add_argument("--agg_mode", default="max", choices=["max","sum"], help="聚合方式：max或sum-clip")
    ap.add_argument("--agg_wF", type=float, default=1.0, help="聚合项内部 Focal 占比")
    ap.add_argument("--agg_wB", type=float, default=0, help="聚合项内部 BCE   占比")
    ap.add_argument("--agg_wD", type=float, default=0, help="聚合项内部 Dice  占比")
    args = ap.parse_args()
    
    # 如果只是清理共享内存，执行清理后退出
    if args.cleanup_shm:
        print(f"清理共享内存: {args.h5}")
        SharedMemoryDatasetCache.cleanup(args.h5)
        print("完成！")
        return
    
    # 确定输出目录
    if args.output_dir is None:
        # 自动生成输出目录名（基于时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"runs/train_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化日志和 checkpoint 管理器
    logger = TrainingLogger(args.output_dir)
    checkpoint_mgr = CheckpointManager(args.output_dir, save_optimizer=not args.no_save_optimizer)
    
    logger.log("=" * 80)
    logger.log("开始训练")
    logger.log("=" * 80)
    logger.log(f"输出目录: {args.output_dir}")
    logger.log(f"数据集: {args.h5}")
    
    # 保存训练参数
    checkpoint_mgr.save_training_args(args)
    logger.log(f"训练参数已保存到: {os.path.join(args.output_dir, 'training_args.json')}")

    # 创建数据集（支持预加载和共享内存）
    logger.log("\n加载数据集...")
    dataset = GlintH5(args.h5, preload=args.preload, use_shared_memory=args.shared_memory)
    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    
    # 创建 DataLoader（启用多进程和 pin_memory 优化）
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch,
        num_workers=max(1, args.num_workers // 2),  # 验证时用较少的 worker
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    logger.log(f"训练集样本数: {n_train}, 验证集样本数: {n_val}")
    logger.log(f"Batch size: {args.batch}, Workers: {args.num_workers}")

    # 设置 GPU 设备
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.log(f"指定 GPU 设备: {args.gpu}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"使用设备: {device}")
    if device == "cuda":
        logger.log(f"GPU 设备数量: {torch.cuda.device_count()}")
        logger.log(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建模型和优化器
    model = UNet(in_ch=1, out_ch=8).to(device)
    loss_fn = build_loss(
        args.loss,
        alpha=args.alpha, gamma=args.gamma,
        lam_focal=args.lam_focal, lam_bce=args.lam_bce, lam_dice=args.lam_dice,
        div_weight=args.div_weight, div_mode=args.div_mode,
        lam_agg=args.lam_agg, agg_mode=args.agg_mode,
        agg_weights=(args.agg_wF, args.agg_wB, args.agg_wD)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logger.log(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"Loss 函数: {args.loss}")
    logger.log(f"保存优化器状态: {'否' if args.no_save_optimizer else '是'}")
    
    # 恢复训练（如果指定）
    start_epoch = 1
    best_loss = float("inf")
    
    if args.resume:
        logger.log(f"\n恢复训练: {args.resume}")
        try:
            checkpoint_info = checkpoint_mgr.load_checkpoint(args.resume, model, optimizer, device)
            start_epoch = checkpoint_info['epoch'] + 1
            best_loss = checkpoint_info['metrics'].get('val', {}).get('total', float("inf"))
            logger.log(f"  从 Epoch {checkpoint_info['epoch']} 恢复")
            logger.log(f"  最佳验证 Loss: {best_loss:.4f}")
            logger.log(f"  Checkpoint 时间: {checkpoint_info['timestamp']}")
        except Exception as e:
            logger.log(f"  ⚠️  恢复失败: {e}")
            logger.log("  将从头开始训练")
            start_epoch = 1
            best_loss = float("inf")
    
    logger.log(f"\n开始训练: Epoch {start_epoch} -> {args.epochs}")
    logger.log("=" * 80 + "\n")
    
    # 训练循环
        # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练和验证
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, logger)
        val_loss = validate(model, val_loader, loss_fn, device, logger)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 创建兼容的指标字典（为了日志记录）
        tr_metrics = {"total": train_loss, "focal": train_loss, "bce": 0.0, "dice": 0.0, "div": 0.0}
        va_metrics = {"total": val_loss, "focal": val_loss, "bce": 0.0, "dice": 0.0, "div": 0.0}
        
        # 记录日志
        logger.log_epoch(epoch, tr_metrics, va_metrics, current_lr, epoch_time)
        
        # 保存最优模型
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            checkpoint_mgr.save_checkpoint(
                epoch, model, optimizer, args,
                {'train': tr_metrics, 'val': va_metrics},
                is_best=True
            )
            logger.log(f"  ✅ 保存最佳模型 (Val Loss: {best_loss:.4f})")
        
        # 定期保存 checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_mgr.save_checkpoint(
                epoch, model, optimizer, args,
                {'train': tr_metrics, 'val': va_metrics},
                filename=f"checkpoint_epoch_{epoch:03d}.pth"
            )
            logger.log(f"  💾 保存 checkpoint (Epoch {epoch})")
        
        # 总是保存最新的 checkpoint
        checkpoint_mgr.save_latest(epoch, model, optimizer, args, {'train': tr_metrics, 'val': va_metrics})
    
    logger.log("\n" + "=" * 80)
    logger.log("训练完成！")
    logger.log(f"最佳验证 Loss: {best_loss:.4f}")
    logger.log(f"模型保存位置: {args.output_dir}")
    logger.log("=" * 80)

if __name__ == "__main__":
    main()
