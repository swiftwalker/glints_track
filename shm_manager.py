#!/usr/bin/env python3
"""
共享内存管理工具
用于查看和清理 glints_track 训练使用的共享内存
"""
import argparse
import hashlib
import os
from multiprocessing import shared_memory


def compute_md5(file_path, chunk_size=8192):
    """计算文件 MD5"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def get_shm_name(h5_path, dataset_name):
    """根据文件路径和数据集名称生成共享内存名称"""
    md5_hash = compute_md5(h5_path)
    return f"glint_{md5_hash}_{dataset_name}"


def list_shared_memory(h5_path=None):
    """列出共享内存状态"""
    if h5_path:
        # 检查特定文件的共享内存
        if not os.path.exists(h5_path):
            print(f"错误: 文件不存在 {h5_path}")
            return
        
        md5_hash = compute_md5(h5_path)
        print(f"\n文件: {h5_path}")
        print(f"MD5: {md5_hash}")
        print(f"\n检查共享内存状态:")
        print("-" * 60)
        
        for dataset_name in ['images', 'heatmaps']:
            shm_name = get_shm_name(h5_path, dataset_name)
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                size_mb = shm.size / 1024**2
                print(f"✓ {dataset_name:12s} - {shm_name}")
                print(f"  大小: {size_mb:.1f} MB")
                shm.close()
            except FileNotFoundError:
                print(f"✗ {dataset_name:12s} - 未找到共享内存")
    else:
        # 列出所有 glint_ 开头的共享内存
        print("\n查找所有 glint_* 共享内存...")
        print("-" * 60)
        
        # 在 /dev/shm 中查找（Linux）
        shm_dir = "/dev/shm"
        if os.path.exists(shm_dir):
            found = False
            for name in os.listdir(shm_dir):
                if name.startswith("glint_"):
                    found = True
                    path = os.path.join(shm_dir, name)
                    size = os.path.getsize(path)
                    size_mb = size / 1024**2
                    print(f"✓ {name}")
                    print(f"  大小: {size_mb:.1f} MB")
            
            if not found:
                print("未找到任何 glint_* 共享内存")
        else:
            print("注意: /dev/shm 不存在，可能不是 Linux 系统")


def cleanup_shared_memory(h5_path):
    """清理共享内存"""
    if not os.path.exists(h5_path):
        print(f"错误: 文件不存在 {h5_path}")
        return
    
    md5_hash = compute_md5(h5_path)
    print(f"\n文件: {h5_path}")
    print(f"MD5: {md5_hash}")
    print(f"\n清理共享内存:")
    print("-" * 60)
    
    for dataset_name in ['images', 'heatmaps']:
        shm_name = get_shm_name(h5_path, dataset_name)
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            size_mb = shm.size / 1024**2
            shm.close()
            shm.unlink()
            print(f"✓ 已清理 {dataset_name:12s} ({size_mb:.1f} MB)")
        except FileNotFoundError:
            print(f"✗ {dataset_name:12s} - 未找到")


def cleanup_all():
    """清理所有 glint_* 共享内存"""
    print("\n清理所有 glint_* 共享内存...")
    print("-" * 60)
    
    shm_dir = "/dev/shm"
    if not os.path.exists(shm_dir):
        print("注意: /dev/shm 不存在，可能不是 Linux 系统")
        return
    
    found = False
    for name in os.listdir(shm_dir):
        if name.startswith("glint_"):
            found = True
            try:
                shm = shared_memory.SharedMemory(name=name)
                size_mb = shm.size / 1024**2
                shm.close()
                shm.unlink()
                print(f"✓ 已清理 {name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"✗ 清理失败 {name}: {e}")
    
    if not found:
        print("未找到任何 glint_* 共享内存")
    else:
        print("\n完成！")


def main():
    parser = argparse.ArgumentParser(
        description="共享内存管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有共享内存
  python shm_manager.py --list
  
  # 检查特定文件的共享内存
  python shm_manager.py --list --h5 train_256.h5
  
  # 清理特定文件的共享内存
  python shm_manager.py --cleanup --h5 train_256.h5
  
  # 清理所有共享内存
  python shm_manager.py --cleanup-all
        """
    )
    
    parser.add_argument("--list", action="store_true", help="列出共享内存状态")
    parser.add_argument("--cleanup", action="store_true", help="清理共享内存")
    parser.add_argument("--cleanup-all", action="store_true", help="清理所有 glint_* 共享内存")
    parser.add_argument("--h5", help="HDF5 文件路径")
    
    args = parser.parse_args()
    
    if args.cleanup_all:
        cleanup_all()
    elif args.cleanup:
        if not args.h5:
            print("错误: --cleanup 需要指定 --h5 参数")
            return
        cleanup_shared_memory(args.h5)
    elif args.list:
        list_shared_memory(args.h5)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
