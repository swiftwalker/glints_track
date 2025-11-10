import h5py
import numpy as np
import os

def inspect_h5_file(file_path):
    print(f"检查文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    print(f"文件存在: {os.path.exists(file_path)}")
    print()
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("=" * 50)
            print("数据集结构:")
            print("=" * 50)
            
            # 显示所有数据集和组
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 {name} - 数据集")
                    print(f"{indent}    形状: {obj.shape}")
                    print(f"{indent}    数据类型: {obj.dtype}")
                    print(f"{indent}    压缩: {obj.compression}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name} - 组")
            
            f.visititems(print_structure)
            
            print("\n" + "=" * 50)
            print("全局属性:")
            print("=" * 50)
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            
            print("\n" + "=" * 50)
            print("数据样本预览:")
            print("=" * 50)
            
            if 'images' in f:
                images = f['images']
                print(f"图像数据集: {images.shape}")
                if images.shape[0] > 0:
                    print(f"第一个样本 - 最小值: {images[0].min()}, 最大值: {images[0].max()}")
            
            if 'heatmaps' in f:
                heatmaps = f['heatmaps']
                print(f"热力图数据集: {heatmaps.shape}")
                if heatmaps.shape[0] > 0:
                    print(f"第一个样本 - 最小值: {heatmaps[0].min()}, 最大值: {heatmaps[0].max()}")
                    print(f"热力图非零像素数量: {np.count_nonzero(heatmaps[0] > 0.1)}")
            
            if 'present' in f:
                present = f['present']
                print(f"存在标记数据集: {present.shape}")
                if present.shape[0] > 0:
                    print(f"第一个样本的存在标记: {present[0]}")
            
            if 'stems' in f:
                stems = f['stems']
                print(f"样本名数据集: {stems.shape}")
                if stems.shape[0] > 0:
                    print(f"前5个样本名: {[s.decode('utf-8') for s in stems[:5]]}")
                    
    except Exception as e:
        print(f"❌ 打开文件时出错: {e}")
        return False
    
    return True

if __name__ == "__main__":
    file_path = "glints_dataset.h5"
    success = inspect_h5_file(file_path)
    
    if success:
        print("\n✅ 文件结构完整")
    else:
        print("\n❌ 文件可能损坏")