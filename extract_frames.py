# -*- coding: utf-8 -*-
"""
按帧间隔抽取图片
从输入目录读取图片，按指定间隔抽取并保存到输出目录
"""

import os
import shutil
import argparse
from pathlib import Path


def extract_frames(input_dir, frame_interval, output_dir):
    """
    按帧间隔抽取图片
    
    Args:
        input_dir: 输入目录路径
        frame_interval: 帧间隔（每隔多少帧抽取一张）
        output_dir: 输出目录路径
    """
    # 支持的图片格式
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # 获取所有图片文件并排序
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files, key=lambda x: x.name)
    
    if not image_files:
        print(f"错误：在目录 '{input_dir}' 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"帧间隔: {frame_interval}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按间隔抽取图片
    extracted_count = 0
    for i in range(0, len(image_files), frame_interval):
        src_file = image_files[i]
        dst_file = Path(output_dir) / src_file.name
        
        # 复制文件
        shutil.copy2(src_file, dst_file)
        extracted_count += 1
        
        if extracted_count % 10 == 0:
            print(f"已抽取 {extracted_count} 张图片...")
    
    print(f"\n完成！共抽取 {extracted_count} 张图片")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='按帧间隔抽取图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python extract_frames.py input_folder 5 output_folder
  python extract_frames.py ./pictures 10 ./pictures_sampled
        """
    )
    
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('frame_interval', type=int, help='帧间隔（每隔N帧抽取一张，N=1表示抽取所有帧）')
    parser.add_argument('output_dir', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.isdir(args.input_dir):
        print(f"错误：输入目录 '{args.input_dir}' 不存在")
        return
    
    if args.frame_interval < 1:
        print(f"错误：帧间隔必须 >= 1")
        return
    
    # 执行抽取
    extract_frames(args.input_dir, args.frame_interval, args.output_dir)


if __name__ == "__main__":
    main()
