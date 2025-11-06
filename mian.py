# -*- coding: utf-8 -*-
"""
Glint extraction (dynamic/adaptive threshold) for eye images.
Dependencies: opencv-python, numpy, PyQt5
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import cv2
import numpy as np
import sys
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog,
                             QGroupBox, QGridLayout, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# -------- Params --------
@dataclass
class GlintParams:
    denoise_ksize: int = 3                   # 中值滤波核
    tophat_ksize: int = 15                   # Top-hat 核(椭圆)
    adapt_blocksize: int = 41                # 自适应阈值局部窗口(奇数)
    adapt_C: int = 5                         # 阈值偏移，越大越严格
    open_ksize: int = 3                      # 开运算核
    dilate_ksize: int = 3                    # 膨胀核
    area_min: float = 5                      # 轮廓最小面积
    area_max: float = 3000                   # 轮廓最大面积
    circularity_min: float = 0.30            # 圆度阈值(4πA/P^2)
    roi_circle: Optional[Tuple[int,int,int]] = None  # 仅在该圆形ROI内保留 (cx,cy,r)

# -------- Core steps --------
def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def preprocess(img: np.ndarray, p: GlintParams) -> np.ndarray:
    blur = cv2.medianBlur(img, p.denoise_ksize)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.tophat_ksize, p.tophat_ksize))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, se)
    return tophat

def adaptive_mask(tophat: np.ndarray, p: GlintParams) -> np.ndarray:
    # 自适应阈值 + Otsu 交集
    mask_adapt = cv2.adaptiveThreshold(
        tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        p.adapt_blocksize, p.adapt_C
    )
    _, mask_otsu = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask_adapt, mask_otsu)
    return mask

def clean_mask(mask: np.ndarray, p: GlintParams) -> np.ndarray:
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize, p.open_ksize))
    se_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.dilate_ksize, p.dilate_ksize))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_o)
    m = cv2.morphologyEx(m, cv2.MORPH_DILATE, se_d)
    return m

def filter_contours(mask: np.ndarray, p: GlintParams) -> Tuple[np.ndarray, List[Tuple[int,int]], List[float], List[float]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    centers, areas, circs = [], [], []

    # ROI 限制（可选）
    def in_roi(x: int, y: int) -> bool:
        if p.roi_circle is None: return True
        cx, cy, r = p.roi_circle
        return (x - cx)**2 + (y - cy)**2 <= r*r

    for c in cnts:
        a = cv2.contourArea(c)
        if a < p.area_min or a > p.area_max:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0: 
            continue
        circ = 4.0*np.pi*a/(peri*peri)
        if circ < p.circularity_min:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0: 
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        if not in_roi(cx, cy): 
            continue

        cv2.drawContours(filtered, [c], -1, 255, -1)
        centers.append((cx, cy)); areas.append(a); circs.append(circ)
    return filtered, centers, areas, circs

def draw_overlay(gray: np.ndarray, centers: List[Tuple[int,int]], r:int=7) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x,y) in centers:
        cv2.circle(overlay, (x,y), r, (0,0,255), 2)
        cv2.circle(overlay, (x,y), 2, (0,0,255), -1)
    return overlay

# -------- Pipeline --------
def extract_glints(gray: np.ndarray, params: GlintParams = GlintParams()):
    th = preprocess(gray, params)
    m0 = adaptive_mask(th, params)
    m = clean_mask(m0, params)
    mask, centers, areas, circs = filter_contours(m, params)
    overlay = draw_overlay(gray, centers)
    return {
        "tophat": th,
        "mask_raw": m0,
        "mask": mask,
        "centers": centers,
        "areas": areas,
        "circularity": circs,
        "overlay": overlay
    }

# -------- Qt Interactive GUI --------
class GlintDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Glint Detector")
        self.setGeometry(100, 100, 1400, 900)
        
        self.gray_image = None
        self.params = GlintParams()
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_and_display)
        
        # 图片浏览器相关
        self.image_list = []
        self.current_index = 0
        self.current_dir = ""
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部：图片浏览控制栏
        nav_bar = self.create_navigation_bar()
        main_layout.addWidget(nav_bar)
        
        # 中间：主要内容区域
        content_layout = QHBoxLayout()
        
        # 左侧：参数控制面板
        control_panel = self.create_control_panel()
        content_layout.addWidget(control_panel, 1)
        
        # 右侧：图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        content_layout.addWidget(self.image_label, 3)
        
        main_layout.addLayout(content_layout)
    
    def create_navigation_bar(self):
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_widget.setStyleSheet("QWidget { background-color: #e0e0e0; padding: 5px; }")
        
        # 加载目录按钮
        load_dir_btn = QPushButton("Load Directory")
        load_dir_btn.clicked.connect(self.load_directory)
        nav_layout.addWidget(load_dir_btn)
        
        # 加载单张图片按钮
        load_img_btn = QPushButton("Load Single Image")
        load_img_btn.clicked.connect(self.load_image)
        nav_layout.addWidget(load_img_btn)
        
        nav_layout.addStretch()
        
        # 上一张按钮
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        # 图片索引跳转
        nav_layout.addWidget(QLabel("Image:"))
        self.index_spinbox = QSpinBox()
        self.index_spinbox.setMinimum(1)
        self.index_spinbox.setMaximum(1)
        self.index_spinbox.valueChanged.connect(self.jump_to_image)
        self.index_spinbox.setEnabled(False)
        nav_layout.addWidget(self.index_spinbox)
        
        self.total_label = QLabel("/ 0")
        nav_layout.addWidget(self.total_label)
        
        # 下一张按钮
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        
        # 文件名显示
        self.filename_label = QLabel("No image loaded")
        self.filename_label.setStyleSheet("QLabel { font-weight: bold; }")
        nav_layout.addWidget(self.filename_label)
        
        return nav_widget
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 参数保存/加载按钮组
        param_file_group = QGroupBox("Parameter Files")
        param_file_layout = QHBoxLayout()
        
        save_params_btn = QPushButton("Save Params")
        save_params_btn.clicked.connect(self.save_parameters)
        param_file_layout.addWidget(save_params_btn)
        
        load_params_btn = QPushButton("Load Params")
        load_params_btn.clicked.connect(self.load_parameters)
        param_file_layout.addWidget(load_params_btn)
        
        param_file_group.setLayout(param_file_layout)
        layout.addWidget(param_file_group)
        
        # 参数组
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout()
        
        self.sliders = {}
        row = 0
        
        # 定义参数及其范围
        param_configs = [
            ("denoise_ksize", "Denoise Kernel", 1, 15, 2, self.params.denoise_ksize),
            ("tophat_ksize", "Top-hat Kernel", 3, 51, 2, self.params.tophat_ksize),
            ("adapt_blocksize", "Adapt Block Size", 3, 101, 2, self.params.adapt_blocksize),
            ("adapt_C", "Adapt C", -20, 20, 1, self.params.adapt_C),
            ("open_ksize", "Open Kernel", 1, 15, 2, self.params.open_ksize),
            ("dilate_ksize", "Dilate Kernel", 1, 15, 2, self.params.dilate_ksize),
            ("area_min", "Min Area", 1, 100, 1, int(self.params.area_min)),
            ("area_max", "Max Area", 100, 5000, 100, int(self.params.area_max)),
            ("circularity_min", "Min Circularity", 0, 100, 1, int(self.params.circularity_min * 100)),
        ]
        
        for param_name, label_text, min_val, max_val, step, default_val in param_configs:
            # 标签
            label = QLabel(f"{label_text}:")
            params_layout.addWidget(label, row, 0)
            
            # 值显示
            value_label = QLabel(str(default_val))
            value_label.setMinimumWidth(50)
            params_layout.addWidget(value_label, row, 1)
            
            # 滑条
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setSingleStep(step)
            slider.setValue(default_val)
            slider.valueChanged.connect(lambda v, pl=value_label, pn=param_name: 
                                       self.on_slider_changed(pn, v, pl))
            params_layout.addWidget(slider, row, 2)
            
            self.sliders[param_name] = (slider, value_label)
            row += 1
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        layout.addStretch()
        return panel
    
    def on_slider_changed(self, param_name, value, value_label):
        # 更新显示的值
        display_value = value
        if param_name == "circularity_min":
            display_value = f"{value/100:.2f}"
            setattr(self.params, param_name, value / 100.0)
        else:
            # 确保核大小为奇数
            if param_name in ["denoise_ksize", "tophat_ksize", "adapt_blocksize", 
                             "open_ksize", "dilate_ksize"]:
                if value % 2 == 0:
                    value = value + 1
                display_value = value
            
            setattr(self.params, param_name, float(value) if param_name in ["area_min", "area_max"] else value)
        
        value_label.setText(str(display_value))
        
        # 延迟更新（300ms后执行）
        if self.gray_image is not None:
            self.update_timer.stop()
            self.update_timer.start(300)
    
    def load_image(self):
        start_dir = self.current_dir if self.current_dir else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", start_dir, "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            try:
                self.gray_image = load_gray(file_path)
                self.filename_label.setText(os.path.basename(file_path))
                # 清空浏览列表，单张模式
                self.image_list = [file_path]
                self.current_index = 0
                self.update_navigation_controls()
                self.process_and_display()
            except Exception as e:
                print(f"Error loading image: {str(e)}")
    
    def load_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.current_dir if self.current_dir else ""
        )
        if dir_path:
            self.current_dir = dir_path
            # 查找所有支持的图片文件
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff',
                         '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIF', '*.TIFF']
            image_files = []
            for ext in extensions:
                image_files.extend(Path(dir_path).glob(ext))
            
            # 按文件名排序
            self.image_list = sorted([str(f) for f in image_files])
            
            if self.image_list:
                self.current_index = 0
                self.load_image_at_index(0)
                self.update_navigation_controls()
            else:
                QMessageBox.warning(self, "No Images", "No image files found in the selected directory.")
    
    def load_image_at_index(self, index):
        if 0 <= index < len(self.image_list):
            try:
                self.current_index = index
                file_path = self.image_list[index]
                self.gray_image = load_gray(file_path)
                self.filename_label.setText(os.path.basename(file_path))
                self.process_and_display()
            except Exception as e:
                print(f"Error loading image: {str(e)}")
    
    def prev_image(self):
        if self.current_index > 0:
            self.load_image_at_index(self.current_index - 1)
            self.index_spinbox.setValue(self.current_index + 1)
    
    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.load_image_at_index(self.current_index + 1)
            self.index_spinbox.setValue(self.current_index + 1)
    
    def jump_to_image(self, value):
        target_index = value - 1
        if target_index != self.current_index and 0 <= target_index < len(self.image_list):
            self.load_image_at_index(target_index)
    
    def update_navigation_controls(self):
        has_images = len(self.image_list) > 0
        multiple_images = len(self.image_list) > 1
        
        self.prev_btn.setEnabled(multiple_images and self.current_index > 0)
        self.next_btn.setEnabled(multiple_images and self.current_index < len(self.image_list) - 1)
        self.index_spinbox.setEnabled(multiple_images)
        
        if has_images:
            self.index_spinbox.setMaximum(len(self.image_list))
            self.index_spinbox.setValue(self.current_index + 1)
            self.total_label.setText(f"/ {len(self.image_list)}")
        else:
            self.index_spinbox.setMaximum(1)
            self.total_label.setText("/ 0")
    
    def save_parameters(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                params_dict = asdict(self.params)
                with open(file_path, 'w') as f:
                    json.dump(params_dict, f, indent=4)
                QMessageBox.information(self, "Success", "Parameters saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save parameters: {str(e)}")
    
    def load_parameters(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params_dict = json.load(f)
                
                # 更新参数对象
                for key, value in params_dict.items():
                    if hasattr(self.params, key):
                        setattr(self.params, key, value)
                
                # 更新界面上的滑条
                self.update_sliders_from_params()
                
                # 重新处理图像
                if self.gray_image is not None:
                    self.process_and_display()
                
                QMessageBox.information(self, "Success", "Parameters loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {str(e)}")
    
    def update_sliders_from_params(self):
        """根据参数对象更新滑条位置"""
        slider_map = {
            "denoise_ksize": self.params.denoise_ksize,
            "tophat_ksize": self.params.tophat_ksize,
            "adapt_blocksize": self.params.adapt_blocksize,
            "adapt_C": self.params.adapt_C,
            "open_ksize": self.params.open_ksize,
            "dilate_ksize": self.params.dilate_ksize,
            "area_min": int(self.params.area_min),
            "area_max": int(self.params.area_max),
            "circularity_min": int(self.params.circularity_min * 100),
        }
        
        for param_name, value in slider_map.items():
            if param_name in self.sliders:
                slider, value_label = self.sliders[param_name]
                slider.setValue(value)
                if param_name == "circularity_min":
                    value_label.setText(f"{value/100:.2f}")
                else:
                    value_label.setText(str(value))
    
    def process_and_display(self):
        if self.gray_image is None:
            return
        
        try:
            # 处理图像
            res = extract_glints(self.gray_image, self.params)
            
            # 创建拼接图像
            gray_bgr = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
            tophat_bgr = cv2.cvtColor(res["tophat"], cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2BGR)
            
            def add_title(img, title):
                img_copy = img.copy()
                cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
                return img_copy
            
            num_glints = len(res['centers'])
            gray_titled = add_title(gray_bgr, "Original")
            tophat_titled = add_title(tophat_bgr, "Top-hat")
            mask_titled = add_title(mask_bgr, "Mask")
            overlay_titled = add_title(res["overlay"], f"Result ({num_glints} glints)")
            
            # 拼接成2x2网格
            top_row = np.hstack([gray_titled, tophat_titled])
            bottom_row = np.hstack([mask_titled, overlay_titled])
            combined = np.vstack([top_row, bottom_row])
            
            # 转换为Qt格式并显示
            self.display_image(combined)
            
        except Exception as e:
            print(f"Error processing: {str(e)}")
    
    def display_image(self, img):
        # BGR to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放以适应窗口
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

# -------- Main --------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式，更现代
    window = GlintDetectorGUI()
    window.show()
    sys.exit(app.exec_())
