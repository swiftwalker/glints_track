import argparse, os, cv2, torch, numpy as np
from unet_glint import UNet
from scipy.ndimage import maximum_filter


def extract_single_peak(hm, threshold=0.3, kernel=5):
    """
    从单通道热力图提取一个最高点（若低于阈值则忽略）
    hm: (H,W) numpy array in [0,1]
    return: (x, y, score) or None
    """
    # 可选: 局部平滑去除噪声
    hm_smooth = cv2.GaussianBlur(hm, (3,3), 0)
    # 找局部最大值
    max_f = maximum_filter(hm_smooth, size=kernel, mode='constant')
    peaks = (hm_smooth == max_f)
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return None
    # 找最大值
    idx = np.argmax(hm_smooth[ys, xs])
    score = float(hm_smooth[ys[idx], xs[idx]])
    if score < threshold:
        return None
    return (int(xs[idx]), int(ys[idx]), score)


def infer_and_draw(model, img_path, out_path=None, device="cuda", threshold=0.3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    H, W = img.shape
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0].cpu().numpy()  # (C,H,W)

    # 创建3x3网格显示：原图 + 8个通道热力图
    def add_title(img_bgr, title):
        img_copy = img_bgr.copy()
        cv2.putText(img_copy, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img_copy
    
    # 原图（转BGR）
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_titled = add_title(img_bgr, "Original")
    
    # 准备热力图（使用JET colormap）
    heatmaps = []
    for c in range(pred.shape[0]):
        hm = pred[c]
        # 转换为0-255范围
        hm_uint8 = (hm * 255).astype(np.uint8)
        # 应用JET colormap
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        # 添加标题
        hm_titled = add_title(hm_colored, f"Channel {c+1}")
        heatmaps.append(hm_titled)
    
    # 拼接成3x3网格（原图在左上角，后面8个通道）
    images = [img_titled] + heatmaps
    
    # 确保有9张图（原图+8通道）
    rows = []
    for i in range(0, 9, 3):
        row = np.hstack(images[i:i+3])
        rows.append(row)
    
    combined = np.vstack(rows)
    
    # 根据屏幕分辨率调整窗口大小
    import tkinter as tk
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # 使用屏幕尺寸的80%作为最大显示尺寸
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        combined_h, combined_w = combined.shape[:2]
        
        # 计算缩放比例
        scale_w = max_width / combined_w if combined_w > max_width else 1.0
        scale_h = max_height / combined_h if combined_h > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w = int(combined_w * scale)
            new_h = int(combined_h * scale)
            combined_resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            combined_resized = combined
    except:
        # 如果获取屏幕分辨率失败，使用原图
        combined_resized = combined
    
    if out_path:
        cv2.imwrite(out_path, combined)
        print("Saved:", out_path)

    cv2.namedWindow("Original + Heatmaps", cv2.WINDOW_NORMAL)
    cv2.imshow("Original + Heatmaps", combined_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = UNet(in_ch=1, out_ch=8)
    state = torch.load(args.model, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device)

    infer_and_draw(model, args.img, args.out, args.device, args.threshold)

if __name__ == "__main__":
    main()
