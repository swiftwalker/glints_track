# visualize_h5_glints.py
# 用法示例：
#   python visualize_h5_glints.py --h5 path\to\dataset.h5
#   python visualize_h5_glints.py --h5 path\to\dataset.h5 --idx 10
# 键位：
#   n / p    下一张 / 上一张
#   r        随机一张
#   m        显示/隐藏「类无关(max)热图」
#   1..8     显示/隐藏对应通道热图（叠加）
#   s        保存当前叠加图到同目录
#   q / ESC  退出
import argparse, os, random, time
import numpy as np
import h5py, cv2

def load_sample(f, idx):
    img  = f["images"][idx]           # H×W uint8
    hm   = f["heatmaps"][idx]         # C×h×w float16
    pres = f["present"][idx]          # C
    raw_stem = f["stems"][idx]
    stem = raw_stem.decode("utf-8") if isinstance(raw_stem, (bytes, bytearray)) else str(raw_stem)
    return img, hm.astype(np.float32), pres.astype(np.uint8), stem

def to_overlay(img_gray, hm_stack, show_max=True, show_ch=None, alpha=0.5):
    """
    img_gray: H×W
    hm_stack: C×h×w
    show_ch:  形如 {0,2,5} 的集合，表示要叠加哪些通道
    """
    H, W = img_gray.shape[:2]
    C, h, w = hm_stack.shape
    # 统一出一个热图到图像分辨率
    hm_sum = np.zeros((H, W), np.float32)

    # 类无关：max over channel
    if show_max:
        hm_max = np.max(hm_stack, axis=0)
        hm_max = cv2.resize(hm_max, (W, H), interpolation=cv2.INTER_AREA)
        hm_sum = np.maximum(hm_sum, hm_max)

    # 指定通道叠加（取最大，方便观察）
    if show_ch:
        for c in sorted(show_ch):
            if 0 <= c < C:
                hm_c = cv2.resize(hm_stack[c], (W, H), interpolation=cv2.INTER_AREA)
                hm_sum = np.maximum(hm_sum, hm_c)

    # 归一化到 [0,255] 伪彩
    hm_norm = hm_sum.copy()
    if hm_norm.max() > 0:
        hm_norm = hm_norm / hm_norm.max()
    hm_u8 = (hm_norm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_rgb, 1.0, hm_color, alpha, 0)
    return overlay, hm_color

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="打包生成的 HDF5 数据集路径")
    ap.add_argument("--idx", type=int, default=-1, help="起始索引（默认随机）")
    ap.add_argument("--alpha", type=float, default=0.5, help="热力图叠加透明度")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    N = f["images"].shape[0]
    C = f["heatmaps"].shape[1]

    idx = args.idx if 0 <= args.idx < N else random.randrange(N)
    show_max = True
    show_ch = set()  # 叠加的指定通道（0..C-1）

    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("heatmap_only", cv2.WINDOW_NORMAL)

    while True:
        img, hm, pres, stem = load_sample(f, idx)
        overlay, hm_color = to_overlay(img, hm, show_max=show_max, show_ch=show_ch, alpha=args.alpha)

        # 额外叠加一些文字信息
        info = f"idx={idx}/{N-1}  stem={stem}  present={list(pres)}  show_max={int(show_max)}  ch={sorted(show_ch)}"
        vis = overlay.copy()
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        cv2.imshow("overlay", vis)
        cv2.imshow("heatmap_only", hm_color)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('n'):
            idx = (idx + 1) % N
        elif key == ord('p'):
            idx = (idx - 1 + N) % N
        elif key == ord('r'):
            idx = random.randrange(N)
        elif key == ord('m'):
            show_max = not show_max
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = os.path.join(os.path.dirname(args.h5), f"viz_{stem}_{idx}_{ts}.png")
            cv2.imwrite(out_path, vis)
            print("saved:", out_path)
        else:
            # 数字键 1..8 切换对应通道（0..7）
            if ord('1') <= key <= ord('9'):
                c = key - ord('1')  # 0-based
                if c < C:
                    if c in show_ch:
                        show_ch.remove(c)
                    else:
                        show_ch.add(c)

    f.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
