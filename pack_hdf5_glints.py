# pack_hdf5_glints.py
import os, json, math, glob, argparse
import numpy as np
import cv2, h5py
from typing import Tuple, Optional

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---------- 生成热力图工具 ----------
def parse_idx(label: str, num_classes=8, prefix='g'):
    s = (label or "").strip().lower()
    if s.startswith(prefix) and s[len(prefix):].isdigit():
        i = int(s[len(prefix):]) - 1
        return i if 0 <= i < num_classes else None
    return None

def circle_from_shape(shape):
    if shape.get("shape_type") != "circle":
        return None
    pts = shape.get("points", [])
    if len(pts) != 2:
        return None
    (cx, cy), (px, py) = pts
    r = math.hypot(px - cx, py - cy)
    return float(cx), float(cy), float(r)

def draw_gaussian(Hc, cx, cy, r, sigma_scale=0.6, sigma_min=1.5, truncate=3.0):
    h, w = Hc.shape
    sigma = max(float(sigma_min), float(r) * float(sigma_scale))
    ks = int(truncate * sigma)
    x0, y0 = max(0, int(cx) - ks), max(0, int(cy) - ks)
    x1, y1 = min(w - 1, int(cx) + ks), min(h - 1, int(cy) + ks)
    if x1 <= x0 or y1 <= y0:
        return Hc
    xs = np.arange(x0, x1 + 1); ys = np.arange(y0, y1 + 1)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2.0 * sigma**2)).astype(np.float32)
    Hc[y0:y1+1, x0:x1+1] = np.maximum(Hc[y0:y1+1, x0:x1+1], g)
    return Hc

def resize_stack(stack: np.ndarray, out_hw: Tuple[int,int]):
    C, H, W = stack.shape
    out_h, out_w = out_hw
    out = np.zeros((C, out_h, out_w), np.float32)
    for c in range(C):
        out[c] = cv2.resize(stack[c], (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 1)

def resolve_image_path(json_path, ann_image_path, in_dir):
    # 优先 imagePath；否则按 stem 在 in_dir 搜匹配图像
    if ann_image_path:
        p = ann_image_path
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(json_path), p)
        if os.path.exists(p):
            return p
    stem = os.path.splitext(os.path.basename(json_path))[0]
    for root, _, files in os.walk(in_dir):
        for f in files:
            if os.path.splitext(f)[0] == stem and f.lower().endswith(IMG_EXTS):
                return os.path.join(root, f)
    return None

def build_sample(json_path, in_dir, num_classes, sigma_scale, sigma_min,
                 out_img_hw: Optional[Tuple[int,int]],
                 out_hm_hw: Optional[Tuple[int,int]],
                 prefix='g'):
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    img_path = resolve_image_path(json_path, ann.get("imagePath"), in_dir)
    if img_path is None:
        raise FileNotFoundError(f"Image not found for {json_path}")
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Fail to read image: {img_path}")

    Hh, Ww = gray.shape[:2]
    H = np.zeros((num_classes, Hh, Ww), np.float32)
    present = np.zeros((num_classes,), np.uint8)

    for sh in ann.get("shapes", []):
        idx = parse_idx(sh.get("label", ""), num_classes, prefix)
        if idx is None: 
            continue
        circ = circle_from_shape(sh)
        if circ is None:
            continue
        cx, cy, r = circ
        present[idx] = 1
        H[idx] = draw_gaussian(H[idx], cx, cy, r, sigma_scale, sigma_min)

    # 图像与热力图尺寸（可不同：img→网络输入，hm→网络输出分辨率）
    if out_img_hw is not None:
        img_out = cv2.resize(gray, (out_img_hw[1], out_img_hw[0]), interpolation=cv2.INTER_AREA)
    else:
        img_out = gray

    if out_hm_hw is not None:
        H = resize_stack(H, out_hm_hw)
    # 量化到 float16 节省空间
    H = H.astype(np.float16)
    img_out = img_out.astype(np.uint8)

    stem = os.path.splitext(os.path.basename(json_path))[0]
    return stem, img_out, H, present

# ---------- HDF5 写入 ----------
def create_h5(out_path, N_max, img_hw, hm_chw, compress="gzip", chunks=True):
    H, W = img_hw
    C, h, w = hm_chw
    f = h5py.File(out_path, "w")
    d_images = f.create_dataset(
        "images", shape=(0, H, W), maxshape=(N_max, H, W),
        dtype="uint8", compression=compress, chunks=(min(64, N_max), H, W) if chunks else None
    )
    d_heatmaps = f.create_dataset(
        "heatmaps", shape=(0, C, h, w), maxshape=(N_max, C, h, w),
        dtype="float16", compression=compress, chunks=(min(64, N_max), C, h, w) if chunks else None
    )
    d_present = f.create_dataset(
        "present", shape=(0, C), maxshape=(N_max, C),
        dtype="uint8", compression=compress, chunks=(min(256, N_max), C) if chunks else None
    )
    dt = h5py.string_dtype(encoding="utf-8")
    d_stems = f.create_dataset(
        "stems", shape=(0,), maxshape=(N_max,), dtype=dt, compression=compress
    )
    return f, d_images, d_heatmaps, d_present, d_stems

def append_row(dset, data):
    n = dset.shape[0]
    dset.resize(n+1, axis=0)
    dset[n] = data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="输入目录（图像与 Labelme JSON 混放，可递归）")
    ap.add_argument("--out_h5", required=True, help="输出 HDF5 文件路径，如 dataset.h5")
    ap.add_argument("--num_classes", type=int, default=8)
    ap.add_argument("--sigma_scale", type=float, default=1.2)
    ap.add_argument("--sigma_min", type=float, default=1.5)
    ap.add_argument("--img_h", type=int, default=512, help="存储/训练的图像高度")
    ap.add_argument("--img_w", type=int, default=512, help="存储/训练的图像宽度")
    ap.add_argument("--hm_h", type=int, default=512, help="热力图高度（可与图像不同，如下采样）")
    ap.add_argument("--hm_w", type=int, default=512, help="热力图宽度")
    ap.add_argument("--prefix", type=str, default="g")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--compression", type=str, default="gzip", choices=["gzip","lzf","None"])
    args = ap.parse_args()

    json_pattern = "**/*.json" if args.recursive else "*.json"
    json_files = glob.glob(os.path.join(args.in_dir, json_pattern), recursive=args.recursive)
    json_files = [p for p in json_files if os.path.isfile(p)]
    json_files.sort()
    if not json_files:
        print("No JSON found."); return

    img_hw = (args.img_h, args.img_w)
    hm_chw = (args.num_classes, args.hm_h, args.hm_w)
    N_max = len(json_files)

    compress = None if args.compression == "None" else args.compression
    f, d_images, d_heatmaps, d_present, d_stems = create_h5(args.out_h5, N_max, img_hw, hm_chw, compress)

    # 写入全局属性（元信息）
    f.attrs["num_classes"] = args.num_classes
    f.attrs["sigma_scale"] = args.sigma_scale
    f.attrs["sigma_min"] = args.sigma_min
    f.attrs["img_h"] = args.img_h; f.attrs["img_w"] = args.img_w
    f.attrs["hm_h"] = args.hm_h;   f.attrs["hm_w"] = args.hm_w
    f.attrs["prefix"] = args.prefix

    ok = 0
    for i, jp in enumerate(json_files, 1):
        try:
            stem, img, H, present = build_sample(
                jp, args.in_dir, args.num_classes, args.sigma_scale, args.sigma_min,
                out_img_hw=img_hw, out_hm_hw=(args.hm_h, args.hm_w), prefix=args.prefix
            )
            append_row(d_images, img)
            append_row(d_heatmaps, H)
            append_row(d_present, present)
            append_row(d_stems, stem)
            ok += 1
            if i % 50 == 0: print(f"[{i}/{len(json_files)}] packed")
        except Exception as e:
            print(f"[WARN] {jp}: {e}")

    print(f"Done. {ok}/{len(json_files)} samples packed into {args.out_h5}")
    f.close()

if __name__ == "__main__":
    main()
