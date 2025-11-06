import torch, h5py, random, os, argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from unet_glint import UNet
from losses import build_loss
from tqdm import tqdm

# ------------------------------
# Dataset
# ------------------------------
class GlintH5(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self.imgs = self.f["images"]
        self.hms  = self.f["heatmaps"]
    def __len__(self): return self.imgs.shape[0]
    def __getitem__(self, i):
        img = torch.from_numpy(self.imgs[i][...]).float().unsqueeze(0) / 255.0
        hm  = torch.from_numpy(self.hms[i][...].astype("float32"))
        return img, hm

# ------------------------------
# Training Loop
# ------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_meter = {"total":0, "focal":0, "bce":0, "dice":0, "div":0, 
                  "agg_total":0, "agg_focal":0, "agg_bce":0, "agg_dice":0}
    n_batches = len(loader)

    for imgs, targets in tqdm(loader, desc="Train", ncols=90):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        losses = loss_fn(out, targets)      # ← returns dict
        total_loss = losses["total"]

        total_loss.backward()
        optimizer.step()

        for k in loss_meter.keys():
            if k in losses:
                loss_meter[k] += losses[k].item()

    # 求平均
    for k in loss_meter.keys():
        loss_meter[k] /= n_batches
    return loss_meter

def validate(model, loader, loss_fn, device):
    model.eval()
    loss_meter = {"total":0, "focal":0, "bce":0, "dice":0, "div":0,
                  "agg_total":0, "agg_focal":0, "agg_bce":0, "agg_dice":0}
    n_batches = len(loader)

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", ncols=90):
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            losses = loss_fn(out, targets)
            for k in loss_meter.keys():
                if k in losses:
                    loss_meter[k] += losses[k].item()

    for k in loss_meter.keys():
        loss_meter[k] /= n_batches
    return loss_meter

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default="checkpoints")
    ap.add_argument("--model_path", default="best.pt", help="模型保存路径")
    ap.add_argument("--val_split", type=float, default=0.1)
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

    os.makedirs(args.save, exist_ok=True)

    dataset = GlintH5(args.h5)
    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    best_loss = float("inf")
    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = validate(model, val_loader, loss_fn, device)

        print(f"\nEpoch {epoch:03d}:")
        print(
        f"  Train: total={tr['total']:.4f} focal={tr['focal']:.4f} bce={tr['bce']:.4f} "
        f"dice={tr['dice']:.4f} div={tr['div']:.4f} | "
        f"agg_total={tr['agg_total']:.4f} (aggF={tr['agg_focal']:.4f} aggB={tr['agg_bce']:.4f} aggD={tr['agg_dice']:.4f})"
        )
        print(
        f"  Val:   total={va['total']:.4f} focal={va['focal']:.4f} bce={va['bce']:.4f} "
        f"dice={va['dice']:.4f} div={va['div']:.4f} | "
        f"agg_total={va['agg_total']:.4f} (aggF={va['agg_focal']:.4f} aggB={va['agg_bce']:.4f} aggD={va['agg_dice']:.4f})"
        )

        # 保存最优模型
        if va["total"] < best_loss:
            best_loss = va["total"]
            torch.save(model.state_dict(), args.model_path)
            print(f"  ✅ Saved best model to {args.model_path}")

if __name__ == "__main__":
    main()
