import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Basic Losses
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        eps = 1e-6
        p_t = pred_sig * target + (1 - pred_sig) * (1 - target)
        focal = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + eps)
        return focal.mean() if self.reduction == 'mean' else focal.sum()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum(dim=(2,3))
        den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + self.eps
        dice = 1 - (num / den).mean()
        return dice
    
import itertools    
class ChannelDivergenceLoss(nn.Module):
    """
    惩罚不同通道的热力图过于相似
    """
    def __init__(self, mode="overlap", weight=0.05):
        super().__init__()
        self.mode = mode
        self.weight = weight

    def forward(self, pred):
        """
        pred: (B, C, H, W)
        """
        B, C, H, W = pred.shape
        pred_sig = torch.sigmoid(pred)
        loss = 0.0
        count = 0

        if self.mode == "overlap":
            # 简单重叠惩罚：平均两通道乘积
            for i, j in itertools.combinations(range(C), 2):
                loss += (pred_sig[:, i] * pred_sig[:, j]).mean()
                count += 1
        elif self.mode == "cosine":
            # 余弦相似度惩罚
            v = pred_sig.flatten(2)  # (B, C, H*W)
            v = F.normalize(v, dim=2)
            for i, j in itertools.combinations(range(C), 2):
                cos_sim = (v[:, i] * v[:, j]).sum(dim=1).mean()
                loss += cos_sim
                count += 1
        elif self.mode == "kl":
            # KL divergence 形式（假设sigmoid后已正且归一化）
            p = pred_sig / (pred_sig.sum(dim=(2,3), keepdim=True) + 1e-8)
            for i, j in itertools.combinations(range(C), 2):
                kl = (p[:, i] * (torch.log(p[:, i] + 1e-8) - torch.log(p[:, j] + 1e-8))).mean()
                loss += torch.abs(kl)
                count += 1

        if count > 0:
            loss = loss / count
        return self.weight * loss

def aggregate_maps(tensor, mode="max"):
    """
    tensor: (B,C,H,W) logits 或 target (已是热力图的连续值)
    返回: (B,1,H,W)
    """
    if mode == "max":
        if tensor.ndim == 4:
            return tensor.max(dim=1, keepdim=True).values
        else:
            return tensor
    elif mode == "sum":
        agg = tensor.sum(dim=1, keepdim=True)
        return agg.clamp(0, 1)
    else:
        raise ValueError(f"Unknown agg_mode: {mode}")

# -------------------------------
# Hybrid Loss Wrapper
# -------------------------------
class HybridLoss(nn.Module):
    """
    total = (lam_focal*Focal + lam_bce*BCE + lam_dice*Dice)   # 多通道逐通道
            + div_weight*Divergence
            + lam_agg*( wF*Focal_agg + wB*BCE_agg + wD*Dice_agg )  # 聚合通道损失
    """
    def __init__(self,
                 lam_focal=0.5, lam_bce=0.3, lam_dice=0.2,
                 alpha=1.0, gamma=2.0,
                 div_weight=0.05, div_mode="overlap",
                 lam_agg=0.2, agg_mode="max", agg_weights=(0.5, 0.3, 0.2)):
        super().__init__()
        self.lam_focal = lam_focal
        self.lam_bce   = lam_bce
        self.lam_dice  = lam_dice
        self.div_weight = div_weight

        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce   = nn.BCEWithLogitsLoss()
        self.dice  = DiceLoss()
        self.div   = ChannelDivergenceLoss(weight=div_weight, mode=div_mode)

        # 聚合项
        self.lam_agg    = lam_agg
        self.agg_mode   = agg_mode
        self.agg_wF, self.agg_wB, self.agg_wD = agg_weights

    def forward(self, pred, target):
        # ---- 基础逐通道项 ----
        out = {}
        out["focal"] = self.lam_focal * self.focal(pred, target) if self.lam_focal > 0 else torch.tensor(0., device=pred.device)
        out["bce"]   = self.lam_bce   * self.bce(pred, target)   if self.lam_bce   > 0 else torch.tensor(0., device=pred.device)
        out["dice"]  = self.lam_dice  * self.dice(pred, target)  if self.lam_dice  > 0 else torch.tensor(0., device=pred.device)
        out["div"]   = self.div(pred) if self.div_weight > 0 else torch.tensor(0., device=pred.device)

        # ---- 聚合类无关项 ----
        if self.lam_agg > 0:
            # pred 聚合需要在 logits 域做 BCE / Focal / Dice：保持接口一致，用 1 通道 logits
            # 这里把聚合后的目标也做相同聚合（target 为 [0,1] 的热力图）
            pred_agg = aggregate_maps(pred, self.agg_mode)           # (B,1,H,W) logits
            with torch.no_grad():                                     # 目标聚合（数值已在[0,1]，无需sigmoid）
                target_agg = aggregate_maps(target, self.agg_mode)    # (B,1,H,W)
            agg_focal = self.agg_wF * self.focal(pred_agg, target_agg) if self.agg_wF > 0 else torch.tensor(0., device=pred.device)
            agg_bce   = self.agg_wB * self.bce(pred_agg, target_agg)   if self.agg_wB > 0 else torch.tensor(0., device=pred.device)
            agg_dice  = self.agg_wD * self.dice(pred_agg, target_agg)  if self.agg_wD > 0 else torch.tensor(0., device=pred.device)
            out["agg_focal"], out["agg_bce"], out["agg_dice"] = agg_focal, agg_bce, agg_dice
            out["agg_total"] = self.lam_agg * (agg_focal + agg_bce + agg_dice)
        else:
            out["agg_focal"] = torch.tensor(0., device=pred.device)
            out["agg_bce"]   = torch.tensor(0., device=pred.device)
            out["agg_dice"]  = torch.tensor(0., device=pred.device)
            out["agg_total"] = torch.tensor(0., device=pred.device)

        out["total"] = out["focal"] + out["bce"] + out["dice"] + out["div"] + out["agg_total"]
        return out


# -------------------------------
# Factory function for training script
# -------------------------------
def build_loss(name, **kw):
    name = name.lower()
    if name == "focal":
        return FocalLoss(alpha=kw.get("alpha",1.0), gamma=kw.get("gamma",2.0))
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "dice":
        return DiceLoss()
    elif name == "hybrid":
        return HybridLoss(
            lam_focal=kw.get("lam_focal",0.5),
            lam_bce=kw.get("lam_bce",0.3),
            lam_dice=kw.get("lam_dice",0.2),
            alpha=kw.get("alpha",1.0),
            gamma=kw.get("gamma",2.0),
            div_weight=kw.get("div_weight",0.05),
            div_mode=kw.get("div_mode","overlap"),
            lam_agg=kw.get("lam_agg",0.2),
            agg_mode=kw.get("agg_mode","max"),
            agg_weights=kw.get("agg_weights",(0.5,0.3,0.2))
        )
    else:
        raise ValueError(f"Unknown loss name: {name}")
