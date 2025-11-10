# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ------------------------------
# #   Basic U-Net building blocks
# # ------------------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x): return self.seq(x)

# class UNet(nn.Module):
#     def __init__(self, in_ch=1, out_ch=8, base_ch=16):
#         super().__init__()
#         self.enc1 = ConvBlock(in_ch, base_ch)
#         self.enc2 = ConvBlock(base_ch, base_ch * 2)
#         self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
#         self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
#         self.pool = nn.MaxPool2d(2)

#         self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

#         self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
#         self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)
#         self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
#         self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
#         self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
#         self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
#         self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
#         self.dec1 = ConvBlock(base_ch * 2, base_ch)

#         self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
#         b  = self.bottleneck(self.pool(e4))

#         d4 = self.up4(b)
#         d4 = self.dec4(torch.cat([d4, e4], 1))
#         d3 = self.up3(d4)
#         d3 = self.dec3(torch.cat([d3, e3], 1))
#         d2 = self.up2(d3)
#         d2 = self.dec2(torch.cat([d2, e2], 1))
#         d1 = self.up1(d2)
#         d1 = self.dec1(torch.cat([d1, e1], 1))

#         out = self.out_conv(d2)
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
#   Basic U-Net building blocks
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=8, base_ch=16):
        super().__init__()
        # 编码器 - 4次下采样: 512→256→128→64→32
        self.enc1 = ConvBlock(in_ch, base_ch)                    # 输出: (16, 512, 512)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)              # 输出: (32, 256, 256)  
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)          # 输出: (64, 128, 128) ← 目标尺寸层
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)          # 输出: (128, 64, 64)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)   # 输出: (256, 32, 32)

        # 解码器 - 只上采样2次: 32→64→128 (在128×128停止)
        self.up3 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)  # 输出: (128, 64, 64)
        self.dec3 = ConvBlock(base_ch * 16, base_ch * 8)         # 输入: 256→输出: 128
        
        self.up2 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)   # 输出: (64, 128, 128)
        self.dec2 = ConvBlock(base_ch * 8, base_ch * 4)          # 输入: 128→输出: 64
        
        # 输出层 - 在128×128尺寸输出
        self.out_conv = nn.Conv2d(base_ch * 4, out_ch, 1)        # 输入: 64→输出: 8

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)                    # (16, 512, 512)
        e2 = self.enc2(self.pool(e1))        # (32, 256, 256)
        e3 = self.enc3(self.pool(e2))        # (64, 128, 128) ← 保存用于跳跃连接
        e4 = self.enc4(self.pool(e3))        # (128, 64, 64)  ← 保存用于跳跃连接
        b  = self.bottleneck(self.pool(e4))  # (256, 32, 32)

        # 解码路径 - 只解码到128×128
        d3 = self.up3(b)                     # (128, 64, 64)
        d3 = self.dec3(torch.cat([d3, e4], 1))  # 输入: 128+128=256 → 输出: 128
        
        d2 = self.up2(d3)                    # (64, 128, 128) ← 目标输出尺寸！
        d2 = self.dec2(torch.cat([d2, e3], 1))  # 输入: 64+64=128 → 输出: 64

        # 最终输出 - 在128×128尺寸
        out = self.out_conv(d2)              # (8, 128, 128)
        return out