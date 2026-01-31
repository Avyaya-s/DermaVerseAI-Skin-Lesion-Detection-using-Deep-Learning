import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Basic Conv ----------
class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ---------- RSU-4F (Official, stable) ----------
class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.in_conv = REBNCONV(in_ch, out_ch)

        self.conv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.conv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.conv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.conv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.conv_d3 = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.conv_d2 = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.conv_d1 = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.in_conv(x)

        hx1 = self.conv1(hxin)
        hx2 = self.conv2(hx1)
        hx3 = self.conv3(hx2)
        hx4 = self.conv4(hx3)

        hx3d = self.conv_d3(torch.cat((hx4, hx3), 1))
        hx2d = self.conv_d2(torch.cat((hx3d, hx2), 1))
        hx1d = self.conv_d1(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin

# ---------- U²-NetP (Recommended) ----------
class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.stage1 = RSU4F(in_ch, 16, 64)
        self.stage2 = RSU4F(64, 16, 64)
        self.stage3 = RSU4F(64, 16, 64)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.out_conv = nn.Conv2d(3 * out_ch, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool(hx2)

        hx3 = self.stage3(hx)

        d1 = self.side1(hx1)
        d2 = F.interpolate(self.side2(hx2), d1.shape[2:])
        d3 = F.interpolate(self.side3(hx3), d1.shape[2:])

        d = self.out_conv(torch.cat((d1, d2, d3), 1))
        return torch.sigmoid(d)
