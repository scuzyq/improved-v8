import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
 
from ultralytics.nn.modules import Conv,C2f
 
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
 
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
 
    x = torch.transpose(x, 1, 2).contiguous()
 
    # flatten
    x = x.view(batchsize, -1, height, width)
 
    return x
 
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
class CACS_Bottleneck(nn.Module):
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.add = shortcut and c1 == c2
        c1 >>= 1
        c2 >>= 1
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
 
        self.conv = nn.Conv2d(c_, c2, k[1], 1, autopad(3, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
 
        self.ca = CoordAttlayer(c2, c2, reduction=32)
 
        self.act = nn.LeakyReLU(0.1, inplace=True)
 
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.cv1(x2)
        x2 = self.conv(x2)
        x2 = self.bn(x2)
 
        x2 = self.ca(x2)
        out = torch.cat((x2, x1), dim=1)
        out = self.act(out)
 
        return channel_shuffle(out, 2)
 
class CAM_Bottleneck(nn.Module):
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.add = shortcut and c1 == c2
 
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
 
        self.conv = nn.Conv2d(c_, c2, k[1], 1, autopad(3, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
 
        self.ca = CoordAttlayer(c2, c2, reduction=32)
 
        self.act = nn.LeakyReLU(0.1, inplace=True)
 
    def forward(self, x):
        x1=x
        x1 = self.cv1(x1)
        x1 = self.conv(x1)
        x1= self.bn(x1)
 
        x1 = self.ca(x1)
        out = x1+x
        out = self.act(out)
 
        return out
 
class CSO_Bottleneck(nn.Module):
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.add = shortcut and c1 == c2
        c1 >>= 1
        c2 >>= 1
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
 
        self.conv = nn.Conv2d(c_, c2, k[1], 1, autopad(3, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
 
        self.act = nn.LeakyReLU(0.1, inplace=True)
 
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.cv1(x2)
        x2 = self.conv(x2)
        x2 = self.bn(x2)
 
        out = torch.cat((x2, x1), dim=1)
        out = self.act(out)
 
        return channel_shuffle(out, 2)
 
 
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAttlayer(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttlayer, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
 
 
class CACS_C2f(nn.Module):
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CACS_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
