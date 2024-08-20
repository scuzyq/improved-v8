import torch
from torch import nn
from torch.nn import init
 
from ultralytics.nn.modules.block import C2f
 
 
 
### by CSDN    AI  Little monster    https://blog.csdn.net/m0_63774211?type=lately
 
""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
 
 
### by CSDN    AI  Little monster    https://blog.csdn.net/m0_63774211?type=lately
 
""" Adaptive Feature Fusion """
class AdaptiveFeatureFusionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(AdaptiveFeatureFusionBlock, self).__init__()
 
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
 
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
 
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
 
        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
 
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
 
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)
 
        x4 = x2 + x3
        x4 = self.relu(x4)
 
        return x4
 
class WeightedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
 
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
 
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        input_conv = self.input_conv(x)
        return input_conv * self.weight(input_conv)
 
 
class DecodingBlock(nn.Module):
    def __init__(self, low_channels, in_channels, out_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, low_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True),
        )
 
        self.u = nn.Upsample(scale_factor=2, mode='bilinear')
        self.output_conv = AdaptiveFeatureFusionBlock(low_channels, out_channels)
 
    def forward(self, low_x, x):
        return self.output_conv(low_x + self.u(self.input_conv(x)))
 
''' Efficient Fusion Attention Module '''
class EFAttention(nn.Module):
    def __init__(self, in_channels, kernel_size = 3):
        super().__init__()
 
        # x_c
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
 
        # x_s
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
 
    def forward(self,x):
        # x_c
        y1 = self.gap(x)
        y1 = y1.squeeze(-1).permute(0, 2, 1)
        y1 = self.conv(y1)
        y1 = self.sigmoid(y1)
        y1 = y1.permute(0, 2, 1).unsqueeze(-1)
        x_c =  x * y1.expand_as(x)
 
        # x_s
        q = self.Conv1x1(x)
        q = self.norm(q)
        x_s = x * q
        return x_c + x_s
 
class C2f_EFAttention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EFAttention(self.c) for _ in range(n))
 
### by CSDN    AI  Little monster    https://blog.csdn.net/m0_63774211?type=lately
