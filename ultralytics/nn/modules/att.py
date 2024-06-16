import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)




class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    

from torch.nn import init

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from efficientnet_pytorch.model import MemoryEfficientSwish

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                         )
    def forward(self, x):
        return self.act_block(x)

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, group_split=[4, 4], kernel_sizes=[5], window_size=4, 
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        #projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2d(dim, 3*group_head*self.dim_head, 1, 1, 0, bias=qkv_bias))
            #projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias=qkv_bias)
            #self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size!=1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x) #(b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous() #(3 b (m d) h w)
        q, k, v = qkv #(b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v) #(b (m d) h w)
        return res
        
    def low_fre_attention(self, x : torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        
        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        kv = avgpool(x) #(b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h*w)//(self.window_size**2)).permute(1, 0, 2, 4, 3).contiguous() #(2 b m (H W) d)
        k, v = kv #(b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2) #(b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v #(b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

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

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


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

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class ImprovedSimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(ImprovedSimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Local statistics
        local_mean = x.mean(dim=[2, 3], keepdim=True)
        local_var = x.var(dim=[2, 3], keepdim=True)

        # Global statistics
        global_mean = self.global_pool(x)

        # Compute local and global importance
        local_importance = (x - local_mean).pow(2) / (4 * (local_var + self.e_lambda)) + 0.5
        global_importance = (x - global_mean).pow(2)

        # Combine local and global importance
        combined_importance = local_importance + global_importance

        # Apply activation and return weighted features
        return x * self.activation(combined_importance)

class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

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

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out



# import paddle
# import paddle.fluid as fluid
# import numpy as np
# import matplotlib.pyplot as plt
# from paddle.vision.datasets import Cifar10
# from paddle.vision.transforms import Transpose
# from paddle.io import Dataset, DataLoader
# from paddle import nn
# import paddle.nn.functional as F
# import paddle.vision.transforms as transforms
# import os
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# import paddlex
# import itertools
# from einops import rearrange

# class stem(nn.Layer):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.conv1 = nn.Conv2D(in_dim, out_dim // 2, 3, padding=1, stride=2, bias_attr=False)

#         self.conv2 = nn.Conv2D(out_dim // 2, out_dim, 3, padding=1, stride=2, bias_attr=False)

#         self.conv3 = nn.Conv2D(out_dim, out_dim, 3, padding=1, bias_attr=False)

#         self.conv4 = nn.Conv2D(out_dim, out_dim, 3, padding=1, bias_attr=False)

#         self.conv5 = nn.Conv2D(out_dim, out_dim, 1, bias_attr=False)

#         self.gelu = nn.GELU()
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.gelu(x)
#         x = self.conv2(x)
#         x = self.gelu(x)
#         x = self.conv3(x)
#         x = self.gelu(x)
#         x = self.conv4(x)
#         x = self.gelu(x)
#         x = self.conv5(x)
#         return x
# class CloBlock(nn.Layer):
#     def __init__(self, global_dim, local_dim, kernel_size, pool_size, head, qk_scale=None, drop_path_rate=0.0):
#         super().__init__()
#         self.global_dim = global_dim
#         self.local_dim = local_dim
#         self.head = head

#         self.norm = nn.LayerNorm(global_dim + local_dim)
        
#         # global branch
#         self.global_head = int(self.head * self.global_dim / (self.global_dim + self.local_dim))
#         self.fc1 = nn.Linear(global_dim, global_dim * 3)
#         self.pool1 = nn.AvgPool2D(pool_size)
#         self.pool2 = nn.AvgPool2D(pool_size)
#         self.qk_scale = qk_scale or global_dim ** -0.5
#         self.softmax = nn.Softmax(axis=-1)

#         # local branch
#         self.local_head = int(self.head * self.local_dim / (self.global_dim + self.local_dim))
#         self.fc2 = nn.Linear(local_dim, local_dim * 3)
#         self.qconv = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
#                 padding=kernel_size//2, groups=local_dim // self.local_head)
#         self.kconv = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
#                 padding=kernel_size//2, groups=local_dim // self.local_head)
#         self.vconv = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
#                 padding=kernel_size//2, groups=local_dim // self.local_head)
#         self.fc3 = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, 1)
#         self.swish = nn.Swish()
#         self.fc4 = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, 1)
#         self.tanh = nn.Tanh()

#         # fuse
#         self.fc5 = nn.Conv2D(global_dim + local_dim, global_dim + local_dim, 1)
#         self.drop_path = DropPath(drop_path_rate)

#     def forward(self, x):
#         identity = x

#         B, C, H, W = x.shape

#         x = rearrange(x, 'b c h w->b (h w) c')
#         x = self.norm(x)
#         x_local, x_global = paddle.split(x, [self.local_dim, self.global_dim], axis=-1)

#         # global branch
#         global_qkv = self.fc1(x_global)
#         global_qkv = rearrange(global_qkv, 'b n (m h c)->m b h n c', m=3, h=self.global_head)
#         global_q, global_k, global_v = global_qkv[0], global_qkv[1], global_qkv[2]
#         global_k = rearrange(global_k, 'b m (h w) c->b (m c) h w', h=H, w=W)
#         global_k = self.pool1(global_k)
#         global_k = rearrange(global_k, 'b (m c) h w->b m (h w) c', m=self.global_head)
#         global_v = rearrange(global_v, 'b m (h w) c->b (m c) h w', h=H, w=W)
#         global_v = self.pool1(global_v)
#         global_v = rearrange(global_v, 'b (m c) h w->b m (h w) c', m=self.global_head)
#         attn = global_q @ global_k.transpose([0, 1, 3, 2]) * self.qk_scale
#         attn = self.softmax(attn)
#         x_global = attn @ global_v
#         x_global = rearrange(x_global, 'b m (h w) c-> b (m c) h w', h=H, w=W)

#         # local branch
#         local_qkv = self.fc2(x_local)
#         local_qkv = rearrange(local_qkv, 'b (h w) (m n c)->m (b n) c h w', m=3, h=H, w=W, n=self.local_head)
#         local_q, local_k, local_v = local_qkv[0], local_qkv[1], local_qkv[2]
#         local_q = self.qconv(local_q)
#         local_k = self.kconv(local_k)
#         local_v = self.vconv(local_v)
#         attn = local_q * local_k
#         attn = self.fc4(self.swish(self.fc3(attn)))
#         attn = self.tanh(attn / (self.local_dim ** -0.5))
#         x_local = attn * local_v
#         x_local = rearrange(x_local, '(b n) c h w->b (n c) h w', b=B)

#         # Fuse
#         x = paddle.concat([x_local, x_global], axis=1)
#         x = self.fc5(x)
#         out = identity + self.drop_path(x)
#         return out
# class ConvFFN(nn.Layer):
#     def __init__(self, in_dim, out_dim, kernel_size, stride, exp_ratio=4, drop_path_rate=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(in_dim)
#         self.fc1 = nn.Conv2D(in_dim, int(exp_ratio * in_dim), 1)
#         self.gelu = nn.GELU()
#         self.dwconv1 = nn.Conv2D(int(exp_ratio * in_dim), int(exp_ratio * in_dim), kernel_size, padding=kernel_size//2, stride=stride, groups=int(exp_ratio * in_dim))
#         self.fc2 = nn.Conv2D(int(exp_ratio * in_dim), out_dim, 1)
#         self.drop_path = DropPath(drop_path_rate)

#         self.downsample = stride>1
#         if self.downsample:
#             self.dwconv2 = nn.Conv2D(in_dim, in_dim, kernel_size, padding=kernel_size//2, stride=stride, groups=in_dim)
#             self.norm2 = nn.BatchNorm2D(in_dim)
#             self.fc3 = nn.Conv2D(in_dim, out_dim, 1)
        
#     def forward(self, x):
        
#         if self.downsample:
#             identity = self.fc3(self.norm2(self.dwconv2(x)))
#         else:
#             identity = x

#         x = rearrange(x, 'b c h w->b h w c')
#         x = self.norm1(x)
#         x = rearrange(x, 'b h w c->b c h w')

#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.dwconv1(x)
#         x = self.fc2(x)

#         out = identity + self.drop_path(x)
#         return out
# class CloFormer(nn.Layer):
#     def __init__(self, global_dim, local_dim, heads, in_dim=3, num_classes=1000, depths=[2, 2, 6, 2], attnconv_ks=[3, 5, 7, 9],
#                 pool_size=[8, 4, 2, 1], convffn_ks=5, convffn_ratio=4, drop_path_rate=0.0):
#         super().__init__()

#         dprs = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]

#         self.stem = stem(in_dim, global_dim[0] + local_dim[0])

#         for i in range(len(depths)):
#             layers = []
#             dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
#             for j in range(depths[i]):
#                 if j < depths[i] - 1 or i == len(depths) - 1:
#                     layers.append(
#                         nn.Sequential(
#                             CloBlock(global_dim[i], local_dim[i], attnconv_ks[i], pool_size[i], heads[i], dpr[j]),
#                             ConvFFN(global_dim[i] + local_dim[i], global_dim[i] + local_dim[i], convffn_ks, 1, convffn_ratio, dpr[j])
#                         )
#                     )
#                 else:
#                     layers.append(
#                         nn.Sequential(
#                             CloBlock(global_dim[i], local_dim[i], attnconv_ks[i], pool_size[i], heads[i], dpr[j]),
#                             ConvFFN(global_dim[i] + local_dim[i], global_dim[i + 1] + local_dim[i + 1], convffn_ks, 2, convffn_ratio, dpr[j])
#                         )
#                     )

#             self.__setattr__(f'stage{i}', nn.LayerList(layers))
        
#         self.norm = nn.LayerNorm(global_dim[-1] + local_dim[-1])
        
#         self.head = nn.Linear(global_dim[-1] + local_dim[-1], num_classes)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         tn = nn.initializer.TruncatedNormal(std=.02)
#         ones = nn.initializer.Constant(1.0)
#         zeros = nn.initializer.Constant(0.0)
#         if isinstance(m, (nn.Conv2D, nn.Linear)):
#             tn(m.weight)
#             if m.bias is not None:
#                 zeros(m.bias)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
#             zeros(m.bias)
#             ones(m.weight)

#     def forward_feature(self, x):
#         for blk in self.stage0:
#             x = blk(x) 
        
#         for blk in self.stage1:
#             x = blk(x) 

#         for blk in self.stage2:
#             x = blk(x) 

#         for blk in self.stage3:
#             x = blk(x) 

#         x = rearrange(x, 'b c h w-> b h w c')
#         x = self.norm(x)
#         return x

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.forward_feature(x)
#         x = paddle.mean(x, axis=[1, 2])
#         x = self.head(x)
#         return x

###################### PolarizedSelfAttention     ####     start    ###############################
import torch
import torch.nn as nn
from torch.nn import functional as F
 
 
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
 
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
    def forward(self, x):
        residual = x
 
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
 
        x = torch.sigmoid(x) * residual  #
 
        return x
 
 
class NAMAttention(nn.Module):
    def __init__(self, channels, shape, out_channels=None, no_spatial=True):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)
 
    def forward(self, x):
        x_out1 = self.Channel_Att(x)
 
        return x_out1
###################### PolarizedSelfAttention    ####     end    ###############################

 
import torch
import torch.nn as nn
from torch.nn import functional as F
 
###################### TripletAttention  ####     start   ###############################
 
class BasicConv(nn.Module):  # https://arxiv.org/pdf/2010.03045.pdf
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
 
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
 
 
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
 
 
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
 
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale
 
 
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
 
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out



import torch
from torch import nn
import math
 
# __all__ = ['MCALayer', 'MCAGate']
 
 
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()
 
    def forward(self, x):
        b, c, _, _ = x.size()
 
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
 
        return std
 
 
class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()
 
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
 
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
 
        self.weight = nn.Parameter(torch.rand(2))
 
    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
 
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"
 
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
 
        out = self.sigmoid(out)
        out = out.expand_as(x)
 
        return x * out
 
 
class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=True):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()
 
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
 
        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)
 
    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()
 
        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()
 
        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)
 
        return x_out

import torch
import torch.nn as nn
import torchvision
 
class DeformConv(nn.Module):
 
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)
 
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)
 
    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out
    
class DeformConv_3x3(nn.Module):
 
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)
 
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)
 
    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out
    
 
class DeformConv_experimental(nn.Module):
 
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_experimental, self).__init__()
        
        self.conv_channel_adjust = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size[0] * kernel_size[1], kernel_size=(1,1))
 
        self.offset_net = nn.Conv2d(in_channels=2 * kernel_size[0] * kernel_size[1],
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    groups=2 * kernel_size[0] * kernel_size[1],
                                    bias=True)
 
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)
 
    def forward(self, x):
        x_chan = self.conv_channel_adjust(x)
        offsets = self.offset_net(x_chan)
        out = self.deform_conv(x, offsets)
        return out
 
 
class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
 
 
    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
 
        return u * attn
 
 
class deformable_LKA_experimental(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv_experimental(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv_experimental(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
 
 
    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
 
        return u * attn
 
 
class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
 
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
 
    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
 
class deformable_LKA_Attention_experimental(nn.Module):
    def __init__(self, d_model):
        super().__init__()
 
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA_experimental(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
 
    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
 
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
 
 
 
 
 
class C2f_deformable_LKA(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(deformable_LKA(self.c) for _ in range(n))
 
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self, channel=512,out_channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V
