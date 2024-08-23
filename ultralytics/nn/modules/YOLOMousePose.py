import torch
from torch import nn
import torch.nn.functional as F
 
import numpy as np
 
from ultralytics.nn.modules import Conv,Bottleneck,C2f
 
def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x
 
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class FusionEncoder(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[C2f(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc))
        )
    def forward(self, x):
        return self.conv(x)
 
 
# import torch
class WeightedInject(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: int
    ) -> None:
        super().__init__()
        self.global_inp = global_inp
        oup_2=int(oup/2)
        self.local_embedding = Conv(inp, oup_2, 1, act=False)
        self.global_embedding = Conv(global_inp, oup_2, 1,act=False)
        self.global_act = Conv(global_inp, oup_2, 1, act=False)
        self.act = h_sigmoid()
 
 
    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x
 
        gloabl_info = x_g
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(gloabl_info)
        sig_act = self.act(global_act)
        global_inj=gloabl_info * sig_act
        out = torch.cat((local_feat , global_inj), dim=1)
        return out
 
class SE_HALF(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE_HALF, self).__init__()
        self.c1 = c1  # Record the number of input channels
        # c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
 
        # Sort channel weights
        _, indices = torch.sort(y, descending=True)
 
        # Retrieve the index of the top 50% channels
        half_ch = c // 2
        indices = indices[:, :half_ch]
 
        # Rearrange input channels based on sorting and only retain the first half
        x_sorted_half = x.new_zeros((b, half_ch, h, w))
        for b_idx in range(x.size(0)):
            x_sorted_half[b_idx] = x[b_idx, indices[b_idx]]
 
        y_half = y.gather(1, indices)
 
        y_half = y_half.view(b, half_ch, 1, 1)
 
        # Note that at this point, the channels x_sorted_half and y_half have already been arranged correspondingly
 
        return x_sorted_half * y_half.expand_as(x_sorted_half)
 
class ECA_SORT(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super(ECA_SORT, self).__init__()
        self.c1 = c1  # the number of input channels
        self.c2 = c2  # the number of output channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # Calculate channel weights
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y).squeeze(-1).squeeze(-1)  # 将权重调整为一维
 
        # Sort the channel weights
        _, indices = torch.sort(y, descending=True)
 
        # Retrieve the index of the first c2 channels
        indices = indices[:, :self.c2]
 
        # Rearrange input channels based on sorting and only retain the first c2 channels
        b, _, h, w = x.size()
        x_sorted_out = x.new_zeros((b, self.c2, h, w))
        for b_idx in range(b):
            x_sorted_out[b_idx] = x[b_idx, indices[b_idx]]
 
        return x_sorted_out
 
class SE_SORT(nn.Module):
    def __init__(self, c1,c2 ,ratio=16):
        super(SE_SORT, self).__init__()
        self.c1 = c1  # Record the number of input channels
        self.c2 = c2  # Record the number of output channels
        # c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        # Sort channel weights
        _, indices = torch.sort(y, descending=True)
        # Obtain the index of the output channel
        out_ch =self.c2
        indices = indices[:, :out_ch]
        # Rearranges the input channels according to sorting and retains only the top half.
        x_sorted_out = x.new_zeros((b, out_ch, h, w))
        for b_idx in range(x.size(0)):
            x_sorted_out[b_idx] = x[b_idx, indices[b_idx]]
        # y_out = y.gather(1, indices)
        # y_out = y_out.view(b, out_ch, 1, 1)
        return x_sorted_out
 
class ChannelSelection_Top(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        out_channels_2=int(out_channels/3)
        self.cv0 = nn.Sequential(SE_SORT(in_channel_list[0], out_channels_2) if in_channel_list[0] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2, act=nn.ReLU()) if in_channel_list[0] > out_channels_2
                                 else Conv(in_channel_list[0], out_channels_2, act=nn.ReLU())
                                 )
        self.cv2 = nn.Sequential(SE_SORT(in_channel_list[2], out_channels_2) if in_channel_list[
                                                                                    2] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2, act=nn.ReLU()) if in_channel_list[2] > out_channels_2
                                 else Conv(in_channel_list[2], out_channels_2, act=nn.ReLU())
                                 )
        self.cv3 = nn.Sequential(SE_SORT(in_channel_list[3], out_channels_2) if in_channel_list[
                                                                                    3] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2, act=nn.ReLU()) if in_channel_list[
                                                                                            3] > out_channels_2
                                 else Conv(in_channel_list[3], out_channels_2, act=nn.ReLU())
                                 )
        self.downsample = nn.functional.adaptive_avg_pool2d
 
    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)
 
        x_l = self.cv0(self.downsample(x[0], output_size))
        x_s =self.cv2(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))
        x_n = self.cv3(F.interpolate(x[3], size=(H, W), mode='bilinear', align_corners=False))
        out = torch.cat([x_l,x_s,x_n], 1)
        return out
 
class ChannelSelection_Medium(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        out_channels_2 = int(out_channels /3)
        self.cv0 = nn.Sequential(SE_SORT(in_channel_list[0], out_channels_2) if in_channel_list[
                                                                                    0] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2, act=nn.ReLU()) if in_channel_list[
                                                                                            0] > out_channels_2
                                 else Conv(in_channel_list[0], out_channels_2, act=nn.ReLU())
                                 )
        self.cv1 = nn.Sequential(SE_SORT(in_channel_list[1], out_channels_2) if in_channel_list[
                                                                                    1] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2, act=nn.ReLU()) if in_channel_list[
                                                                                            1] > out_channels_2
                                 else Conv(in_channel_list[1], out_channels_2, act=nn.ReLU())
                                 )
        self.cv3 = nn.Sequential(SE_SORT(in_channel_list[3], out_channels_2) if in_channel_list[
                                                                                    3] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2, act=nn.ReLU()) if in_channel_list[
                                                                                            3] > out_channels_2
                                 else Conv(in_channel_list[3], out_channels_2, act=nn.ReLU())
                                 )
        self.downsample = nn.functional.adaptive_avg_pool2d
 
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x[2].shape
        output_size = np.array([H, W])
 
        #
        x_l = self.cv0(self.downsample(x[0], output_size))
        x_m = self.cv1(self.downsample(x[1], output_size))
        x_n = self.cv3(F.interpolate(x[3], size=(H, W), mode='bilinear', align_corners=False))
 
 
        out = torch.cat([x_l, x_m, x_n], 1)
        return out
 
class ChannelSelection_Bottom(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        out_channels_2 = int(out_channels / 3)
        self.cv0 = nn.Sequential(SE_SORT(in_channel_list[0], out_channels_2) if in_channel_list[
                                                                                    0] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2, out_channels_2,act=nn.ReLU()) if in_channel_list[
                                                                                            0] > out_channels_2
                                 else Conv(in_channel_list[0], out_channels_2, act=nn.ReLU())
                                 )
 
        self.cv1 = nn.Sequential(SE_SORT(in_channel_list[1],  out_channels_2) if in_channel_list[
                                                                                    1] >  out_channels_2 else nn.Identity(),
                                 Conv( out_channels_2,  out_channels_2 ,act=nn.ReLU()) if in_channel_list[
                                                                                            1] >  out_channels_2
                                 else Conv(in_channel_list[1],  out_channels_2, act=nn.ReLU())
                                 )
 
        self.cv2 = nn.Sequential(SE_SORT(in_channel_list[2], out_channels_2+1) if in_channel_list[
                                                                                    2] > out_channels_2 else nn.Identity(),
                                 Conv(out_channels_2+1, out_channels_2+1, act=nn.ReLU()) if in_channel_list[
                                                                                            2] > out_channels_2
                                 else Conv(in_channel_list[2], out_channels_2+1, act=nn.ReLU())
                                 )
        self.downsample = nn.functional.adaptive_avg_pool2d
 
 
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x[3].shape
 
        output_size = np.array([H, W])
 
        x_l = self.cv0(self.downsample(x_l, output_size))
        x_m = self.cv1(self.downsample(x_m, output_size))
        x_s =self.cv2(self.downsample(x_s, output_size))
 
 
        out = torch.cat([x_l, x_m, x_s], 1)
        return out
 
 
 
 
 
