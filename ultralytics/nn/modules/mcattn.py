import numpy as np
from typing import Any, Callable
 
import torch
from torch import nn, Tensor
from typing import List, Optional
import math
 
from ultralytics.nn.modules.conv import Conv
 
 
def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.Py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
 
def callMethod(self, ElementName):
    return getattr(self, ElementName)
 
def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)
def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    # shuffle multiple tensors with the same indexs
    # all tensors must have the same shape
    if isinstance(Feature, Tensor):
        Feature = [Feature]
 
    Indexs = None
    Output = []
    for f in Feature:
        # not in-place operation, should update output
        B, C, H, W = f.shape
        if Mode == 1:
            # fully shuffle
            f = f.flatten(2)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, Indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        else:
            # shuflle along y and then x axis
            if Indexs is None:
                Indexs = [torch.randperm(H, device=f.device),
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
        Output.append(f)
    return Output
 
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)
 
    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0
 
 
class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size)
 
    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0
 
 
 
 
NormLayerTuple = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.BatchNorm3d,
)
 
 
def initWeight(Module):
    # init conv, norm , and linear layers
    ## empty module
    if Module is None:
        return
    ## conv layer
    elif isinstance(Module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(Module.bias, -bound, bound)
    ## norm layer
    elif isinstance(Module, NormLayerTuple):
        if Module.weight is not None:
            nn.init.ones_(Module.weight)
        if Module.bias is not None:
            nn.init.zeros_(Module.bias)
    ## linear layer
    elif isinstance(Module, nn.Linear):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(Module.bias, -bound, bound)
    elif isinstance(Module, (nn.Sequential, nn.ModuleList)):
        for m in Module:
            initWeight(m)
    elif list(Module.children()):
        for m in Module.children():
            initWeight(m)
 
class BaseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
 
        if bias is None:
            bias = not BNorm
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
 
        self.Conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)
 
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()
 
        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer
 
        self.apply(initWeight)
 
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x
 
    def profileModule(self, Input: Tensor):
        if Input.dim() != 4:
            print('Conv2d requires 4-dimensional Input (BxCxHxW). Provided Input has shape: {}'.format(Input.size()))
 
        BatchSize, in_channels, in_h, in_w = Input.size()
        assert in_channels == self.in_channels, '{}!={}'.format(in_channels, self.in_channels)
 
        k_h, k_w = pair(self.kernel_size)
        stride_h, stride_w = pair(self.stride)
        pad_h, pad_w = pair(self.padding)
        groups = self.groups
 
        out_h = (in_h - k_h + 2 * pad_h) // stride_h + 1
        out_w = (in_w - k_w + 2 * pad_w) // stride_w + 1
 
        # compute MACs
        MACs = (k_h * k_w) * (in_channels * self.out_channels) * (out_h * out_w) * 1.0
        MACs /= groups
 
        if self.bias:
            MACs += self.out_channels * out_h * out_w
 
        # compute parameters
        Params = sum([p.numel() for p in self.parameters()])
 
        Output = torch.zeros(size=(BatchSize, self.out_channels, out_h, out_w), dtype=Input.dtype, device=Input.device)
        # print(MACs)
        return Output, Params, MACs
 
 
class MoCAttention(nn.Module):
    # Monte carlo attention
    def __init__(
        self,
        InChannels: int,
        HidChannels: int=None,
        SqueezeFactor: int=4,
        PoolRes: list=[1, 2, 3],
        Act: Callable[..., nn.Module]=nn.ReLU,
        ScaleAct: Callable[..., nn.Module]=nn.Sigmoid,
        MoCOrder: bool=True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)
        
        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)
            
        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
        
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
        
    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None] # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)
            
        return AttnMap
        
    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)
 
 
class PSMoCA(nn.Module):
 
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
 
        self.attn = MoCAttention(self.c)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )
 
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
