import torch
import torch.nn as nn
 
from ultralytics.nn.modules.conv import Conv,autopad
 
 
class SimAMWithSlicing(nn.Module):
    def __init__(self,e_lambda=1e-4):
        super(SimAMWithSlicing, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
 
        block_size_h = height // 2
        block_size_w = width // 2
 
 
        block1 = x[:, :, :block_size_h, :block_size_w]
        block2 = x[:, :, :block_size_h, block_size_w:]
        block3 = x[:, :, block_size_h:, :block_size_w]
        block4 = x[:, :, block_size_h:, block_size_w:]
 
        enhanced_blocks = []
        for block in [block1, block2, block3, block4]:
            n = block_size_h * block_size_w - 1
            block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
            y = block_minus_mu_square / (
                        4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
            enhanced_blocks.append(block * self.activation(y))
 
        enhanced_image = torch.cat([torch.cat([enhanced_blocks[0], enhanced_blocks[1]], dim=3),
                                    torch.cat([enhanced_blocks[2], enhanced_blocks[3]], dim=3)], dim=2)
 
        return enhanced_image
 
 
 
class Conv_SWS(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_SWS, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SimAMWithSlicing(c2)
 
    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))
 
    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))
