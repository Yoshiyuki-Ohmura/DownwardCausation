from re import M
from typing import Callable, Optional, Union, Iterable
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class ConvTranspose2d_weight_norm(nn.Module):
    # modified Weight Normalization 
    def __init__(self, in_ch:int, out_ch:int, kernel:int, stride:int, padding:int):
        super(ConvTranspose2d_weight_norm, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dim = 0

        weight = torch.empty(self.in_ch, self.out_ch, self.kernel, self.kernel)
        torch.nn.init.kaiming_uniform_(weight, a=1.)
        weight = weight - weight.mean(dim=self.dim, keepdim=True)
        self.weight = torch.nn.parameter.Parameter(weight)
        g = torch.sqrt( weight.square().sum(dim=self.dim, keepdim=True))
        self.g = torch.nn.parameter.Parameter(g)
        self.g.requires_grad = False

    def forward(self, x:Tensor) -> Tensor:
        m=self.weight.mean(dim=self.dim, keepdim=True)
        weight = self.weight - m
        n = torch.sqrt ( weight.square().sum(dim=self.dim, keepdim=True) )
        weight = torch.abs(self.g) * weight/(n+1e-7)
        return F.conv_transpose2d(x, weight, None, self.stride, self.padding, 0, 1, 1)


class ConvTranspose2d4x4(nn.Module):
    def __init__(self, in_ch:int, out_ch:int):
        super(ConvTranspose2d4x4, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = 4
        self.stride = 2
        self.padding = 1

        weight = torch.empty(self.in_ch, self.out_ch, 4, 4)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = torch.nn.parameter.Parameter(weight)
    def forward(self, x:Tensor) -> Tensor:
        return F.conv_transpose2d(x, self.weight, None, self.stride, self.padding, 0, 1, 1)


class iConvTranspose2d(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, kernel_size:Optional[int]=4, stride:Optional[int]=2, padding:Optional[int]=1):
        super(iConvTranspose2d, self).__init__()

        self.conv =ConvTranspose2d4x4(in_ch, int(in_ch/4))
        self.in_ch = in_ch

    def forward(self, x:Tensor)->Tensor:
        out0 = self.conv(x)
        out1 = -out0
        ret = torch.cat((out0,out1),dim=1)
        return ret

