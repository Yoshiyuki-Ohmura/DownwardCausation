import os
import shutil
from turtle import color
from typing import Iterable, Optional
from typing import Callable, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import model.injectiveLinear as injL
import model.injectiveConv as injC
import math


class Gp_model_inside(nn.Module):
    def __init__(self, label_dim: int, bias:bool, p_ch:Optional[int]=64, in_ch:Optional[int]=3):
        super(Gp_model_inside, self).__init__()
        self.act = nn.ReLU()

        self.in_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, p_ch*2, 4, 2, 1,  bias=bias)  #(3, 32,32) -> (pch*2, 16,16)
        self.conv2 = nn.Conv2d(p_ch*2,p_ch*4,4,2, 1, bias=bias) #(pch*2, 16,16) -> (pch*4, 8,8) 
        self.conv3 = nn.Conv2d(p_ch*4,p_ch*4,4,2, 1,  bias=bias) #(pch*4, 8,8) -> (pch*4, 4,4)
        #self.l1  = nn.Linear(4*4*p_ch*4, 4*4*p_ch*4, bias=bias) 
        #self.l2 = nn.Linear(4*4*p_ch*4, label_dim, bias=bias)  
        self.l1  = nn.Linear(4*4*p_ch*4, p_ch, bias=bias) 
        self.l2 = nn.Linear(p_ch, label_dim, bias=bias)  
        self.flt = nn.Flatten()

    def forward(self, x:Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.act(out)
        out = self.flt(out)
        out = self.l1(out)
        out = self.act(out)
        out = self.l2(out)
        return out


class Gp_model(nn.Module):
    def __init__(self, label_dim:Iterable[int],  p_ch:Optional[int]=64, in_ch:Optional[int]=3, gp_bias:Optional[bool]=False):
        super(Gp_model, self).__init__()
        self.in_ch = in_ch
        self.Gp0 = Gp_model_inside(label_dim[0], bias=gp_bias, p_ch=p_ch, in_ch=in_ch)
        self.Gp1 = Gp_model_inside(label_dim[1], bias=gp_bias, p_ch=p_ch, in_ch=in_ch)

    def forward(self, x:Tensor) ->Iterable[Tensor]:
        out0 = self.Gp0(x)
        out1 = self.Gp1(x)
        return (out0,out1)

