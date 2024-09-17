#from dbm import gnu
import os
import shutil
from turtle import color
from typing import Iterable, Optional
from typing import Callable, Optional, Union
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair 
from torch.nn.common_types import _size_2_t
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import model.injectiveLinear as injL
import model.injectiveConv as injC

class Gn_model(nn.Module):
    def __init__(self, label_dim: Iterable[int], n_ch: int, 
                orth:Optional[bool]=True):

        super(Gn_model, self).__init__()
        self.latent = label_dim
        self.relu = nn.ReLU()

        self.uf1 = nn.Unflatten(-1, ( (n_ch)*8, 4, 4))
        self.nch1 = n_ch

        self.l1 = nn.Linear(sum(label_dim), 2*sum(label_dim), bias=True)
        self.l2 = nn.Linear(2*sum(label_dim), n_ch*8*4, bias=True)
        self.l3 = nn.Linear(n_ch*8*4, n_ch*8*4*4, bias=True)
        self.convt1 = nn.ConvTranspose2d((n_ch)*8, (n_ch)*4, 4, 2, 1, bias=True)
        self.convt2 = nn.ConvTranspose2d((n_ch)*4, (n_ch)*2, 4, 2, 1, bias=True)
        self.convt3 = nn.ConvTranspose2d((n_ch)*2, (n_ch), 4, 2, 1, bias=True)

        #self.conv = injC.ConvTranspose2d_weight_norm(n_ch, n_ch, 1, 1, 0)
        self.conv1 = torch.nn.ConvTranspose2d(n_ch, 3, 1, 1, 0, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(n_ch, n_ch-3, 1, 1, 0, bias=False)
        self.conv2.requires_grad = False
        self.n_ch1 = n_ch

    def forward(self, x:Tensor) :

        out = self.relu( self.l1(x) )
        out = self.relu( self.l2(out))
        out = self.relu( self.l3(out))

        out = self.uf1(out)
        out = self.relu( self.convt1(out) ) 
        out = self.relu( self.convt2(out) )
        out = self.relu( self.convt3(out) )
        self.preout =out 
        ret = self.conv1( out )
        n = self.conv2( out )
        return ret, n 

