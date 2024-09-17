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

class Linear_wn(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super(Linear_wn, self).__init__()
        weight = torch.empty(out_dim, in_dim)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight=nn.parameter.Parameter(weight)
    def forward(self, input:Tensor) -> Tensor:
        weight = self.weight - self.weight.mean(dim=0, keepdim=True)
        n = torch.sqrt ( weight.square().sum(dim=0, keepdim=True) )
        weight =  weight/(n+1e-7)
        return F.linear(input, weight, None)        

class iLinear_mod(nn.Module):
    def __init__(self, in_dim:int, eps:Optional[float]=0., orth:Optional[bool]=True, w_norm:Optional[bool]=False):
        super(iLinear_mod, self).__init__() 

        if w_norm:
            print("iLinear: Weight Norm")
            self.l0 = Linear_wn(in_dim, in_dim) 
        else:
            if orth:
                self.l0 = nn.utils.parametrizations.orthogonal( nn.Linear(in_dim, in_dim, bias=False) )
            else:
                print("Orthogonal is False")
                self.l0 = nn.Linear(in_dim, in_dim, bias=False) 
            
        self.eps = eps
    def forward(self, x:Tensor) -> Tensor:
        out = self.l0( x )
        out0 =  out 
        out1 = -out
        ret = torch.cat((out0,out1),dim=1)
        return ret
    def para_reset(self):
        torch.nn.init.orthogonal_(self.l0.weight)

class iLinear_mod_bias(nn.Module):
    def __init__(self, in_dim:int, zero_bias:Optional[bool]=False, orth:Optional[bool]=True):
        super(iLinear_mod_bias, self).__init__() 

        self.b = nn.parameter.Parameter( torch.zeros(in_dim) )
        torch.nn.init.uniform_(self.b, -1/math.sqrt(in_dim), 1/math.sqrt(in_dim))
        if orth:
            self.l0 = nn.utils.parametrizations.orthogonal( nn.Linear(in_dim*2, in_dim*2, bias=False) )
        else:
            self.l0 = nn.Linear(in_dim*2, in_dim*2, bias=False)
        if zero_bias:
            torch.nn.init.zeros_(self.b)
            self.b.requires_grad=False

    def forward(self, x:Tensor) -> Tensor:
        b = self.b.repeat( x.size()[0],1)
        xin = torch.cat((x,b),dim=1)
        out = self.l0( xin )
        out0 =  out 
        out1 = -out
        ret = torch.cat((out0,out1),dim=1)
        return ret
    def para_reset(self):
        torch.nn.init.orthogonal_(self.l0.weight)


class iLinear_bias(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, orth:Optional[bool]=True):
        super(iLinear_bias,self).__init__()
        assert out_dim>=2*in_dim
        print("iLinear bias: " + str(int(out_dim/2 - in_dim )))
        self.b = nn.parameter.Parameter( torch.zeros( int(out_dim/2)- in_dim))
        torch.nn.init.uniform_(self.b, -1/math.sqrt(in_dim), 1/math.sqrt(in_dim))
        if orth:
            self.l0 =nn.utils.parametrizations.orthogonal(nn.Linear(int(out_dim/2), int(out_dim/2), bias=False))
        else:
            print("orth False")
            self.l0 = nn.Linear(int(out_dim/2), int(out_dim/2), bias=False)

    def forward(self, x:Tensor)->Tensor:
        b = self.b.repeat(x.size()[0],1)
        xin = torch.cat((x,b), dim=1)
        out = self.l0(xin)
        out0 = out
        out1 = -out
        ret = torch.cat((out0, out1),dim=1)
        return ret
    def para_reset(self):
        torch.nn.init.orthogonal_(self.l0.weight)

class giLinear(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super(giLinear, self).__init__()
        self.l1 = nn.Linear(in_dim, int(out_dim/2),bias=False)
    def forward(self, x:Tensor)->Tensor:
        out0=self.l1(x)
        out1 = -out0
        ret = torch.cat((out0, out1),dim=1)
        return ret

class iLinear(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, orth:Optional[bool]=True, w_norm:Optional[bool]=False):
        super(iLinear, self).__init__()
        assert out_dim>=2*in_dim

        if orth==False:
            self.l1 = iLinear_mod(in_dim, orth=orth, w_norm=w_norm)
        else:
            self.l1 = iLinear_mod(in_dim, orth=orth, w_norm=w_norm)
        #self.l2 = nn.Linear(in_dim, int(out_dim/2)-in_dim, bias=False)
        if w_norm:
            self.l2 = Linear_wn(in_dim, int(out_dim/2)-in_dim)
        else:
            self.l2 = nn.Linear(in_dim, int(out_dim/2)-in_dim, bias=False)
    def forward(self, x:Tensor)->Tensor:
        out = self.l1(x)
        out0 = self.l2(x)
        out1 = -out0
        ret = torch.cat((out, out0, out1), dim=1)
        return ret

class iLinear0(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, orth:Optional[bool]=True, w_norm:Optional[bool]=False):
        super(iLinear0, self).__init__()
        assert out_dim>=2*in_dim

        if orth==False:
            self.l1 = iLinear_mod(in_dim, orth=orth, w_norm=True)
        else:
            self.l1 = iLinear_mod(in_dim, orth=orth, w_norm=False)
        #self.l2 = nn.Linear(in_dim, int(out_dim/2)-in_dim, bias=False)
        self.dim = out_dim - in_dim*2
    def forward(self, x:Tensor)->Tensor:
        out = self.l1(x)
        bs, in_size = x.size()
        z = torch.zeros(bs, self.dim, device=x.device)
        ret = torch.cat((out, z), dim=1)
        return ret
