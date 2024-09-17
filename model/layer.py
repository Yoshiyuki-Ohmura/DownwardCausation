from typing import Iterable
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import numpy as np

class SplitActivation(nn.Module):
    def __init__(self, dims: Iterable[int], activation: Iterable[nn.Module]):
        super(SplitActivation, self).__init__()
        assert len(dims) == len(activation)
        self.dims = dims
        self.activation = nn.ModuleList(activation)

    def forward(self, x: torch.Tensor):
        # assert x.dim() == 2
        xs = torch.split(x, self.dims, dim=1)
        outs = [act(sub_x) for sub_x, act in zip(xs, self.activation)]
        out = torch.cat(outs, dim=1)
        return out

class SplitLatent(nn.Module):
    def __init__(self, in_out_features:int, a:float, device=None, dtype=None) -> None:
        super(SplitLatent, self).__init__()
        self.in_features = in_out_features
        self.out_features = in_out_features
        assert in_out_features%4==0
        self.latent=[int(in_out_features/4)]*4
        self.a=a

    def forward(self, input:Tensor) -> Tensor:
        x0,x1,x2,x3=torch.split(input,self.latent,dim=1)
        _x0 = (3.*self.a+1.)/4.*x0 + (1.-self.a)/4.*(x1+x2+x3)
        _x1 = (3.*self.a+1.)/4.*x1 + (1.-self.a)/4.*(x0+x2+x3)
        _x2 = (3.*self.a+1.)/4.*x2 + (1.-self.a)/4.*(x0+x1+x3)
        _x3 = (3.*self.a+1.)/4.*x3 + (1.-self.a)/4.*(x0+x1+x2)
        out=torch.cat((_x0,_x1,_x2,_x3), dim=1)
        return out
        
class SkipConv(nn.Module):
    def __init__(self, features:int, bias:bool =True, groups:int=1) -> None:
        super(SkipConv, self).__init__()
        self.conv = nn.Conv2d(features, features, 1,1,0, bias=bias, groups=groups)
    
    def forward(self, x:Tensor) -> Tensor:
        out=self.conv(x)
        out += x
        return out


class GroupLinear(nn.Module):
    def __init__(self, in_features: int, out_features:int, bias: bool =True, groups: int =1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype':dtype}
        super(GroupLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups=groups
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        x = self.weight.to('cpu').detach().numpy().copy()
        if self.groups==1:
            mask = np.random.rand(*x.shape)
            mask = np.where(mask>0.5 ,1 ,0 )
            x= x*mask
        else:
            x[0:int(self.out_features/2), int(self.in_features/2):self.in_features]=0.
            x[int(self.out_features/2):self.out_features, 0:int(self.in_features/2)]=0.
        self.weight=Parameter(torch.from_numpy(x).type(torch.FloatTensor))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in) if fan_in >0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input:Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class SkipGroupLinear(nn.Module):
    def __init__(self, features: int, bias: bool =True, groups: int =1) -> None:
        super(SkipGroupLinear, self).__init__()
        self.grlinear = GroupLinear(features, features, bias=bias, groups=groups)
        self.relu = nn.ReLU()
        self.shortcut =nn.Sequential()

    def forward(self, x:Tensor) -> Tensor:
        out=self.grlinear(x)
        out=self.relu(out)
        out = out +  self.shortcut(x)
        return out

class SkipLinear(nn.Module):
    def __init__(self,in_features:int, out_features:int, bias:bool = True)->None:
        super(SkipLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.unflatten = nn.Unflatten(-1, (1,in_features))
        self.upsample = nn.Upsample(scale_factor=float(out_features/in_features))
        self.flatten = nn.Flatten()
        self.ReLU = nn.ReLU()
    def forward(self, x:Tensor)->Tensor:
        out = self.linear(x)
        out = self.ReLU(out)
        _x = self.unflatten(x)
        _x = self.upsample(_x)
        _x = self.flatten(_x)
        out = out + _x
        return out

class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features:int, bias: bool =True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype':dtype}
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.mask = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        with torch.no_grad():
            self.beta = 50
        self.sig_mask = torch.sigmoid(self.mask * self.beta )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.zeros_(self.mask)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in) if fan_in >0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input:Tensor) -> Tensor:
        self.sig_mask = torch.sigmoid(self.mask * self.beta  )
        weight_tmp = self.sig_mask * self.weight 
        return F.linear(input, weight_tmp, self.bias)

    def loss_fn(self) -> Tensor: 
        return self.sig_mask.mean()

    def sparseness(self) -> float:
        mask_bin = torch.where(self.sig_mask > 0.1, torch.ones_like(self.sig_mask), torch.zeros_like(self.sig_mask))
        one_cnt = float(torch.count_nonzero(mask_bin))
        one_cnt /= self.in_features
        one_cnt /= self.out_features
        return one_cnt

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )