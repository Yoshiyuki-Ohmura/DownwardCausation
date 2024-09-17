import argparse
import os
import shutil
import torch
import math
from turtle import color
from typing import Iterable, Optional
from typing import Callable, Optional, Union

from model.main_model import BisectionTrans
from model import Gn_model as nIJ
from model import Gp_model as Gpm

def init_modelInj(label_dim: Iterable[int], p_ch:int, n_ch:int, add_ch: Optional[bool]=False,
                  orth:Optional[bool]=True, gp_bias:Optional[bool]=False) -> BisectionTrans:
    assert len(label_dim) == 2

    print("pch:"+str(p_ch))
    print("nch:"+str(n_ch))

    Gn = nIJ.Gn_model(label_dim, n_ch=n_ch,  orth=orth)

    if add_ch:
        Gp = Gpm.Gp_model(label_dim, p_ch=p_ch, in_ch=Gn.n_ch1, gp_bias=gp_bias)
    else:
        Gp = Gpm.Gp_model(label_dim, p_ch=p_ch, in_ch=3, gp_bias=gp_bias)

    model = BisectionTrans(label_dim, Gp, Gn)
    return model

