from asyncio import BaseTransport
from collections import defaultdict
from operator import truediv
import os
import math
from re import A
from turtle import color
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import trange, tqdm
from utils import formatting, eval
import numpy as np

class BisectionTrans:

    def __init__(self,
                 latent: tuple[int, int],
                 Gp: nn.Module,
                 Gn: nn.Module,
                 device: torch.device = "cpu",
                 ):
        self.latent = latent
        self.Gp = Gp
        self.Gn = Gn
        self.device = torch.device(device)
        self.to(self.device)

    def config(self,
               color_random:Optional[bool]=True,
               add_ch: Optional[bool] = False,
               ):
        self.color_random=color_random
        self.add_ch = add_ch

    def eval_config(self,
                    batch_size:Optional[int]=None):
        self.batch_size=batch_size

    def Gn_config(self,
                  orth:bool,
                    ):
        self.orth = orth

    def train_config(self,
                     optimizer: optim.Optimizer,
                     commute: Optional[float] = 1.0,
                     learning_rate: Optional[ float ] = None,
                     aux_rate: Optional[float]=0., 
                     batch_size: Optional [int ] =None,
                     color_random: Optional[ bool ] = True,
                     add_ch: Optional[bool] = False,
                     ):
        self.optimizer = optimizer
        self.commute = commute
        self.learning_rate = learning_rate
        self.aux_rate = aux_rate
        self.batch_size = batch_size
        pattern_loss_fn = nn.MSELoss()
        self.pattern_loss_fn = pattern_loss_fn
        self.config(color_random, add_ch)

    def log_config(self, logdir: Optional[str] = None):
        """Set members related to logging."""
        print(logdir)
        self.logdir = logdir
        if self.logdir:
            self.writer = SummaryWriter(self.logdir)

    def save(self, f: str, prefix: str = "models"):
        """Save the model in a file. Location is `self.logdir`/`prefix`/`f`"""
        if self.logdir is None:
            # logger.warn("call `log_config()` before calling `save()`.")
            return

        if not os.path.exists(os.path.join(self.logdir, prefix)):
            os.makedirs(os.path.join(self.logdir, prefix))

        torch.save({
            "Gp_state_dict": self.Gp.state_dict(),
            "Gn_state_dict": self.Gn.state_dict(),
        }, os.path.join(self.logdir, prefix, f))

    def load_state_dict(self, state_dict):
        """Load state dict to recall learned params.

        Usage
        -----
        >>> model.load_state_dict(torch.load("path/to/saved_model.pt"))
        """
        self.Gp.load_state_dict(state_dict["Gp_state_dict"], strict=False)
        self.Gn.load_state_dict(state_dict["Gn_state_dict"], strict=False)

    def to(self, device: torch.device):
        self.device = torch.device(device)
        self.Gp.to(device)
        self.Gn.to(device)

 
    def write_embedding(self,
                        loader_pair: Union[DataLoader, tuple[DataLoader, DataLoader]], step:Optional[int]=None):
        if isinstance(loader_pair, DataLoader):
            loader = loader_pair
        else:
            loader = zip(*loader_pair)
        for i, (x,y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device) 
            log_dict = self.calc_embedding(x,y)
        if self.writer:                
            for k, (mat, img) in log_dict["embed"].items():
                if step is not None:
                    self.writer.add_embedding(
                        mat, label_img=img,  tag=k, global_step=step
                    )
                else:
                    self.writer.add_embedding(
                        mat, label_img=img,  tag=k
                    )

    # This function is task-specific
    def evaluation(self,
                        loader_pair: Union[DataLoader, tuple[DataLoader, DataLoader]], step:Optional[int]=None)->dict:
        if isinstance(loader_pair, DataLoader):
            loader = loader_pair
        else:
            loader = zip(*loader_pair)
        accumulated_eval = defaultdict(float)
        cnt =0
        for i, (x,y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device)

            if self.color_random:
                rand_x=torch.rand(x.size(dim=0), 3,1,1, device=self.device)*80./100. + 0.2
                rand_y=torch.rand(y.size(dim=0), 3,1,1, device=self.device)*80./100. + 0.2
                x = x*rand_x
                y = y*rand_y

            log_dict=self.eval_step(x,y)
            cnt = cnt + 1
            for k, v in log_dict["eval"].items():
                accumulated_eval[k] += v

        for k in accumulated_eval:
            accumulated_eval[k] /= float(cnt)
            print(str(k)+": " + str(accumulated_eval[k]))
        
        if self.writer:
            self.writer.add_text("0:color_eval", "x_F0x: " + str(accumulated_eval["color_x_F0x"]) 
            + ", x_F1x: " +str(accumulated_eval["color_x_F1x"])
            + ", x_y: " + str(accumulated_eval["color_x_y"]), global_step=0,
            )

            self.writer.add_text("1:shape_eval", "x_F0x: " + str(accumulated_eval["shape_x_F0x"]) 
            + ", x_F1x: " +str(accumulated_eval["shape_x_F1x"])
            + ", x_y: " + str(accumulated_eval["shape_x_y"]), global_step=0,
            )
        else:
            print("writer_error")
        return accumulated_eval

    # This function is task specific 
    def eval_step(self, x: torch.Tensor, y: torch.Tensor) ->dict: 
        
        batch_size = x.size(dim=0)

        if self.add_ch:
            z = torch.zeros([batch_size, self.Gp.in_ch-3, 32, 32], device=x.device)
            _x = torch.cat((x,z), dim=1)
            _y = torch.cat((y,z), dim=1)
            _latent_x0, _latent_x1 = self.Gp (_x)
            _latent_y0, _latent_y1 = self.Gp (_y)
        else:
            _latent_x0, _latent_x1 = self.Gp (x)
            _latent_y0, _latent_y1 = self.Gp (y)

        with torch.no_grad():
            _input_to_F0_x = torch.cat((_latent_y0, _latent_x1), dim=1)    # (lambda0_x,0)    
            _pattern_F0x, N_F0x = self.Gn(_input_to_F0_x)
            _input_to_F0_y = torch.cat((_latent_x0, _latent_y1), dim=1)  # (lambda0_y, 0) 
            _pattern_F1x, N_F1x = self.Gn(_input_to_F0_y)

            if self.add_ch:
                pattern_F0x = torch.cat((_pattern_F0x, N_F0x), dim=1)
                pattern_F1x = torch.cat((_pattern_F1x, N_F1x), dim=1)
                _latent_F0x0, _latent_F0x1 = self.Gp(pattern_F0x)
                _latent_F1x0, _latent_F1x1 = self.Gp(pattern_F1x)
            else:
                _latent_F0x0, _latent_F0x1 = self.Gp(_pattern_F0x)
                _latent_F1x0, _latent_F1x1 = self.Gp(_pattern_F1x)

            _input_to_F1_F0x = torch.cat((_latent_F0x0, _latent_y1), dim=1)
            _pattern_F0F1x, N_F0F1x  = self.Gn(_input_to_F1_F0x)
            _input_to_F0_F1x = torch.cat((_latent_y0, _latent_F1x1), dim=1)
            _pattern_F1F0x, N_F1F0x  = self.Gn(_input_to_F0_F1x)

            latent_loss0 = self.new_latent_loss_fn(_latent_x0, _latent_y0, _latent_F0x0)
            latent_loss1 = self.new_latent_loss_fn(_latent_x1, _latent_y1, _latent_F1x1)
            lambda0_mean = (_latent_x0 - _latent_y0).square().sum(dim=1).mean()
            lambda1_mean = (_latent_x1 - _latent_y1).square().sum(dim=1).mean()
            commute_loss = (_pattern_F0F1x - _pattern_F1F0x).square().mean()

            eval_color_x_F0x = eval.color_invariance(x, _pattern_F0x, 0.1)
            eval_color_x_F1x = eval.color_invariance(x, _pattern_F1x, 0.1)
            eval_color_x_y = eval.color_invariance(x,y, 0.1)

            eval_shape_x_F0x = eval.shape_invariance(x, _pattern_F0x, 0.1)
            eval_shape_x_F1x = eval.shape_invariance(x, _pattern_F1x, 0.1)
            eval_shape_x_y = eval.shape_invariance(x, y, 0.1)

            eval_pattern = torch.minimum(
                        torch.minimum ( (x - _pattern_F0x).square().mean()
                        , (x- _pattern_F1x).square().mean() )
                        ,torch.minimum( (y- _pattern_F0x).square().mean()
                        , (y-  _pattern_F1x).square().mean() )
                        )

        return {
            "eval": {
                "color_x_F0x": (eval_color_x_F0x),
                "color_x_F1x": (eval_color_x_F1x),
                "color_x_y": (eval_color_x_y),
                "shape_x_F0x": (eval_shape_x_F0x),
                "shape_x_F1x": (eval_shape_x_F1x),
                "shape_x_y": (eval_shape_x_y),
                "eval_pattern": float(eval_pattern),
                "latent_loss0": float(latent_loss0),
                "latent_loss1": float(latent_loss1),
                "lambda0": float(lambda0_mean),
                "lambda1": float(lambda1_mean),
                "commute_loss": float(commute_loss),
            },
        }

    def calc_embedding(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        
        size = x.size(dim=0)
        if size>16:
            x0, x1 = torch.split(x, (16, size-16), dim=0)
            x = torch.tile(x0, (int(size/16),1,1,1))

        if self.color_random:
            rand_x=torch.rand(x.size(dim=0), 3,1,1, device=self.device)*80./100. + 0.2
            x,index = x.max(dim=1,keepdim=True)
            x= x*rand_x

        # Background Color Change
        #alpha = torch.where(x.sum(dim=1,keepdim=True)>0.1, torch.ones(x.size()[0],1, x.size()[2],x.size()[3]).to(x.device), torch.zeros(x.size()[0],1,x.size()[2],x.size()[3]).to(x.device))
        _x = torch.where( x.sum(dim=1,keepdim=True)>0.1 ,x, torch.ones_like(x))
        #_x = torch.cat((_x,alpha), dim=1)

        batch_size = x.size(dim=0)
        if self.add_ch:
            z = torch.zeros([batch_size, self.Gp.in_ch-3, 32,32], device=x.device)
            _x2 = torch.cat((x, z), dim=1)
            _latent_x0, _latent_x1 = self.Gp (_x2)
        else:
            _latent_x0, _latent_x1 = self.Gp(x)

        return {
            "embed": {
                "space0": (_latent_x0, _x),
                "space1": (_latent_x1, _x),
            },
        }


    def latent_loss_fn(self, x:torch.Tensor, y:torch.Tensor) -> float:
        ret = (x-y).square().sum(dim=1).mean()
        return ret

    def new_latent_loss_fn(self, x0:torch.Tensor, y0:torch.Tensor, _y0:torch.Tensor) ->float:
        c =  (_y0 - y0).square().sum(dim=1) 
        d =  (_y0 - x0).square().sum(dim=1) 
        loss = c/(c+d+1e-8)
        out = (-torch.log((1.-loss) +1e-8)).mean() 
        return out

    def train(self,
              loader_pair: Union[DataLoader, tuple[DataLoader, DataLoader, DataLoader]],
              epoch: int):
        if self.writer:
            self.writer.add_text('log', 'logdir is '+str(self.logdir))
            self.writer.add_text('bsize', 'batch size is ' + str(self.batch_size))
            self.writer.add_text('orth', 'orth is '+ str(self.orth))

        for e in trange(epoch+1, ascii=True, position=0):
            self.step = e 
            accumulated_loss = defaultdict(float)
            injec_loss = defaultdict(float)
            if isinstance(loader_pair, DataLoader):
                loader = loader_pair
                total = len(loader_pair)
            else:
                loader = zip(*loader_pair)
                total = len(loader_pair[0])

            if e%2==0:
                eval=True
            else:
                eval=False
            cnt =0

            for i, (x, y) in tqdm(enumerate(loader),
                                  ascii=True,
                                  leave=False,
                                  position=1,
                                  desc=f"Epoch{e}",
                                  total=total):

                x = x.to(self.device)
                y = y.to(self.device)

                if eval:
                    if i==0:
                        log_dict_e = self.eval_step(x,y)

                if self.color_random:
                    rand_x=torch.rand(x.size(dim=0), 3,1,1, device=self.device)*80./100. + 0.2
                    rand_y=torch.rand(y.size(dim=0), 3,1,1, device=self.device)*80./100. + 0.2
                    x = x*rand_x
                    y = y*rand_y

                log_dict=self.train_step(x,y)
                cnt = cnt + 1

                for k, v in log_dict["loss"].items():
                    accumulated_loss[k] += v
                for k, v in log_dict["injec_loss"].items():
                    injec_loss[k] += v

            for k in accumulated_loss:
                accumulated_loss[k] /= float(cnt)
            for k in injec_loss:
                injec_loss[k] /= float(cnt)

            if self.writer:
                # scalars
                for k, v in accumulated_loss.items():
                    self.writer.add_scalar(f"loss/{k}", v, global_step=e)
                
                self.writer.add_scalars(f"injectivity/injec_loss", injec_loss, global_step=e)
                self.writer.add_scalars(f"eval/eval", log_dict["eval"], global_step=e)
                self.writer.add_scalars(f"injectivity/injectivity", log_dict["injectivity"], global_step=e)


                if eval:
                    self.writer.add_scalars(f"eval/invariance",log_dict_e["eval"], global_step=e )

                # images
                if  e%50==0 or (e<100 and e%10==0):
                    for k, v in log_dict["image"].items():
                        # interleave
                        img_size = v[0].size()[1:]
                        img = torch.stack(v, dim=1).view(-1, *img_size)
                        img_grid = vutils.make_grid(img.abs().clamp(0., 1.),
                                                    nrow=len(v),
                                                    pad_value=.5)
                        self.writer.add_image(f"images/{k}",
                                            img_grid,                                        
                                            global_step=e)
                    
            if self.logdir:
                self.save(f"model.pt")

    def Gn0(self, latent_0, latent_1):
        input_to_Gn=torch.cat((latent_0, latent_1), dim=1)
        return self.Gn(input_to_Gn)



    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:

        batch_size = x.size(dim=0)
        if self.add_ch:
            zero = torch.zeros([batch_size, self.Gp.in_ch-3, 32,32], device=x.device)
            _x = torch.cat((x, zero), dim=1)
            _y = torch.cat((y, zero), dim=1)
            _latent_x0, _latent_x1 = self.Gp (_x)
            _latent_y0, _latent_y1 = self.Gp (_y)
        else:
            _latent_x0, _latent_x1 = self.Gp (x)
            _latent_y0, _latent_y1 = self.Gp (y)

        # F0*X
        _pattern_F0x, N_F0x = self.Gn0(_latent_y0, _latent_x1)

        # F0*Y
        _pattern_F1x, N_F1x = self.Gn0(_latent_x0, _latent_y1)

        if self.add_ch:
            pattern_F0x =torch.cat((_pattern_F0x, N_F0x), dim=1)
            pattern_F1x =torch.cat((_pattern_F1x, N_F1x), dim=1)
            _latent_F0x0 , _latent_F0x1  = self.Gp(pattern_F0x)
            _latent_F1x0 , _latent_F1x1  = self.Gp(pattern_F1x)
        else:
            _latent_F0x0 , _latent_F0x1  = self.Gp(_pattern_F0x)
            _latent_F1x0 , _latent_F1x1  = self.Gp(_pattern_F1x)

        # F1*F0*X
        _pattern_F1F0x, N_F1F0x = self.Gn0(_latent_F0x0, _latent_y1)

        # F0*F1*X
        _pattern_F0F1x, N_F0F1x = self.Gn0(_latent_y0, _latent_F1x1)

        _pattern_y, N_y = self.Gn0(_latent_y0, _latent_y1)

        with torch.no_grad():
            latent_loss_F0x0 = self.latent_loss_fn(_latent_F0x0, _latent_y0)
            latent_loss_F0x1 = self.latent_loss_fn(_latent_F0x1, _latent_x1)
            latent_loss_F1x0 = self.latent_loss_fn(_latent_F1x0, _latent_x0)
            latent_loss_F1x1 = self.latent_loss_fn(_latent_F1x1, _latent_y1)
            new_latent_loss1 = self.new_latent_loss_fn(_latent_x0, _latent_y0, _latent_F0x0)
            new_latent_loss2 = self.new_latent_loss_fn(_latent_x1, _latent_y1, _latent_F1x1)

        
        trans_pattern_loss =  (y - _pattern_y).square().mean() 
        commute_loss = (_pattern_F1F0x - _pattern_F0F1x).square().mean()
        loss = trans_pattern_loss 
        if self.commute>0:
            loss += commute_loss * self.commute 

        self.optimizer.zero_grad()   
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            eval_pattern = torch.minimum(
                            torch.minimum (self.pattern_loss_fn(x, _pattern_F0x)
                            , self.pattern_loss_fn(x, _pattern_F1x) )
                            ,torch.minimum( self.pattern_loss_fn(y, _pattern_F0x)
                            , self.pattern_loss_fn(y, _pattern_F1x) )
                            )

        loss_str = "total_loss: " + str(self.learning_rate) 
        commute_loss_str ="commute_loss: " + str(self.commute)

        return {
            "loss": {
                loss_str: loss,
                commute_loss_str:commute_loss,
            },
            "eval":{
                "eval_pattern": eval_pattern,
            },
            "injectivity":{
                "latent0": new_latent_loss1,
                "latent1": new_latent_loss2,
            },
            "injec_loss":{
                "y0' and y0": latent_loss_F0x0,
                "x0' and x0": latent_loss_F1x0,
                "x1' and x1": latent_loss_F0x1,
                "y1' and y1": latent_loss_F1x1,
            },
            "image": {
                "trans_x2y": [x, _pattern_F0x, _pattern_F1x,
                          _pattern_F1F0x, _pattern_F0F1x, _pattern_y, y],
            },
        }

