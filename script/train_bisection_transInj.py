import argparse
import os
import shutil
from turtle import color
from typing import Iterable, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.main_model import BisectionTrans
from dataset.colorfont import ColorFontPairDataset, NPZImagesDataset
import utils.json_model as jm
import numpy as np
import random
from script.init_modelInj import (init_modelInj)
import math

def torch_fix_seed(seed: int):
    print("seed is " + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    # Pytorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def main(args):

    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)
    if len(args.label_dim) == 1:
        args.label_dim = [args.label_dim[0], args.label_dim[0]]
    elif len(args.label_dim) > 2:
        raise ValueError

    print (args.label_dim)

    if args.seed is not None:
        torch_fix_seed(args.seed)

    model = init_modelInj(args.label_dim, p_ch=args.p_ch, n_ch=args.n_ch, orth=args.orth, add_ch=args.add_ch, gp_bias=args.gp_bias)
    model.log_config(args.logdir)

    model.Gn_config(orth=args.orth)

    # device
    if torch.cuda.is_available():
        if args.gpu < 0:
            # logger.info("CPU will be used.")
            device = torch.device("cpu")
        elif args.gpu >= torch.cuda.device_count():
            # logger.warn(f"Specified GPU ID {gpu_id} is invalid. "
            #             "CPU will be used instead.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cpu")
        # logger.warn("No CUDA device is available. CPU will be used.")
    model.to(device)

    optimizer = optim.RAdam(
        [
        {"params": model.Gp.Gp0.parameters()},
        {"params": model.Gp.Gp1.parameters()},
        {"params": filter(lambda p: p.requires_grad, model.Gn.parameters()) },
        ],
        args.learning_rate,
        )

    model.train_config(
        optimizer=optimizer,
        commute=args.commute,
        learning_rate = args.learning_rate,
        aux_rate =args.aux_rate,
        batch_size = args.batch_size,
        color_random=args.color_random,
        add_ch = args.add_ch
        )
           
    dataset = NPZImagesDataset("color_font_all.npz")
    loaders = (DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),
               DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
    
    model.train(loaders, args.epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--logdir", "-l")
    parser.add_argument("--seed", "-sd", type=int)

    parser.add_argument("--label-dim", "-ldim",type=int, nargs="*", default=32)
    parser.add_argument("--p_ch", "-pch", type=int, default=64)
    parser.add_argument("--n_ch", "-nch", type=int, default=32)
    parser.add_argument("--add_ch", "-addc", action="store_true", default=False)

    # Important parameters
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-bs", type=int, default=128)
    parser.add_argument("--epoch", "-e", type=int, default=1000)
    parser.add_argument("--commute", "-cm", type=float, default=1.0)
    parser.add_argument("--aux_rate", "-ar", type=float, default=0.0)
    parser.add_argument("--orth", "-or", action="store_true", default=False)
    parser.add_argument("--gp_bias", "-gp_b", action="store_true", default=False)

    parser.add_argument("--color_random", "-cro", action="store_false", default=True)
    parser.add_argument("--shape_split", '-ss', action="store_true", default=False)
    main(parser.parse_args())
