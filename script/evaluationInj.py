import argparse
import os
import shutil
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.main_model import BisectionTrans
import model.layer as L
from dataset.colorfont import ColorFontPairDataset, NPZImagesDataset
import utils.json_model as jm 
import time
import numpy as np
import random
from script.init_modelInj import (init_modelInj)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def torch_fix_seed(seed: int):
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
    if len(args.label_dim) == 1:
        args.label_dim = [args.label_dim[0], args.label_dim[0]]
    elif len(args.label_dim) > 2:
        raise ValueError

    if args.seed is not None:
        torch_fix_seed(args.seed)

    f: str="model.pt"
    prefix: str ="models"
    path_to_model = os.path.join(args.logdir, prefix, f)

    model = init_modelInj(args.label_dim ,p_ch=args.p_ch, n_ch=args.n_ch,  orth=args.orth, add_ch=args.add_ch, gp_bias=args.gp_bias)

    model.load_state_dict( torch.load(path_to_model) )        
    model.log_config(args.logdir)

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
    
    dataset = NPZImagesDataset("color_font_all.npz")
    loaders = (DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),
               DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
    
    model.config(args.color_random, args.add_ch)

    if args.write_embedding:
        model.eval_config(args.batch_size)
        model.write_embedding(loaders)

    dict=model.evaluation(loaders)
    time.sleep(2) # wait for writing log date to tensorboard

    if args.out_file is not None:
        with open(args.out_file, 'a') as f:
            if dict['shape_x_F0x'] > dict['shape_x_F1x']:
                color_invariance = dict['color_x_F1x']
            else:
                color_invariance = dict['color_x_F0x']
            if dict['color_x_F0x'] > dict['color_x_F1x']:
                shape_invariance = dict['shape_x_F1x']
            else:
                shape_invariance = dict['shape_x_F0x']

            if dict["lambda0"] > dict["lambda1"]:                 
                print(args.logdir, color_invariance, shape_invariance, (color_invariance+shape_invariance)/2., 
                dict['latent_loss0'], dict['lambda0'],  dict['latent_loss1'], dict['lambda1'], 
                dict['commute_loss'], dict['aux_loss'],
                dict['eval_pattern'], dict['color_x_F0x'], dict['color_x_F1x'], dict['shape_x_F0x'], dict['shape_x_F1x'], file=f, sep=', ')
            else:
                print(args.logdir, color_invariance, shape_invariance, (color_invariance+shape_invariance)/2., 
                dict['latent_loss1'], dict['lambda1'],  dict['latent_loss0'], dict['lambda0'], 
                dict['commute_loss'], dict['aux_loss'],
                dict['eval_pattern'], dict['color_x_F0x'], dict['color_x_F1x'], dict['shape_x_F0x'], dict['shape_x_F1x'], file=f, sep=', ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-bs", type=int, default=128)
    parser.add_argument("--logdir", "-l")
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--seed", "-sd", type=int)
    parser.add_argument("--out_file", "-of", type=str )
    parser.add_argument("--add_ch", "-addc", action="store_true", default=False)

    parser.add_argument("--label-dim", "-ldim", type=int, nargs="*", default=[32,32])
    parser.add_argument("--p_ch", "-pch", type=int, default=64)
    parser.add_argument("--n_ch", "-nch", type=int, default=32)
    parser.add_argument("--orth", "-or", action="store_true", default=False)
    parser.add_argument("--gp_bias", "-gp_b", action="store_true", default=False)

    parser.add_argument("--color_random", "-cro", action="store_false", default=True)

    parser.add_argument("--write_embedding", '-we', action="store_true", default=False)
    main(parser.parse_args())
