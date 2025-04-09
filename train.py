import torch


from diffusion import create_diffusion
from model import DiffusionModel
# from train_utils import requires_grad, update_ema

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from diffusion import create_diffusion
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob

import time
import argparse
import logging
from datasets import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
import random
import anndata

import sys






def load_model_and_datasets(args):
    train_set = Her2st(args)
    args = train_set.get_args()
    model = DiffusionModel(
        input_size=args.input_gene_size,
        depth=args.DiT_num_blocks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=args.cond_size,
    ).to(args.device)
    print(f"Dataset contains {len(train_set):,} images ({args.data_path})")
    return train_set, model, args

def save_checkpoint(model,optimizer,train_steps,args):
    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict()
    }
    checkpoint_path = f"{args.checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def train(model,train_loader,args):
    device = args.device
    train_loader = train_loader
    args = args
    model = model.to(device)
    path_checkpoints = './her2st_results/runs/004/checkpoints/0001450.pt'
    checkpoint = torch.load(path_checkpoints)
    model.load_state_dict(checkpoint['model'])

    diffusion = create_diffusion(timestep_respacing="")
    optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=args.lr, weight_decay=0)
    optimizer.load_state_dict(checkpoint['opt'])
    epoch_checkpoint = 1451
    model.train()

    avg_loss = 0
    for epoch in range(epoch_checkpoint,args.total_epochs):
        total_loss = 0
        tqdm_train = tqdm(train_loader, total=len(train_loader))
        for gene_exp,local_ebd,neighbor_ebd,global_ebd,pos,neighbor_pos,global_pos in tqdm_train:
            x = gene_exp.unsqueeze(1).to(device)  # (N, 1, NumGene)
            x = x.float()
            local_ebd = local_ebd.to(device)
            neighbor_ebd = neighbor_ebd.to(device)
            global_ebd = global_ebd.to(device)
            neighbor_pos = neighbor_pos.to(device)
            global_pos = global_pos.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (gene_exp.size(0),), device=x.device)
            model_kwargs = dict(local_ebd=local_ebd,
                                neighbor_ebd=neighbor_ebd,
                                global_ebd = global_ebd,
                                pos = pos,
                                neighbor_pos = neighbor_pos,
                                global_pos = global_pos)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            loss.backward()
            total_loss +=loss.item()
            optimizer.step()
            tqdm_train.set_postfix(train_loss=loss.item(), lr=args.lr, epoch=epoch,avg_loss = avg_loss)
        avg_loss = total_loss/len(train_loader)
        if epoch % args.ckpt_every == 0 and epoch!=0:
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            train_steps=epoch,
                            args=args)


def prepare_dataloader(args, dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def main(input_args):
    device = input_args.device
    torch.cuda.set_device(device)
    print("mkdir & set up logger...")
    # mkdir for logs and checkpoints
    os.makedirs(input_args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{input_args.results_dir}/*"))
    input_args.experiment_dir = f"{input_args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
    input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(input_args.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)  # Store sampling results

    # set up training objects

    dataset, model, args = load_model_and_datasets(input_args)

    train_loader = prepare_dataloader(args, dataset, input_args.batch_size)
    train(model = model,train_loader = train_loader,args = args)

    print(f"Starting...")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument("--expr_name", type=str, default="her2st")
    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/her2st/", help="Dataset path")
    parser.add_argument("--results_dir", type=str, default="./her2st_results/runs/", help="Path to hold runs")
    parser.add_argument("--slide_out", type=str, default="SPA152",
                        help="Test slide ID. Multiple slides separated by comma.")
    parser.add_argument("--slidename_list", type=str, default="all_slide_lst.txt",
                        help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list", type=str, default="selected_gene_list.txt", help="Selected gene list")
    parser.add_argument("--mode", type=str, default="train", help="Running mode (train/test)")
    # model related arguments

    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")
    parser.add_argument("--device", type=str, default='cuda:0', help="Gpu")
    # training related arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--ckpt_every", type=int, default=50, help="Number of epoch to save checkpoints.")


    input_args = parser.parse_args()

    main(input_args)
