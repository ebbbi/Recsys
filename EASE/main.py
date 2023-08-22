import argparse
import os

import numpy as np
import pandas as pd
import torch

from utils import seed_everything
from dataset import load_data
from model import LightGCN
from train import run, inference

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--save_dir_path", default="./model_saved", type=str)
    parser.add_argument("--seed", default=42, type=int)

    # model args
    parser.add_argument("--reg", type=float, default=100, help="regularization strength")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    data = load_data(args)
    train_data,  validset = data["train_data"], data["validset"]
    model = EASE(args.reg)
    model.fit(train_data)
    model.evaluate(train_data, validset)
    
if __name__=="__main__":
    main()