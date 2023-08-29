import argparse

import torch

from utils import seed_everything
from dataset import load_data
from model import EASE

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--save_dir_path", default="./model_saved", type=str)
    parser.add_argument("--seed", default=42, type=int)

    # model args
    parser.add_argument("--reg", type=float, default=100, help="regularization strength")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    train_data, validset = load_data(args)
    model = EASE(args.reg)
    model.fit(train_data)
    model.evaluate(train_data, validset)
    
if __name__=="__main__":
    main()