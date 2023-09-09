import argparse

import torch

from utils import seed_everything
from dataset import load_data
from model import MultiVAE
from train import run

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--save_dir_path", default="./model_saved", type=str)
    parser.add_argument("--seed", default=42, type=int)
    
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=500, help="batch_size")
    
    # model args
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--total_anneal_steps", type=int, default=200000)
    parser.add_argument("--anneal_cap", type=int, default=0.2)
    parser.add_argument("--dim", type=list, default=[200, 600])


    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    train_loader, dataset = load_data(args)
    model = MultiVAE(args.dim + [dataset.n_items]).to(args.device)
    run(model, train_loader, dataset, args)
    
if __name__=="__main__":
    main()