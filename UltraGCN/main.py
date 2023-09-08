import argparse
import torch

from utils import seed_everything
from dataset import load_data
from model import UltraGCN
from train import run

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--save_dir_path", default="./model_saved", type=str)
    parser.add_argument("--seed", default=42, type=int)
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    
    # model args
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--neighbor_num", type=int, default=10)
    parser.add_argument("--initial_w", type=int, default=0.001)
    parser.add_argument("--w1", type=int, default=1e-7)
    parser.add_argument("--w2", type=int, default=1)
    parser.add_argument("--w3", type=int, default=1e-7)
    parser.add_argument("--w4", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=1e-4)
    parser.add_argument("--lambda_w", type=int, default=1e-3)
    parser.add_argument("--negative_num", type=int, default=200)
    parser.add_argument("--negative_w", type=int, default=200)
    
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(args.seed)
    train_loader,  dataset = load_data(args)
    model = UltraGCN(dataset.train_mat, dataset.n_users, dataset.n_items, args).to(args.device)
    run(model, train_loader, dataset, args)
    
if __name__=="__main__":
    main()