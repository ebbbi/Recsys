import pandas as pd
import numpy as np
import os

import torch
from torch.optim import Adam

from utils import recall_at_10


def run(model, data_loader, dataset, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    valid = dataset.valid
    update_cnt = 0
    best_recall = 0.0
    cnt = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for user in data_loader:
            mat = dataset.make_matrix(user).to(args.device)
            
            optimizer.zero_grad()
            
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_cnt / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            _, loss = model(mat, anneal)
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            update_cnt += 1
        
        model.eval()
        with torch.no_grad():
            for user in data_loader:
                mat = dataset.make_matrix(user).to(args.device)
                
                if args.total_anneal_steps > 0:
                    anneal = min(args.anneal_cap, 1. * update_cnt / args.total_anneal_steps)
                else:
                    anneal = args.anneal_cap
                
                recon, _ = model(mat, anneal)
                recon[mat == 1] = -np.inf
                _, idx = torch.topk(recon, 10)
                pred = idx.to("cpu").detach().numpy()
                
                recall = 0.0
                for u, items in zip(user, pred):
                    recall += recall_at_10(valid[u.item()], items)
                    
            recall /= mat.shape[0]        
            if recall > best_recall:
                best_recall = recall
                cnt = 0
                
                if not os.path.exists(args.save_dir_path):
                    os.makedirs(args.save_dir_path)
                torch.save(model, f"{args.save_dir_path}/best_model.pt")
                torch.save(model.state_dict(), f"{args.save_dir_path}/state_dict.pt")
                
        if cnt >= 50:
            print(best_recall)
            break
        cnt+=1
        
        print(f"epoch: {epoch}, loss: {total_loss / len(data_loader)}, recall: {recall}")        