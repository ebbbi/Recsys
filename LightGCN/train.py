import os

import torch
from torch.optim import Adam

from utils import recall_batch

def run(model, train_loader, dataset, args):
    best_recall = 0
    optimizer = Adam(model.parameters(), lr = args.lr)
    print("START TRAIN")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for user, pos, nega in train_loader:
            optimizer.zero_grad()
            
            user, pos, nega = user.to(args.device), pos.to(args.device), nega.to(args.device),
            bprloss = model(user, pos, nega)
            
            bprloss.backward()
            optimizer.step()
            total_loss += bprloss.item()
        
        model.eval()
        with torch.no_grad():
            score = torch.mm(model.user_final_embedding, model.item_final_embedding.T)
            score += dataset.train_mask.to(args.device)
            _, idx = torch.topk(score, k=10)
            pred_items = idx.to("cpu").detach().numpy()
            recall = recall_batch(dataset.valid, pred_items)
            
            if recall > best_recall:
                best_recall = recall
                if not os.path.exists(args.save_dir_path):
                    os.makedirs(args.save_dir_path)
                torch.save(model, f"{args.save_dir_path}/best_model.pt")
                torch.save(model.state_dict(), f"{args.save_dir_path}/state_dict.pt")
        
        print(f'epoch: {epoch}, loss: {total_loss/len(train_loader)}, recall: {recall}')     
        
    print(best_recall)   
