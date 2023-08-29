import numpy as np
import os 

import torch
from torch.optim import Adam

from utils import BPRLoss, recall_batch

def run(model, train_loader, dataset, args):
    model = model.to(args.device)
    bprloss = BPRLoss()
    optimizer = Adam(model.parameters(), lr = args.lr)
    best_recall = 0
    
    print("START TRAIN")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for user, item in train_loader:

            nega_user, nega_item = dataset.negative_sampling(user.numpy())
            u = torch.cat([user, nega_user]).to(args.device)
            i = torch.cat([item, nega_item]).to(args.device)
            
            optimizer.zero_grad()
            pred = model(u, i)   

            pos, nega = torch.split(pred, [len(user), len(user)*3])
            pos = pos.view(-1, 1).repeat(1, 3).view(-1)
            loss = bprloss(pos, nega)  

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        recall = 0.0
        with torch.no_grad():
            pred = model.user_embedding.weight @ model.item_embedding.weight.T
            pred = pred.to("cpu").detach().numpy()
            for u, v in dataset.train.items():
                pred[u, v] = -np.inf
            top = np.argpartition(pred, -10)[:, -10:]
            recall = recall_batch(dataset.valid, top)

        if recall > best_recall:
            best_recall = recall
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model, f"{args.save_dir}/best_model.pt")
            torch.save(model.state_dict(), f"{args.save_dir}/state_dict.pt")

        print(f'epoch: {epoch}, loss: {total_loss/len(train_loader)}, recall: {recall}')      
        
    print(best_recall)