import pandas as pd
import numpy as np

import torch
from torch.optim import Adam

from utils import BPRLoss, recall_at_10

def run(model, train_loader, valid_data, train, pp, args):
    model=model.to(args.device)
    bprloss=BPRLoss()
    optimizer=Adam(model.parameters(), lr = args.lr)
    best_recall=0
    for epoch in range(args.epochs):
        model.train()
        total_loss=0
        for user, item in train_loader:
            nega_user, nega_item=pp.negative_sampling(user.numpy())
            ux=torch.cat([user, nega_user]).to(args.device)
            ix=torch.cat([item, nega_item]).to(args.device)

            optimizer.zero_grad()

            pred=model(ux, ix)   #(user,pos)...(user,neg)*3
            pos, neg=torch.split(pred, [len(user), len(user)*3])
            pos=pos.view(-1, 1).repeat(1, 3).view(-1)  #(user, pos),(user, pos),(user, pos)
            loss=bprloss(pos, neg)  

            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        model.eval()
        recall=0
        pred_items=[]
        with torch.no_grad():
            U=pp.all_users
            I=torch.tensor(pp.all_items).to(args.device)
            for u in U:
                user=torch.tensor([u]*len(I)).to(args.device)
                pred=model(user, I)
                train_idx=train[u]
                pred = pred.to("cpu").detach().numpy()
                pred[train_idx]=-np.inf
                top=np.argpartition(pred, -10)[-10:]
                pred_items.append(top)
                recall+=recall_at_10(valid_data[u], top)

        if recall>best_recall:
            best_recall=recall
            if not os.path.exists(args.save_dir_path):
                os.makedirs(args.save_dir_path)
                torch.save(model, f"{args.save_dir_path}/best_model.pt")
                torch.save(model.state_dict(), f"{args.save_dir_path}/state_dict.pt")

        print(f'epoch: {epoch}, loss: {total_loss/len(train_loader)}, recall : {recall}')      