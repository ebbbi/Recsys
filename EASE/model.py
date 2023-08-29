import torch
import torch.nn as nn

import numpy as np
from utils import recall_at_10


class EASE(nn.Module):
    def __init__(self, reg):
        super(EASE, self).__init__()
        self.reg = reg
        
    def fit(self, X):
        G = X.T@X  
        
        diag_ind = torch.eye(G.shape[0])
        G[diag_ind == 1] += self.reg
        P = G.inverse()
        B = P/(-P.diag())
        
        B[diag_ind == 1] = 0
        self.pred = X @ B
    
    def evaluate(self, X, valid):
        recall = 0.0
        output = self.pred
        output[X == 1] = -np.inf
        
        _, idx = torch.topk(output, k=10)
        pred_items = idx.to("cpu").detach().numpy()
        
        for u, items in enumerate(pred_items):
            recall += recall_at_10(valid[u], items)

        print(f"recall : {recall / X.shape[0]}")    