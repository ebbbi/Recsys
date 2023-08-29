import torch
import os
import numpy as np
import random

import torch.nn as nn
import torch

def recall_at_10(answer, toplist):
    return len(set(answer) & set(toplist)) / len(answer)

def recall_batch(valid, pred):
    recall = 0.0
    for i, p in enumerate(pred):
        v = valid[i]
        recall += len(set(v)&set(p)) / len(v)
    return recall / pred.shape[0]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos, neg):
        return -torch.log(self.gamma+torch.sigmoid(pos-neg)).mean()
