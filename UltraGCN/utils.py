import torch
import os
import numpy as np
import random

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

def mat_to_tensor(mat):
    coo=mat.tocoo().astype(np.float32)
    value=torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    
    return torch.sparse.FloatTensor(indices, value, coo.shape)