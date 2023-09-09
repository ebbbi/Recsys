import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

class VAEDataSet(Dataset):
    def __init__(self, dataset):
        self.users = torch.LongTensor(dataset.all_users)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx): 
        return self.users[idx]

class Preprocessing():
    def __init__(self, data):
        self.data = data
        self.data["user_id"], self.user_map = pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map = pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        self.user_itemset = self.data.groupby("user_id")["item_id"].apply(list).to_dict()
        self.train, self.valid = self.split()

    def split(self):
        train = {}
        valid = {}
        for u, v in self.user_itemset.items():
            valitem = np.random.choice(v, 10, replace = False).tolist()  
            trainitem = list(set(v) - set(valitem))
            train[u] = trainitem
            valid[u] = valitem
            
        return train, valid
    
    def make_matrix(self, user):
        mat = torch.zeros(size = (user.size(0), self.n_items))
        for idx, u in enumerate(user):
            mat[idx, self.train[u.item()]] = 1
            
        return mat
    
def load_data(args):
    rating_df = pd.read_csv(os.path.join(args.data_dir, 'ratings.csv'))
    rating_df = rating_df.sort_values(["user", "timestamp"])
    dataset = Preprocessing(rating_df)
    trainset = VAEDataSet(dataset)
    train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    
    return train_loader, dataset
