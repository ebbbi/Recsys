import pandas as pd
import numpy as np
from collections import defaultdict
import os

import torch
from torch.utils.data import Dataset, DataLoader

class MFDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.users = dataset.all_users
        self.posset = dataset.train
        user, item = [], []
        
        for u, v in self.posset.items():
            user+=[u]*len(v)
            item.extend(v)

        self.user, self.item = torch.tensor(user), torch.tensor(item)
        
    def __getitem__(self, index):
        return self.user[index], self.item[index]
    
    def __len__(self):
        return self.users.shape[0]


class Preprocessing():
    def __init__(self, data):
        self.data = data
        self.data["user_id"], self.user_map=pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map=pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        
        self.negative_itemset = self.get_user_negitemset()        
        self.train, self.valid = self.split()
        
    
    def split(self):
        train = {}
        valid = {}
        np.random.seed(42)
        for u, v in self.user_itemset.items():
            valitem = np.random.choice(v[int(len(v)*0.2):], 10, replace=False).tolist()
            trainitem = list(set(v)-set(valitem))
            train[u] = trainitem
            valid[u] = valitem
    
        return train, valid
    
    def get_user_negitemset(self):
        user_negative_itemset = defaultdict(list)
        self.user_itemset = self.data.groupby("user_id")["item_id"].apply(list).to_dict()
        for u in self.user_itemset:
            user_negative_itemset[u] = list(set(self.all_items)-set(self.user_itemset[u]))     
            
        return user_negative_itemset
    
    def negative_sampling(self, users):
        nega_user, nega_item=[], []
        for u in users:
            nega_item += np.random.choice(self.negative_itemset[u], 3, replace = False).tolist()   
            nega_user += [u]*3
            
        return torch.tensor(nega_user), torch.tensor(nega_item)

def load_data(args):
    data = pd.read_csv(os.path.join(args.data_dir, 'ratings.csv'))
    data = data.sort_values(["user", "timestamp"]) 
      
    dataset = Preprocessing(data)
    
    train_data = MFDataset(dataset)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    
    return train_loader, dataset 