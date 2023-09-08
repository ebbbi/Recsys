import pandas as pd
import numpy as np
from collections import defaultdict
import os
from utils import mat_to_tensor

from scipy.sparse import dok_matrix

import torch
from torch.utils.data import Dataset, DataLoader


class GCNDataset(Dataset):
    def __init__(self, dataset, negative_num):
        users = []
        pos_items = []
        neg_items = []
        for u in dataset.all_users: 
            items = dataset.train[u]
            users += [u] * len(items)
            pos_items.extend(items)
            for _ in range(len(items)):
                neg_items.append(np.random.choice(dataset.user_negative_itemset[u], negative_num, replace = True).reshape(1, -1))  #(len(pos_train_data[0]), neg_ratio)
        nega = np.concatenate(neg_items, axis=0)
        self.users, self.pos, self.nega = torch.tensor(users), torch.tensor(pos_items), torch.from_numpy(nega)  
        
    def __getitem__(self, index):
        return self.users[index], self.pos[index], self.nega[index]
    
    def __len__(self):
        return self.users.shape[0]
    
    
class Preprocessing():
    def __init__(self, data):
        self.data = data
        self.data["user_id"], self.user_map = pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map = pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        
        self.user_negative_itemset = self.get_user_negitemset()
        self.train, self.valid = self.split()
        self.train_mat, self.train_mask  = self.generate_mask()
        
    
    def get_user_negitemset(self):
        user_negative_itemset = defaultdict(list)
        self.user_itemset = self.data.groupby("user_id")["item_id"].apply(list).to_dict()   
        for u in self.user_itemset:
            user_negative_itemset[u] = list(set(self.all_items)-set(self.user_itemset[u]))    
        return user_negative_itemset
    
    
    def split(self):
        train = {}
        valid = {}
        
        for u, v in self.user_itemset.items():
            valitem = np.random.choice(v, 20, replace = False).tolist()
            trainitem = list(set(v)-set(valitem))
            train[u] = trainitem
            valid[u] = valitem
                
        return train, valid
    
    def generate_mask(self):
        train_users, train_items = [], []
        
        for u, v in self.train.items():
            train_users += [u]*len(v)
            train_items.extend(v)

        train_mat = dok_matrix((self.n_users, self.n_items), dtype = np.float32)
        train_mat[train_users, train_items] = 1.0
        train_mask = mat_to_tensor(train_mat).to_dense()*(-np.inf)
        train_mask = torch.nan_to_num(train_mask, nan=0.0)

        return train_mat, train_mask

def load_data(args):
    rating_df = pd.read_csv(os.path.join(args.data_dir, 'ratings.csv'))
    rating_df = rating_df.sort_values(["user", "timestamp"])
    dataset = Preprocessing(rating_df)
    train_data = GCNDataset(dataset, args.negative_num)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    
    return train_loader, dataset
