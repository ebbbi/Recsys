import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix

class Preprocessing():
    def __init__(self, data):
        self.data=data
        self.data["user_id"], self.user_map = pd.factorize(self.data["user"])
        self.data["item_id"], self.item_map = pd.factorize(self.data["item"])
        self.n_users = max(self.data["user_id"])+1
        self.n_items = max(self.data["item_id"])+1
        self.all_items = self.data['item_id'].unique()
        self.all_users = self.data['user_id'].unique()
        self.user_itemset = self.data.groupby("user_id")["item_id"].apply(list).to_dict()
        self.train, self.valid = self.split()
        self.mat=self.make_matrix()
            
    def split(self):
        train = {}
        valid = {}
        for u, v in self.user_itemset.items():
            valitem = v[-2:]
            valitem += np.random.choice(v[int(len(v)*0.2):], 8, replace = False).tolist()
            #valitem=np.random.choice(v[int(len(v)*0.4):], 10, replace = False).tolist()
            trainitem = list(set(v)-set(valitem))
            train[u] = trainitem
            valid[u] = valitem
            
        return train, valid
    
    def make_matrix(self):
        R = dok_matrix((self.n_users, self.n_items), dtype = np.float32)
        for u, v in self.train.items():
            R[u, v] = 1.0
            
        return self.mat_to_tensor(R.tocsr())
    
    def mat_to_tensor(self, mat):
        coo = mat.tocoo().astype(np.float32)
        value = torch.FloatTensor(coo.data)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))

        return torch.sparse.FloatTensor(indices, value, coo.shape).to_dense()
    
def load_data(args):
    rating_df = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    rating_df = rating_df.sort_values(["user", "time"])
    dataset = Preprocessing(rating_df)
        
    data={
        "train_data":dataset.mat,
        "validset":dataset.valid,
        "dataset":dataset
    }
    
    return data