import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class UltraGCN(nn.Module):
    def __init__(self, mat, n_users, n_items, args):
        super(UltraGCN, self).__init__()
        self.device = args.device 
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = args.latent_dim
        
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.w4 = args.w4
        self.initial_w = args.initial_w
        self.negative_w = args.negative_w
        self.gamma = args.gamma
        self.lambda_w = args.lambda_w
        
        self.neighbor_num = args.neighbor_num
        
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.init_weight()
        
        self.get_matrix(mat)
    
    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std = self.initial_w)
        nn.init.normal_(self.item_embedding.weight, std = self.initial_w)
        
    def get_matrix(self, mat):
        item = np.sum(mat, axis=0).reshape(-1)
        user = np.sum(mat, axis=1).reshape(-1)
        i_beta = (1 / np.sqrt(item + 1)).reshape(1, -1)
        u_beta = (np.sqrt(user + 1) / user).reshape(-1, 1)
        self.constraint_mat = {"beta_uD" : torch.from_numpy(u_beta).reshape(-1), 
                               "beta_iD" : torch.from_numpy(i_beta).reshape(-1)}
        
        self.il_neighbor_mat, self.il_constraint_mat = self.get_il_constarint_mat(mat, self.neighbor_num)
    
    def get_il_constarint_mat(self, mat, neighbor_num, ii_diagonal_zero=False):
        n_items = mat.shape[1]
        A = mat.T @ mat
        ng_mat = torch.zeros((n_items, neighbor_num))
        sim_mat = torch.zeros((n_items, neighbor_num))
        
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0
        
        item = np.sum(A, axis=0).reshape(-1)
        user = np.sum(A, axis=1).reshape(-1)
        i_beta = (1 / np.sqrt(item + 1)).reshape(1, -1)
        u_beta = (np.sqrt(user + 1) / user).reshape(-1, 1)
        all_il_constraint_mat = torch.from_numpy(u_beta.dot(i_beta)) #dot, mul, @ 차이
        for i in range(n_items):
            row = all_il_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            sim, idx = torch.topk(row, neighbor_num)
            ng_mat[i] = idx
            sim_mat[i] = sim
        return ng_mat.long(), sim_mat.float()

    
    def get_omega(self, user, pos, neg):
        if self.w2>0:
            pos_w = torch.mul(self.constraint_mat["beta_uD"][user], self.constraint_mat["beta_iD"][pos]).to(self.device)  
            pos_w = self.w1 + self.w2 * pos_w   
        else:
            pos_w = self.w1 * torch.ones(len(pos)).to(self.device) 
            
        if self.w4>0:
            neg_w = torch.mul(torch.repeat_interleave(self.constraint_mat["beta_uD"][user], neg.size(1)), self.constraint_mat["beta_iD"][neg.flatten()]).to(self.device)
            neg_w = self.w3 + self.w4 * neg_w
        else:
            neg_w = self.w3 + torch.ones(neg.size(0) * neg.size(1)).to(self.device) 
        
        return torch.cat((pos_w, neg_w))

    def bceloss(self, user, pos, neg, omega):
        user_embed = self.user_embedding(user)
        pos_embed = self.item_embedding(pos)
        neg_embed = self.item_embedding(neg)
        
        pos = (user_embed * pos_embed).sum(dim=-1)
        neg = (user_embed.unsqueeze(1) * neg_embed).sum(dim=-1)
        
        # diff weight for calculating loss
        pos_loss = F.binary_cross_entropy_with_logits(pos, 
                                        torch.ones(pos.size()).to(self.device), 
                                        weight=omega[:len(pos)], 
                                        reduction='none')
        
        neg_loss = F.binary_cross_entropy_with_logits(neg, 
                                        torch.zeros(neg.size()).to(self.device), 
                                        weight=omega[len(pos):].view(neg.size()), 
                                        reduction='none').mean(dim=-1)
        
        
        return (pos_loss + neg_loss * self.negative_w).sum()
        
        
    def loss_I(self, user, pos):
        neighbor = self.il_neighbor_mat[pos].to(self.device)
        neighbor_embed = self.item_embedding(neighbor)
        user_embed = self.user_embedding(user).unsqueeze(1)
        
        loss = -self.il_neighbor_mat[pos].to(self.device) * (user_embed * neighbor_embed).sum(dim=-1).sigmoid().log()
        
        return loss.sum()
    
    def forward(self, user, pos, neg):
        omega = self.get_omega(user, pos, neg)
        loss = self.bceloss(user, pos, neg, omega)
        norm = 0.0
        for p in self.parameters():
            norm += torch.sum(p.pow(2))        
        loss += (norm/2) * self.gamma
        loss += self.loss_I(user, pos) * self.lambda_w
        
        return loss