import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiVAE(nn.Module):
    def __init__(self, p_dims, dropout = 0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]
        
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([
                                        nn.Linear(d_in, d_out) 
                                        for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
                                    ])
        self.p_layers = nn.ModuleList([
                                        nn.Linear(d_in, d_out) 
                                        for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
                                        ])
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        
    def init_weights(self):
        
        for layer in self.q_layers:
            fan_out, fan_in = layer.weight.size()
            std = np.sqrt(2.0 / (fan_out + fan_in))
            layer.weight.data.normal_(0.0, std=std)
            layer.bias.data.normal_(0.0, std=0.001)

        for layer in self.p_layers:
            fan_out, fan_in = layer.weight.size()
            std = np.sqrt(2.0 / (fan_out + fan_in))
            layer.weight.data.normal_(0.0, std=std)
            layer.bias.data.normal_(0.0, std=0.001)
            
    def encode(self, x):
        h = F.normalize(x)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]

        return mu, logvar
    
    def reparameterize(self, mu, logvar):  
        #making z a standard normal distribution
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  #sampled from a standard normal distribution
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def forward(self, x, anneal):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)

        bce = -torch.mean(torch.sum(F.log_softmax(output, 1) * x, -1))
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return output, bce + anneal * kld  #kl annealing