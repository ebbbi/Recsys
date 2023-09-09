import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiDAE(nn.Module):
    def __init__(self, p_dims, dropout = 0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]
        
        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([
                                        nn.Linear(d_in, d_out) 
                                        for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
                                    ])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        
    def init_weights(self):
        
        for layer in self.layers:
            fan_out, fan_in = layer.weight.size()
            std = np.sqrt(2.0 / (fan_out + fan_in))
            layer.weight.data.normal_(0.0, std=std)
            layer.bias.data.normal_(0.0, std=0.001)

            
    def forward(self, x):
        h = F.normalize(x)
        h = self.drop(h)
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
                
        bce_loss = -(F.log_softmax(h, 1) * x).sum(dim = -1).mean()

        return h, bce_loss