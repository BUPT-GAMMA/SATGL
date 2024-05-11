import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 activation="relu",
                 num_layer=3,
                 dropout_ratio=0):    
        super(MLP, self).__init__()
        assert(num_layer >= 1)
        self.num_layer = num_layer
        
        self.linear_list = nn.ModuleList()
        if num_layer == 1:
            self.linear_list.append(nn.Linear(input_size, output_size))
        else:
            self.linear_list.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_layer - 2):
                self.linear_list.append(nn.Linear(hidden_size, hidden_size))
            self.linear_list.append(nn.Linear(hidden_size, output_size))
        if activation is None:
            self.activation = lambda x: x
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError("activation function not suported")

    def forward(self, x):
        h = x
        for layer in self.linear_list[:-1]:
            h = self.activation(layer(h))
        h = self.linear_list[-1](h)
        return h