import torch
from torch import cat, rand
import torch.nn as nn
import torch.nn.functional as F

# Henz's implementation of Linear filter
class LinearFilter(nn.Module):
    def __init__(self, P, C):
        super(LinearFilter, self).__init__()
        self.w_init = nn.Parameter(rand(size=(C,P,P)))
        self.b_init = nn.Parameter(rand(size=(1,P,P)))
        
        self.P = P
        self.C = C

    def forward(self, input_tensor):
        linear_filter = torch.tensor(self.w_init)
        linear_filter = F.relu(linear_filter)
        bias = torch.tensor(self.b_init)
        
        linear_filter = cat((linear_filter,linear_filter,linear_filter), 1)
        linear_filter = cat((linear_filter,linear_filter,linear_filter), 2)
        
        bias = cat((bias,bias,bias), 1)
        bias = cat((bias,bias,bias), 2)
        
        inputs_weighted = torch.mul(input_tensor, linear_filter)
        inputs_summed = torch.sum(inputs_weighted, 1, keepdim=True)
        inputs_filtered = inputs_summed + bias
        return [inputs_filtered, inputs_weighted]

# Linear filter with only weight values
class LinearFilterNoBias(nn.Module):
    def __init__(self, P, C):
        super(LinearFilterNoBias, self).__init__()
        self.w_init = nn.Parameter(rand(size=(C,P,P)))
        
        self.P = P
        self.C = C
    
    def forward(self, input_tensor):
        linear_filter = torch.tensor(self.w_init)
        linear_filter = F.relu(linear_filter)
        
        linear_filter = cat((linear_filter,linear_filter,linear_filter), 1)
        linear_filter = cat((linear_filter,linear_filter,linear_filter), 2)
        
        inputs_weighted = torch.mul(input_tensor, linear_filter)
        #inputs_filtered = torch.sum(inputs_weighted, 1, keepdim=True)
        return inputs_weighted
