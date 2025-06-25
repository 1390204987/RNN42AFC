# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:19:32 2021

@author: NaN
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import numpy as np

class EIRecLinear(nn.Module):
    """Recurrent E-I Linear transformation.
    
    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self,hidden_size,e_prop,bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eprop = e_prop
        self.e_size = int(e_prop*hidden_size)
        self.i_size = hidden_size-self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        mask = np.tile([1]*self.e_size+[-1]*self.i_size,(hidden_size,1))
        np.fill_diagonal(mask,0)
        self.mask = torch.tensor(mask,dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
                
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        self.weight.data[:,:self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)
            
    def effective_weight(self):
        return torch.abs(self.weight)*self.mask
    
    def forward(self, input):
        return F.linear(input, self.effective_weight(), self.bias)