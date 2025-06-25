# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:03:36 2022
here the network has one hidden layer, the neuron of this hidden layer has no 
recurrent connection, it's a feedforward neuron network
@author: NZ
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import torch.optim as optim
import numpy as np

class ReadoutLinear(nn.Module):
    r"""Eexcitatory neuron read out Linear transformation.
    
    Args:
        in_features: the activity of neuron was readed out size
        out_features: the dimension of output
    """ 
    __constant__ = ['bias', 'in_features', 'out_features']
    
    def __init__(self,in_features,out_features,bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def effective_weight(self):
        return torch.abs(self.weight)
    
    def forward(self, input):
        # weight is non-negative
        return F.linear(input,self.effective_weight(),self.bias)
class ForwardNN(nn.Module):
    """Feedforward NN
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    
    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
    """
    def __init__(self, input_size, hidden_size, dt=None, sigma_rec=0.1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt/self.tau
            self.alpha = alpha
            self.oneminusalpha = 1-alpha
            ## maybe need add noise here
            self._sigma_rec = np.sqrt(2*alpha)*sigma_rec
            self.input2h = nn.Linear(input_size,hidden_size)
            
    def init_hidden(self, inputs):
        batch_size = inputs.shape[1]
        return(torch.zeros(batch_size,self.hidden_size).to(inputs.device),
               torch.zeros(batch_size, self.hidden_size).to(inputs.device))
     
    def feedandleakey(self,inputs,hidden):
        state,output = hidden
        total_input = self.input2h(inputs)
        state = state * self. oneminusalpha + total_input * self.alpha
        state += self._sigma_rec * torch.randn_like(state)
        output = F.relu(state)
        return state, output
    
class Net(nn.Module):
    """Feedforward network model.
    
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    """
    
    def __init__(self, hp, **kwargs):
        super().__init__()
        
        input_size = hp['n_input']
        hidden_size = hp['hidden_size']
        output_size = hp['n_output']
        
        self.forwardnn = ForwardNN(input_size,hidden_size,**kwargs)
        self.fc = nn.Linear(self.forwardnn.hidden_size,output_size)
    def forward(self,x):
        forwardnn_activity,_ = self.forwardnn(x)
        out = self.fc(forwardnn_activity)
        return out, forwardnn_activity
        
    
    
    