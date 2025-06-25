# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:02:11 2021

only one hidden layer

@author: NaN
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
        return self.weight
        # return torch.abs(self.weight)
    
    def forward(self, input):
        # weight is non-negative
        return F.linear(input,self.effective_weight(),self.bias)
                
class EIRecLinear(nn.Module):
    """Recurrent E-I Linear transformation.
    
    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']
    
    
    def __init__(self, hidden_size, e_prop, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop*hidden_size)
        self.i_size = hidden_size-self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        # mask = np.tile([1]*self.e_size+[-1]*self.i_size,(hidden_size, 1))
        # np.fill_diagonal(mask, 0)
        mask = np.eye(hidden_size)*0.5
        self.mask = torch.tensor(mask, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        #scale E weight by E-I ratio
        self.weight.data[:,:self.e_size]/=(self.e_size/self.i_size)
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def effective_weight(self):
        return self.weight
        # return torch.abs(self.weight)*self.mask
        
    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)

class EIRNN(nn.Module):
    """E-I RNN.
    
    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).

    Args:
        input_size: Number of input neurons
        hidden_size1: Number of first layer hidden neurons
        hidden_size2: Number of second layer hidden neurons
        hidden_size3: Number of third layer hidden neurons

    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
        e_prop: float between 0 and 1, proportion of excitatory neurons
    """            
    def __init__(self, input_size, hidden_size, dt=None, e_prop=0.5, sigma_rec=0.1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size*e_prop)
        self.i_size = hidden_size - self.e_size
        
        self.num_layers = 3
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt/self.tau
            self.alpha = alpha
            self.oneminusalpha = 1-alpha
            # Recurrent noise
            self._sigma_rec = np.sqrt(2*alpha)*sigma_rec
            
            self.input2h = nn.Linear(input_size, hidden_size)
            self.h2h = EIRecLinear(hidden_size, e_prop=0.8)
            
    def init_hidden(self, inputs):
        batch_size = inputs.shape[1]
        return(torch.zeros(batch_size,self.hidden_size).to(inputs.device),
               torch.zeros(batch_size, self.hidden_size).to(inputs.device))
    
    def recurrence(self, inputs, hidden):
        """Recurrence helper."""
        state, output = hidden
        total_input = self.input2h(inputs)+self.h2h(output)
        # total_input = self.input2h(inputs)+self.h2h(state)
        state = state * self.oneminusalpha + total_input * self.alpha
        state += self._sigma_rec * torch.randn_like(state)
        output = F.relu(state)
        # output = F.softplus(state)
        return state, output
    
    def forward(self, inputs, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(inputs)
        
        output = []
        steps = range(inputs.size(0))
        
        
        for i in steps:
            hidden = self.recurrence(inputs[i],hidden)
            output.append(hidden[1])
            
        output = torch.stack(output,dim=0)
        return output, hidden
    
class Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    
    def __init__(self, hp, **kwargs):
        super().__init__()
        
        input_size = hp['n_input']
        hidden_size = hp['hidden_size1']
        output_size = hp['n_output']
        # Excitatory-inhibitory RNN first layer
        self.rnn = EIRNN(input_size,hidden_size,**kwargs)
        self.fc = nn.Linear(self.rnn.e_size,output_size)
        # self.fc = ReadoutLinear(self.rnn.e_size,output_size)
    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        rnn_e = rnn_activity[:, :, :self.rnn.e_size]
        out = self.fc(rnn_e)
        return out, rnn_activity
        
    
    
        