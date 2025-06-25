# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:28:04 2022
here the net has 2 hidden layer,the first layer connected direct to input,
biologically like the VIP or a subpolulation in LIP, the second layer connneted
with the first hidden layer and the output layer,biologically like another sub-
population in LIP
this script theoritically different from mynetwork.py, but different implemen-
tation
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
    
    def __init__(self,hidden_size1,hidden_size2,out_features,bias=True):
        super().__init__()
        in_features = hidden_size1+hidden_size2
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        mask1 = np.zeros((out_features,hidden_size1))
        mask2 = np.ones((out_features,hidden_size2))
        mask = np.concatenate((mask1,mask2),axis=1)
        self.mask = torch.tensor(mask,dtype=torch.float32)
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
        return self.weight*self.mask
    
    def forward(self, input):
        # weight is non-negative
        return F.linear(input,self.effective_weight(),self.bias)
    
class EIRecLinear(nn.Module):
    """Recurrent E-I Linear transformation.
    
    Args:
        hidden_size1: int, layer size
        hidden_size2: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constant__ = ['bias','hidden_size1','hidden_size2']
    
    def __init__(self, hidden_size1,hidden_size2,bias=True):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size = self.hidden_size1+self.hidden_size2
        self.weight = nn.Parameter(torch.Tensor(self.hidden_size,self.hidden_size))
        mask11 = np.ones((hidden_size1,hidden_size1))
        mask12 = np.zeros((hidden_size1,hidden_size2))
        mask21 = np.ones((hidden_size2,hidden_size1))
        mask22 = np.ones((hidden_size2,hidden_size2))
        mask = np.concatenate((np.concatenate((mask11,mask12),axis=1),np.concatenate((mask21,mask22),axis=1)))
        self.mask = torch.tensor(mask,dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        else:
            self.registser_parameter('bias',None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_=init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)
            
    def effective_weight(self):
        return self.weight*self.mask
        # return self.weight
    
    def forward(self, input):
        return F.linear(input, self.effective_weight(), self.bias)
    
class ReadinLinear(nn.Module):
    """neuron read in Linear transformation.
    Args:
        in_feature: the input feature size
        out_feature: the dimension of hidden1
    """
    __constant__ = ['bias','in_features']
    
    def __init__(self,in_features,hidden_size1,hidden_size2,bias=True):
        super().__init__()
        self.in_features = in_features
        out_features = hidden_size1 + hidden_size2
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        mask1 = np.ones((hidden_size1,in_features))
        mask2 = np.zeros((hidden_size2,in_features))
        mask = np.concatenate((mask1,mask2),axis=0)
        self.mask = torch.tensor(mask,dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)
        
    def effective_weight(self):
        return self.weight*self.mask
    
    def forward(self,input):
        return F.linear(input,self.effective_weight(),self.bias)
        
class EIRNN(nn.Module):
    """ E-I RNN.
    Args:
        input_sise: Number of input neurons
        hidden_size: The sum number of hidden layer1 neuron and hidden layer2 neuron
        
    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch.hidden_size)
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, dt=None, sigma_rec=0.1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size1+hidden_size2
        
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt/self.tau
            self.alpha = alpha
            self.oneminusalpha = 1-alpha
            #Recurrent noise
            self._sigma_rec =np.sqrt(2*alpha)*sigma_rec
            self.input2h = ReadinLinear(input_size,hidden_size1,hidden_size2)
            self.h2h = EIRecLinear(hidden_size1,hidden_size2)
    
    def init_hidden(self,inputs):
        batch_size = inputs.shape[1]
        return(torch.zeros(batch_size,self.hidden_size).to(inputs.device),
               torch.zeros(batch_size,self.hidden_size).to(inputs.device))
    
    def recurrence(self, inputs,hidden):
        state,output = hidden
        total_input = self.input2h(inputs)+self.h2h(output)
        state = state*self.oneminusalpha + total_input*self.alpha
        state += self._sigma_rec*torch.randn_like(state)
        output = F.relu(state)
        return state, output
    
    def forward(self, inputs, hidden=None):
        """ Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(inputs)
            
        outputs = []
        steps = range(inputs.size(0))
       
        for i in steps:
            hidden = self.recurrence(inputs[i],hidden)
            outputs.append(hidden[1])
        
        outputs = torch.stack(outputs,dim=0)
        return outputs, hidden
    
class Net(nn.Module):
    """ Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size1: int, first hidden layer size
        hidden_size2: int, second hidden layer size
        rnn: str, type of RNN, rnn, or lstm
    """
    def __init__(self,hp,**kwargs):
        super().__init__()
            
        input_size = hp['n_input']
        hidden_size1 = hp['hidden_size1']
        hidden_size2 = hp['hidden_size2']
        output_size = hp['n_output']
            
        self.rnn = EIRNN(input_size, hidden_size1, hidden_size2,**kwargs)            
        self.fc = ReadoutLinear(hidden_size1,hidden_size2,output_size)
            
    def forward(self, x):
        rnn_activity,_ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    