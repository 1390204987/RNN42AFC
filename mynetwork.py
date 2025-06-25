# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:43:41 2021

@author: NaN
"""


import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import torch.optim as optim
import numpy as np


class EIRecLinear(nn.Module):
    r"""Recurrent E-I Linear transformation.
    
    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def effective_weight(self):
        return torch.abs(self.weight) * self.mask

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)

class ForwardLinear(nn.Module):
    r"""between layers feedforward Linear transformation.
    
    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['hidden_size1', 'e_prop1','bias2', 'hidden_size2']
    
    def __init__(self, hidden_size1, e_prop1, hidden_size2, bias2=True):
        super().__init__()
#         higer layer
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.e_size1 = int(e_prop1 * hidden_size1)
        self.i_size1 = hidden_size1-self.e_size1

        # let the excitatory neuron in layer1 feedforward to layer2
        self.weight = nn.Parameter(torch.Tensor(self.hidden_size2, self.e_size1))
        mask = np.tile([1]*self.e_size1, (hidden_size2, 1))# get mask size hidden_size2*e_size1
        # let all the neuron in layer1 feedforward to layer2
        # self.weight = nn.Parameter(torch.Tensor(self.hidden_size2, self.hidden_size1))
        # mask = np.tile([1]*self.e_size1+[-1]*self.i_size1, (hidden_size2, 1))
        
        
        

        self.mask = torch.tensor(mask, dtype=torch.float32)
        if bias2:
            self.bias2 = nn.Parameter(torch.Tensor(hidden_size2))
        else:
            self.register_parameter('bias2', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
#         self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias2 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias2, -bound, bound)
            
    def effective_weight(self):
        return torch.abs(self.weight) * self.mask
    
    def forward(self, hidden1):
        # weight is non-negative
        e_hidden1 = hidden1[:, :self.e_size1] #let the excitatory neuron in layer1 feedforward to layer2 
        return F.linear(e_hidden1, self.effective_weight(), self.bias2)
    
    
class FeedbackLinear(nn.Module):
    r"""between layers feedforward Linear transformation.
    
    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['hidden_size2', 'e_prop2','bias1', 'hidden_size1']
    
    def __init__(self, hidden_size2, e_prop2, hidden_size1, bias1=True):
        super().__init__()
#         higer layer
        self.hidden_size2 = hidden_size2
        self.hidden_size1 = hidden_size1
        self.e_size2 = int(e_prop2 * hidden_size2)
        self.i_size2 = self.hidden_size2-self.e_size2

        # let the excitatory neuron in layer2 feedback to layer1
        self.weight = nn.Parameter(torch.Tensor(self.hidden_size1, self.e_size2))
        maskline = [1]*self.e_size2
        maskline[:int(self.e_size2*0.8)]=[0]*int(self.e_size2*0.8)
        mask = np.tile(maskline, (hidden_size1, 1))
        indx = range(len(maskline))
        for iy in range(self.hidden_size1):
            permmaskline = np.random.permutation(maskline)
            mask[iy,:] = permmaskline

        # let all the neuron in layer2 feedback to layer1
        # self.weight = nn.Parameter(torch.Tensor(self.hidden_size1,self.hidden_size2))
        # mask = np.tile([1]*self.e_size2+[-1]*self.i_size2, (hidden_size1, 1))

        self.mask = torch.tensor(mask, dtype=torch.float32)
        if bias1:
            self.bias1 = nn.Parameter(torch.Tensor(hidden_size1))
        else:
            self.register_parameter('bias1', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
#         self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias1, -bound, bound)
            
    def effective_weight(self):
        return torch.abs(self.weight) * self.mask
    
    def forward(self, hidden2):
        # weight is non-negative
        e_hidden2 = hidden2[:, :self.e_size2]#let the excitatory neuron in layer2 feedback to layer1 
        return F.linear(e_hidden2, self.effective_weight(), self.bias1)   
    
    
    
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

    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
        e_prop: float between 0 and 1, proportion of excitatory neurons
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, dt=None,
                 e_prop1=0.2, e_prop2=0.2, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        # first layer
        self.hidden_size1 = hidden_size1
        self.e_size1 = int(hidden_size1 * e_prop1)
        self.i_size1 = hidden_size1 - self.e_size1
        # second layer
        self.hidden_size2 = hidden_size2
        self.e_size2 = int(hidden_size2 * e_prop2)
        self.i_size2 = hidden_size2 - self.e_size2
        
        
        self.num_layers = 2
        # self.tau = 100
        # if dt is None:
        #     alpha = 1
        # else:
        #     alpha = dt / self.tau
        # self.alpha = alpha
        
        # self.oneminusalpha = 1 - alpha
        
        #defing tau seperately for each layer
        
        self.tau1 = 20
        self.tau2 = 20
        
        self.alpha1 = dt/self.tau1
        self.alpha2 = dt/self.tau2
        
        self.oneminusalpha1 = 1-self.alpha1
        self.oneminusalpha2 = 1-self.alpha2
        
        # Recurrent noise
        self._sigma_rec1 = np.sqrt(2*self.alpha1) * sigma_rec
        self._sigma_rec2 = np.sqrt(2*self.alpha2) * sigma_rec

        # self.input2h = PosWLinear(input_size, hidden_size)
        self.input2h = nn.Linear(input_size, hidden_size1)
        self.h2h = EIRecLinear(hidden_size1, e_prop=0.8)
        self.h12h2 = ForwardLinear(hidden_size1, e_prop1,hidden_size2)
        self.h22h1 = FeedbackLinear(hidden_size2, e_prop2,hidden_size1)
#         self.h32h2 = FeedbackLinear(hidden_size1, e_prop1,hidden_size2, e_prop2)
#         self.h32h1 = FeedbackLinear(hidden_size1, e_prop1,hidden_size2, e_prop2)

    def init_hidden1(self,inputs):
        batch_size = inputs.shape[1]
        return (torch.zeros(batch_size, self.hidden_size1).to(inputs.device),
                torch.zeros(batch_size, self.hidden_size1).to(inputs.device))
    
    def init_hidden2(self,inputs):
        batch_size = inputs.shape[1]
        return (torch.zeros(batch_size, self.hidden_size2).to(inputs.device),
                torch.zeros(batch_size, self.hidden_size2).to(inputs.device))
    
    def recurrence1(self, inputs, hidden1, hidden2):
        """Recurrence helper."""
        state1, output1 = hidden1
        state2, output2 = hidden2
        # with feedback
        total_input1 = self.input2h(inputs) + self.h2h(output1) + self.h22h1(output2)
        #without feedback
        # total_input1 = self.input2h(inputs) + self.h2h(output1)   
        
        state1 = state1 * self.oneminusalpha1 + total_input1 * self.alpha1
        state1 += self._sigma_rec1 * torch.randn_like(state1)
        output1 = torch.relu(state1)
        return state1, output1
    
    def recurrence2(self, output1, hidden2):
        """Recurrence helper."""
        state2, output2 = hidden2
        total_input2 = self.h12h2(output1) + self.h2h(output2)
        state2 = state2 * self.oneminusalpha2 + total_input2 * self.alpha2
        state2 += self._sigma_rec2 * torch.randn_like(state2)
        output2 = torch.relu(state2)
        return state2, output2

    def forward(self, inputs, hidden1=None, hidden2=None):
        """Propogate input through the network."""
        if hidden1 is None:
            hidden1 = self.init_hidden1(inputs)
        if hidden2 is None:
            hidden2 = self.init_hidden2(inputs)
        
        output1 = []
        # output1.append(inioutput1)
        output2 = []
        # output2.append(inioutput2)
        steps = range(inputs.size(0))
        for i in steps:
            hidden1 = self.recurrence1(inputs[i], hidden1, hidden2)
            output1.append(hidden1[1])
            hidden2 = self.recurrence2(output1[i], hidden2)
            output2.append(hidden2[1])

        output1 = torch.stack(output1, dim=0)    
        output2 = torch.stack(output2, dim=0)
        # output2 = output2[1:,:,:]
        return output2, hidden2    
    
class Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self,hp, **kwargs):
        super().__init__()
        
        input_size = hp['n_input']
        hidden_size1 = hp['hidden_size1']
        hidden_size2 = hp['hidden_size2']
        output_size = hp['n_output']
        # Excitatory-inhibitory RNN first layer
        self.rnn = EIRNN(input_size,hidden_size1,hidden_size2,**kwargs)
        # self.fc = PosWLinear(self.rnn.e_size, output_size)
        self.fc = nn.Linear(self.rnn.e_size2, output_size)
        # self.softmax = nn.functional.softmax(output_size, dim = 2)
    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        rnn_e = rnn_activity[:, :, :self.rnn.e_size2]
        out = self.fc(rnn_e)
        # out = self.softmax(out,dim = 2)
        return out, rnn_activity