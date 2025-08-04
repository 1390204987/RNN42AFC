# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:22:29 2025

@author: 13902
"""

# -*- coding: utf-8 -*-
"""

here the net has 2 hidden layer,the first layer connected direct to heading direction input,
biologically like the VIP or a subpolulation in LIP,the second layer get the input of color 
target encoding and connneted with the first hidden layer and output layer,biologically like 
another subpopulation in LIP.
between the first hidden laier and the second hidden laier there exist both feed-
forward connection and feedback connection, and the feedback connetion including
exitatory and inhibitory

different from "mynetwork_new3" the two laryer communication is slower than single layer 
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
    
    def __init__(self,device,hidden_size1,hidden_size2,out_features,bias=True):
        super().__init__()
        in_features = hidden_size1+hidden_size2
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        mask1 = np.zeros((out_features,hidden_size1))
        mask2 = np.ones((out_features,hidden_size2))
        mask = np.concatenate((mask1,mask2),axis=1)
        self.mask = torch.tensor(mask,dtype=torch.float32).to(device)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(device)
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
       define the connection between hidden layer1 and hidden layer2 
    Args:
        hidden_size1: int, layer size
        hidden_size2: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constant__ = ['bias','hidden_size1','hidden_size2']    
            
    def __init__(self, device,hidden_size1,hidden_size2,bias=True,recurrency1=1,recurrency2=1,feedforward_stren=1,feedback_stren=1):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size = self.hidden_size1+self.hidden_size2
        self.weight = nn.Parameter(torch.Tensor(self.hidden_size,self.hidden_size))
        mask11 = np.ones((hidden_size1,hidden_size1))*recurrency1
        mask12 = np.ones((hidden_size1,hidden_size2))*feedback_stren
        # mask21 = np.ones((hidden_size2,hidden_size1))
        mask21 = np.ones((hidden_size2,hidden_size1))*feedforward_stren
        mask22 = np.ones((hidden_size2,hidden_size2))*recurrency2
        mask = np.concatenate((np.concatenate((mask11,mask12),axis=1),np.concatenate((mask21,mask22),axis=1)))
        self.mask = torch.tensor(mask,dtype=torch.float32).to(device)
        
        # Build "positivity" mask: only force weights in 12 and 21 blocks to be positive
        pos_mask = np.zeros((self.hidden_size, self.hidden_size), dtype=np.float32)
        # block (1,2)
        pos_mask[:hidden_size1, hidden_size1:] = 1.0  # feedback
        # block (2,1)
        pos_mask[hidden_size1:, :hidden_size1] = 1.0  # feedforward
        self.positive_mask = torch.tensor(pos_mask, dtype=torch.float32).to(device)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.hidden_size)).to(device)
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
        # return self.weight*self.mask
        # weights in mask12 and mask21 (positive_mask==1) are made positive via softplus
        # others remain unconstrained

        W_raw = self.weight
        W_eff = self.positive_mask * (W_raw**2) + (1 - self.positive_mask) * W_raw
        return W_eff * self.mask
        
    def forward(self, input):
        W_eff1 = self.effective_weight()*self.positive_mask
        W_eff2 = self.effective_weight()*(1-self.positive_mask)
        return F.linear(input[0],W_eff1,self.bias) + F.linear(input[1],W_eff2,self.bias)
    
class ReadinLinear(nn.Module):
    """neuron read in Linear transformation.
    Args:
        in_feature: the input feature size
        out_feature: the dimension of hidden1
    """
    __constant__ = ['bias','in_features']     
    
    def __init__(self,device,in_feature_heading,in_feature_targcolor,in_feature_rules,hidden_size1,hidden_size2,bias=True):
        super().__init__()
        self.in_feature_heading = in_feature_heading
        self.in_feature_targcolor = in_feature_targcolor
        self.in_feature_rules = in_feature_rules
        self.device = device
        in_features = in_feature_heading + in_feature_targcolor + in_feature_rules
        out_features = hidden_size1 + hidden_size2
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        
        mask11 = np.ones((hidden_size1,in_feature_heading))
        mask12 = np.zeros((hidden_size2,in_feature_heading))
        mask21 = np.zeros((hidden_size1,in_feature_targcolor)) 
        mask22 = np.ones((hidden_size2,in_feature_targcolor))
        mask31 = np.zeros((hidden_size1,in_feature_rules))
        mask32 = np.zeros((hidden_size2,in_feature_rules))
        mask1 = np.concatenate((mask11,mask21,mask31),axis=1)
        mask2 = np.concatenate((mask12,mask22,mask32),axis=1)
        mask = np.concatenate((mask1,mask2),axis=0)
        self.mask = torch.tensor(mask,dtype=torch.float32).to(self.device)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(self.device)
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
    def __init__(self, device,insize_heading,insize_targcolor,insizse_rules,
                 hidden_size1, hidden_size2,n_rule,n_eachring,num_ring, 
                 recur1,recur2,fforwardstren,fbackstren,sigma_feedforward,sigma_feedback,L1_tau,L2_tau,
                 dt=None, sigma_rec1=0.1,sigma_rec2=0.1, **kwargs):
        super().__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size1+hidden_size2
        self.n_rule = n_rule
        self.n_eachring = n_eachring
        self.num_ring = num_ring
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.L1_tau = L1_tau
        self.L2_tau = L2_tau
        self.tau = 30
        if dt is None:
            alpha = 1
        else:
            alpha = dt/self.tau
            alpha1 = dt/self.L1_tau
            alpha2 = dt/self.L2_tau
            self.alpha = alpha
            self.alpha1 = alpha1
            self.alpha2 = alpha2
            self.oneminusalpha = 1-alpha
            self.oneminusalpha1 = 1-alpha1
            self.oneminusalpha2 = 1-alpha2
 
            #Recurrent noise
            self._sigma_rec1 = sigma_rec1
            self._sigma_rec2 = sigma_rec2
            # self._sigma_rec1 =np.sqrt(2*alpha)*sigma_rec1
            # self._sigma_rec2 = np.sqrt(2*alpha)*sigma_rec2
            #feedforward and feedback noise
            # self.sigma_feedforward = sigma_feedforward
            # self.sigma_feedback = sigma_feedback   
            
            self.input2h = ReadinLinear(device,insize_heading,insize_targcolor,insizse_rules,hidden_size1,hidden_size2)
            self.h2h = EIRecLinear(device,hidden_size1,hidden_size2,recurrency1=recur1,recurrency2=recur2,feedforward_stren=fforwardstren,feedback_stren=fbackstren)
            
    def init_hidden(self,inputs):
        batch_size = inputs.shape[1]
        return(torch.zeros(batch_size,self.hidden_size).to(inputs.device),
               torch.zeros(batch_size,self.hidden_size).to(inputs.device))     
    
    def recurrence(self, inputs,hidden,output_1):
        # inputs_heading = inputs[:,:1+self.n_eachring]
        # inputs_targcolor = inputs[:,1+self.n_eachring:1+self.num_ring*self.n_eachring] 
        # inputs_rules = inputs[:,1+self.num_ring*self.n_eachring:1+self.num_ring*self.n_eachring+self.n_rule]
        state,output = hidden
        # output_two = torch.stack([output, output_1], dim=1) 
        # total_input = self.input2h(inputs_heading)+self.input2h(inputs_targcolor)+self.input2h(inputs_rules)+self.h2h(output)
        total_input = self.input2h(inputs)+self.h2h((output,output_1))
        state[:,:self.hidden_size1] = state[:,:self.hidden_size1]*self.oneminusalpha1 + total_input[:,:self.hidden_size1]*self.alpha1
        state[:,self.hidden_size1:] = state[:,self.hidden_size1:]*self.oneminusalpha2 + total_input[:,self.hidden_size1:]*self.alpha2
        
        # state[:,:self.hidden_size1] += self._sigma_rec1*torch.randn_like(state[:,:self.hidden_size1])*torch.mean(state[:,:self.hidden_size1],dim=1,keepdim=True)
        # state[:,self.hidden_size1:] += self._sigma_rec2*torch.randn_like(state[:,self.hidden_size1:])*torch.mean(state[:,self.hidden_size1:],dim=1,keepdim=True)
        # state[:,:self.hidden_size1] += self._sigma_rec1*torch.randn_like(state[:,:self.hidden_size1])
        # state[:,self.hidden_size1:] += self._sigma_rec2*torch.randn_like(state[:,self.hidden_size1:])
        
        state[:,:self.hidden_size1] += self._sigma_rec1*torch.randn_like(state[:,:self.hidden_size1])*state[:,:self.hidden_size1]
        state[:,self.hidden_size1:] += self._sigma_rec2*torch.randn_like(state[:,self.hidden_size1:])*state[:,self.hidden_size1:]
        output = F.relu(state)
        return state, output															 
    
    def forward(self, inputs, hidden=None):
        """ Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(inputs)
            
        outputs = []
        steps = range(inputs.size(0))
       
        for i in steps:
            if i<1:
                hidden = self.recurrence(inputs[i],hidden,hidden[1])
            else:
                hidden = self.recurrence(inputs[i],hidden,outputs[i-1])
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
    def __init__(self,hp,device,**kwargs):
        super().__init__()
            
        insize_heading = hp['n_input_heading']
        insize_targcolor = hp['n_input_targcolor'] 
        insize_rules = hp['n_input_rules']
        hidden_size1 = hp['hidden_size1']
        hidden_size2 = hp['hidden_size2']
        output_size = hp['n_output'] 
        n_rule = hp['n_rule']
        n_eachring = hp['n_eachring']
        num_ring = hp['num_ring']
        sigma_rec1 = hp['sigma_rec1']
        sigma_rec2 = hp['sigma_rec2']
        recur1 = hp['recur1']
        recur2 = hp['recur2']
        fforwardstren = hp['fforwardstren']
        fbackstren = hp['fbackstren']
        sigma_feedforward= hp['sigma_feedforward']
        sigma_feedback= hp['sigma_feedback']
        L1_tau = hp['L1_tau']
        L2_tau = hp['L2_tau']
        # recur1 = 0
        # recur2 = 0
        # fforwardstren = 1
        # fbackstren = 0.3
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.device = device
        self.rnn = EIRNN(self.device,insize_heading,insize_targcolor,insize_rules,
                          hidden_size1, hidden_size2,n_rule,n_eachring,num_ring,recur1,recur2,fforwardstren,fbackstren,
                         sigma_feedforward,sigma_feedback,L1_tau,L2_tau,sigma_rec1=sigma_rec1,sigma_rec2=sigma_rec2,**kwargs)            
        self.fc = ReadoutLinear(self.device,hidden_size1,hidden_size2,output_size)
        
    def forward(self, x):
        
        x=x.to(self.device)
        rnn_activity,_ = self.rnn(x)        
        out = self.fc(rnn_activity) 
        return out, rnn_activity        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        