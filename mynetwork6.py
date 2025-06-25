# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:37:54 2024
here the net has 2 hidden layer,the first layer connected direct to heading direction input,
biologically like the VIP or a subpolulation in LIP,the second layer get the input of color 
target encoding and connneted with the first hidden layer and output layer,biologically like 
another subpopulation in LIP.
between the first hidden laier and the second hidden laier there exist both feed-
forward connection and feedback connection, and the feedback connetion including
exitatory and inhibitory
the readout weight is fixed:分配连接: 每个输出单元与4个相邻的输入单元有较强的连接。
高斯分布衰减: 对于每个输出单元，连接强度按照高斯分布衰减。
the readin weight is fixed:分配连接: 每个输出单元与4个相邻的输入单元有较强的连接。
高斯分布衰减: 对于每个输出单元，连接强度按照高斯分布衰减。
@author: NaN
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F 
import math
import torch.optim as optim
import numpy as np

class fixedReadout(nn.Module):
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
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.device = device
        mask11 = np.zeros((out_features,hidden_size1))
        mask21 = np.ones((out_features,hidden_size2))
        mask = np.concatenate((mask11,mask21),axis=1)
        self.mask = torch.tensor(mask,dtype=torch.float32).to(device)
        self.weight_21 = self.generate_fixed_weight().to(self.device)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(device)
        else:
            # self.register_parameter('bias', None)
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def generate_fixed_weight(self):
        # Create a fixed weight matrix with Gaussian-like connections
        fixed_weight = np.zeros((self.out_features, self.hidden_size2))
        connections_per_output = self.hidden_size2 // self.out_features
        for i in range(self.out_features):
            center = i * connections_per_output + connections_per_output // 2
            for j in range(self.hidden_size2):
                # Gaussian-like weight centered around `center`
                distance = min(abs(j - center),self.hidden_size2-abs(j-center))
                # distance = abs(j - center)
                sigma = connections_per_output / 6.0
                fixed_weight[i, j] = np.exp(-distance**2 / (2 * sigma**2))
        
        return torch.tensor(fixed_weight, dtype=torch.float32)

    
    def effective_weight(self):

        # 使用非原地操作更新 weight
        # 创建一个新的张量以避免原地操作
        new_weight = self.weight.clone()  # 克隆一个新的张量
        new_weight[:,self.hidden_size1:] = self.weight_21
        # 返回应用了掩码的权重
        return new_weight * self.mask
    
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
            
    def __init__(self, device,hidden_size1,hidden_size2,recurrency1=1,recurrency2=1,feedforward_stren=1,feedback_stren=1,bias=True):
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
        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask,dtype=torch.float32).to(device)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.hidden_size)).to(device)
        else:
            self.register_parameter('bias',None)
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
    
    def __init__(self,device,in_feature_heading,in_feature_targcolor,in_feature_rules,hidden_size1,hidden_size2,bias=True):
        super().__init__()
        self.in_feature_heading = in_feature_heading
        self.in_feature_targcolor = in_feature_targcolor
        self.in_feature_rules = in_feature_rules
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.device = device
        in_features = in_feature_heading + in_feature_targcolor + in_feature_rules
        out_features = hidden_size1 + hidden_size2
        
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        
        mask11 = np.ones((hidden_size1,in_feature_heading))
        mask12 = np.zeros((hidden_size2,in_feature_heading))
        mask21 = np.zeros((hidden_size1,in_feature_targcolor)) 
        mask22 = np.ones((hidden_size2,in_feature_targcolor))
        mask31 = np.zeros((hidden_size1,in_feature_rules))
        mask32 = np.ones((hidden_size2,in_feature_rules))
        mask1 = np.concatenate((mask11,mask21,mask31),axis=1)
        mask2 = np.concatenate((mask12,mask22,mask32),axis=1)
        mask = np.concatenate((mask1,mask2),axis=0)
        self.mask = torch.tensor(mask,dtype=torch.float32).to(device)
        self.weight_11 = self.generate_fixed_weight(self.hidden_size1,self.in_feature_heading-1).to(self.device)
        self.weight_22 = self.generate_fixed_weight(self.hidden_size2,int(self.in_feature_targcolor/2)).to(self.device) #T1 ring
        self.weight_23 = self.generate_fixed_weight(self.hidden_size2,int(self.in_feature_targcolor/2)).to(self.device) #T2 ring
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(device)
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()  
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)    
            
    def generate_fixed_weight(self, hidden_size, in_feature_size):
        """Initialize weights with fixed Gaussian distribution for specific connections."""
        fixed_weight = np.zeros((hidden_size, in_feature_size))
        connections_per_input = hidden_size // in_feature_size
        for i in range(in_feature_size):
            center = i * connections_per_input + connections_per_input // 2
            for j in range(hidden_size):
                # Gaussian-like weight centered around `center`
                distance = min(abs(j - center),hidden_size-abs(j-center))
                sigma = connections_per_input / 6.0
                fixed_weight[j, i] = np.exp(-distance**2 / (2 * sigma**2))
        return torch.tensor(fixed_weight, dtype=torch.float32)
    def effective_weight(self):

        # 使用非原地操作更新 weight
        # 创建一个新的张量以避免原地操作
        new_weight = self.weight.clone()  # 克隆一个新的张量
        new_weight[:self.hidden_size1, 1:self.in_feature_heading] = self.weight_11
        new_weight[self.hidden_size1+2:, self.in_feature_heading:self.in_feature_heading + int(self.in_feature_targcolor/2)] = self.weight_22[:-2,:]
        new_weight[self.hidden_size1:, self.in_feature_heading + int(self.in_feature_targcolor/2):self.in_feature_heading + self.in_feature_targcolor] = self.weight_23
    
        # 返回应用了掩码的权重
        return new_weight * self.mask

    
    def forward(self,input):
        # check1 = self.effective_weight().detach().numpy()
        # check2 = self.bias.detach().numpy()
        # check3 = F.linear(input,self.effective_weight(),self.bias)  
        # check3 = check3.detach().numpy()
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
                 dt=None, sigma_rec1=0,sigma_rec2=0, **kwargs):
        super().__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size1+hidden_size2
        self.n_rule = n_rule
        self.n_eachring = n_eachring
        self.num_ring = num_ring
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        self.tau = 100
        self.L1_tau = L1_tau
        self.L2_tau = L2_tau
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
            # self.sigma_feedforward =np.sqrt(2*alpha)*sigma_feedforward
            # self.sigma_feedback = np.sqrt(2*alpha)*sigma_feedback  
            
            self.input2h = ReadinLinear(device,insize_heading,insize_targcolor,insizse_rules,hidden_size1,hidden_size2,bias=True)
            self.h2h = EIRecLinear(device,hidden_size1,hidden_size2,recurrency1=recur1,recurrency2=recur2,
                                   feedforward_stren=fforwardstren,feedback_stren=fbackstren,bias=True)
            
    def init_hidden(self,inputs):
        batch_size = inputs.shape[1]
        return(torch.zeros(batch_size,self.hidden_size).to(inputs.device),
               torch.zeros(batch_size,self.hidden_size).to(inputs.device))     
    
    def recurrence(self, inputs,hidden):
        # inputs_heading = inputs[:,:1+self.n_eachring]
        # inputs_targcolor = inputs[:,1+self.n_eachring:1+self.num_ring*self.n_eachring] 
        # inputs_rules = inputs[:,1+self.num_ring*self.n_eachring:1+self.num_ring*self.n_eachring+self.n_rule]
        state,output = hidden

        # total_input = self.input2h(inputs_heading)+self.input2h(inputs_targcolor)+self.input2h(inputs_rules)+self.h2h(output)
        total_input = self.input2h(inputs)+self.h2h(output)
        state = state*self.oneminusalpha + total_input*self.alpha
        state[:,:self.hidden_size1] += self._sigma_rec1*torch.randn_like(state[:,:self.hidden_size1])
        state[:,self.hidden_size1:] += self._sigma_rec2*torch.randn_like(state[:,self.hidden_size1:])
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
        # fforwardstren = 0
        # fbackstren = 0
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        self.device = torch.device("cpu")
        self.rnn = EIRNN(self.device,insize_heading,insize_targcolor,insize_rules,
                          hidden_size1, hidden_size2,n_rule,n_eachring,num_ring,recur1,recur2,fforwardstren,fbackstren,
                         sigma_feedforward,sigma_feedback,L1_tau,L2_tau,sigma_rec1=sigma_rec1,sigma_rec2=sigma_rec2,**kwargs)            
        self.fc = fixedReadout(self.device,hidden_size1,hidden_size2,output_size,bias=True)
        
    def forward(self, x):
        
        x=x.to(self.device)
        rnn_activity,_ = self.rnn(x)        
        out = self.fc(rnn_activity)
        return out, rnn_activity        
