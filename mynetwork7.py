# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 08:57:01 2025
2 hidden layers, heading information input to a part of neuron in layer1, 
target color information input to a part of neuron in layer1&layer2, 
the connetion weight trained by back propogation,
 2 hideen layer have feedforward and feedback connection between part of neuron
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
        mask11 = torch.zeros(out_features,hidden_size1,device=device)
        mask12 = torch.zeros(out_features,hidden_size2,device=device)
        # genereate random index to set 1/4 hidden2 neuron get read out
        # 固定随机种子，确保可复现性
        torch.manual_seed(11)
        self.readout_size = hidden_size2 //4 # 选择 1/4 的神经元
        selected = torch.randperm(hidden_size2, device=device)[:self.readout_size].sort().values
        sparsity = 0.5
        mask12[:, selected] = self._creat_sparsemask(mask12[:, selected],sparsity)
        mask = torch.cat((mask11,mask12),axis=1).to(device)
        # self.mask = torch.tensor(mask,dtype=torch.float32).to(device)
        # 注册掩码为不可训练的 Buffer
        self.register_buffer('fixed_mask', mask)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def _creat_sparsemask(self,mask,sparsity):
        total = mask.numel()
        k = int((1 - sparsity) * total)
        sparsemask = mask.flatten()
        idx = torch.randperm(total)[:k]
        sparsemask[idx] = 1
        return sparsemask.view_as(mask)
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        self.weight.data *= self.fixed_mask
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def effective_weight(self):
        return self.weight*self.fixed_mask
        # return self.weight      
          
        
    def forward(self, input):
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

        
        # 初始化所有mask
        mask11 = self._creat_sparsemask(
            size=(hidden_size1, hidden_size1),
            sparsity=0.9,
            strength=recurrency1,
            device=device,
            seed=21  
        )
        
        mask12 = self._creat_sparsemask(
            size=(hidden_size1, hidden_size2),
            sparsity=0.7,
            strength=feedback_stren,
            device=device,
            src_ratio=0.5,  # 使用100%的hidden_size1神经元
            tgt_ratio=0.5,   # 使用100%的hidden_size2神经元
            seed=22
        )
        
        mask21 = self._creat_sparsemask(
            size=(hidden_size2, hidden_size1),
            sparsity=0.5,
            strength=feedforward_stren,
            device=device,
            src_ratio=0.5,  # 使用100%的hidden_size2神经元
            tgt_ratio=0.5,   # 使用100%的hidden_size1神经元
            seed=23
        )
        
        mask22 = self._creat_sparsemask(
            size=(hidden_size2, hidden_size2),
            sparsity=0.1,
            strength=recurrency2,
            device=device,
            seed=24
        )
        
        
        mask = torch.cat((torch.cat((mask11,mask12),axis=1),torch.cat((mask21,mask22),axis=1))).to(device)
        mask.fill_diagonal_(0)  # 禁止自连接
        # 注册掩码为不可训练的 Buffer
        # self.register_buffer('fixed_mask', mask)
        self.fixed_mask = mask
        
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
            self.register_parameter('bias',None)
        self.reset_parameters()     
 
    def _creat_sparsemask(self, size, sparsity, strength, device, src_ratio=None, tgt_ratio=None, seed=None):
        """
        可重复随机连接的稀疏mask生成函数
        Args:
            seed: 随机种子 (None表示不固定)
        """
        rows, cols = size
        mask = torch.zeros(size, device=device)
        
        # 设置随机种子（如果提供）
        if seed is not None:
            torch.manual_seed(seed)
        
        # 1. 确定参与连接的神经元索引
        src_indices = (torch.randperm(rows, device=device)[:int(rows*src_ratio)] 
                      if src_ratio is not None 
                      else torch.arange(rows, device=device))
        
        tgt_indices = (torch.randperm(cols, device=device)[:int(cols*tgt_ratio)] 
                      if tgt_ratio is not None 
                      else torch.arange(cols, device=device))
        
        # 2. 计算需要建立的连接数
        total_possible = len(src_indices) * len(tgt_indices)
        num_connections = max(1, int((1 - sparsity) * total_possible))
        
        # 3. 生成可重复的随机连接
        if num_connections > 0 and len(src_indices) > 0 and len(tgt_indices) > 0:
            # 生成所有可能连接的位置
            grid_rows = src_indices.unsqueeze(1).expand(-1, len(tgt_indices)).flatten()
            grid_cols = tgt_indices.expand(len(src_indices), -1).flatten()
            all_positions = torch.stack([grid_rows, grid_cols], dim=1)
            
            # 随机选择（结果可重复）
            perm = torch.randperm(len(all_positions), device=device)
            selected_positions = all_positions[perm[:num_connections]]
            
            # 应用连接
            mask[selected_positions[:, 0], selected_positions[:, 1]] = 1
        
        return mask * strength
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        self.weight.data *= self.fixed_mask
        if self.bias is not None:
            fan_in,_=init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)           
                
    def effective_weight(self):
        # return self.weight*self.fixed_mask
        # return self.weight   
        W_raw = self.weight
        W_eff = self.positive_mask * (W_raw**2) + (1 - self.positive_mask) * W_raw
        return W_eff * self.fixed_mask
        
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
        in_features = in_feature_heading + in_feature_targcolor + in_feature_rules
        out_features = hidden_size1 + hidden_size2
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))
        

        mask11 = self._creat_sparsemask(
            size=(hidden_size1, in_feature_heading),
            sparsity=0.5,
            device=device,
            src_ratio=1.0,  # 使用100%的hidden_size2神经元
            tgt_ratio=0.5,   # 使用100%的hidden_size1神经元
            seed=31  
        )
        
        mask21 = torch.zeros(hidden_size2,in_feature_heading)
        

        mask12 = self._creat_sparsemask(
            size=(hidden_size1, in_feature_targcolor),
            sparsity=0.5,
            device=device,
            src_ratio=1.0,  # 使用100%的hidden_size2神经元
            tgt_ratio=0,   # 使用100%的hidden_size1神经元
            seed=32  
        )
                
        mask22 = self._creat_sparsemask(
            size=(hidden_size2, in_feature_targcolor),
            sparsity=0.5,
            device=device,
            src_ratio=1.0,  # 使用100%的hidden_size2神经元
            tgt_ratio=0.5,   # 使用100%的hidden_size1神经元
            seed=33  
        )
        
        mask13 = self._creat_sparsemask(
            size=(hidden_size1, in_feature_rules),
            sparsity=0.8,
            device=device,
            src_ratio=1.0,  # 使用100%的hidden_size2神经元
            tgt_ratio=0.1,   # 使用100%的hidden_size1神经元
            seed=34  
        )
        
        mask23 = self._creat_sparsemask(
            size=(hidden_size2, in_feature_rules),
            sparsity=0.8,
            device=device,
            src_ratio=1.0,  # 使用100%的hidden_size2神经元
            tgt_ratio=0.1,   # 使用100%的hidden_size1神经元
            seed=35  
        )
        mask1 = torch.cat((mask11,mask12,mask13),axis=1)
        mask2 = torch.cat((mask21,mask22,mask23),axis=1)
        mask = torch.cat((mask1,mask2),axis=0)
        # 注册掩码为不可训练的 Buffer
        self.register_buffer('fixed_mask', mask)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features)).to(device)
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()  
        
    def _creat_sparsemask(self, size, sparsity,  device, src_ratio=None, tgt_ratio=None, seed=None):
        """
        可重复随机连接的稀疏mask生成函数
        Args:
            seed: 随机种子 (None表示不固定)
        """
        rows, cols = size
        mask = torch.zeros(size, device=device)
        
        # 设置随机种子（如果提供）
        if seed is not None:
            torch.manual_seed(seed)
        
        # 1. 确定参与连接的神经元索引
        src_indices = (torch.randperm(rows, device=device)[:int(rows*src_ratio)] 
                      if src_ratio is not None 
                      else torch.arange(rows, device=device))
        
        tgt_indices = (torch.randperm(cols, device=device)[:int(cols*tgt_ratio)] 
                      if tgt_ratio is not None 
                      else torch.arange(cols, device=device))
        
        # 2. 计算需要建立的连接数
        total_possible = len(src_indices) * len(tgt_indices)
        num_connections = max(1, int((1 - sparsity) * total_possible))
        
        # 3. 生成可重复的随机连接
        if num_connections > 0 and len(src_indices) > 0 and len(tgt_indices) > 0:
            # 生成所有可能连接的位置
            grid_rows = src_indices.unsqueeze(1).expand(-1, len(tgt_indices)).flatten()
            grid_cols = tgt_indices.expand(len(src_indices), -1).flatten()
            all_positions = torch.stack([grid_rows, grid_cols], dim=1)
            
            # 随机选择（结果可重复）
            perm = torch.randperm(len(all_positions), device=device)
            selected_positions = all_positions[perm[:num_connections]]
            
            # 应用连接
            mask[selected_positions[:, 0], selected_positions[:, 1]] = 1
        
        return mask
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        self.weight.data *= self.fixed_mask
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)    
    
    def effective_weight(self):
        return self.weight*self.fixed_mask 
    
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
        # self.tau = 30
        if dt is None:
            alpha = 1
        else:
            # alpha = dt/self.tau
            alpha1 = dt/self.L1_tau
            alpha2 = dt/self.L2_tau
            # self.alpha = alpha
            self.alpha1 = alpha1
            self.alpha2 = alpha2
            # self.oneminusalpha = 1-alpha
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
    
    def recurrence(self, inputs,hidden):
        # inputs_heading = inputs[:,:1+self.n_eachring]
        # inputs_targcolor = inputs[:,1+self.n_eachring:1+self.num_ring*self.n_eachring] 
        # inputs_rules = inputs[:,1+self.num_ring*self.n_eachring:1+self.num_ring*self.n_eachring+self.n_rule]
        state,output = hidden
         
        # total_input = self.input2h(inputs_heading)+self.input2h(inputs_targcolor)+self.input2h(inputs_rules)+self.h2h(output)
        total_input = self.input2h(inputs)+self.h2h(output)
        # state = state*self.oneminusalpha + total_input*self.alpha
        state[:,:self.hidden_size1] = state[:,:self.hidden_size1]*self.oneminusalpha1 + total_input[:,:self.hidden_size1]*self.alpha1
        state[:,self.hidden_size1:] = state[:,self.hidden_size1:]*self.oneminusalpha2 + total_input[:,self.hidden_size1:]*self.alpha2
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
        # fforwardstren = 1
        # fbackstren = 0.3
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.rnn = EIRNN(self.device,insize_heading,insize_targcolor,insize_rules,
                          hidden_size1, hidden_size2,n_rule,n_eachring,num_ring,recur1,recur2,fforwardstren,fbackstren,
                         sigma_feedforward,sigma_feedback,L1_tau,L2_tau,sigma_rec1=sigma_rec1,sigma_rec2=sigma_rec2,**kwargs)            
        self.fc = ReadoutLinear(self.device,hidden_size1,hidden_size2,output_size)
        
    def forward(self, x):
        
        x=x.to(self.device)
        rnn_activity,_ = self.rnn(x)        
        out = self.fc(rnn_activity)
        return out, rnn_activity            
    
    
    
    
    
    
    
        