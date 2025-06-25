# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:57:06 2022

here use the hebbian net feedforward with feedback

@author: NZ
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import torch.optim as optim
import numpy as np

class HebbNet(nn.Module):
    def __init__(self, Nx, Nh, Ny, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbNet, self).__init__()
        
        W,b = random_weight_init([Nx,Nh,Ny],bias=True)
        W.append(np.random.randn(Nh,Ny)/np.sqrt(Ny))
            
        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float).unsqueeze(1))#shape=[Nh,1] for broadcasting
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float).t()) #shape=[Nh,Ny] pre-transposed for faster matmul
        self.b2 = nn.Parameter(torch.tensor(b[1],dtype=torch.float))
        self.w21 = nn.Parameter(torch.tensor(W[2],dtype=torch.float))
        
        self.f = f
        self.fOut = fOut
        
        self.init_hebb(**hebbArgs) #parameters of Hebbian rule
            
    def reset_state(self, batchSize=None):
        if batchSize is None:
            batchSize,_,_=self.A.shape
        self.A = torch.zeros(batchSize, *self.w1.shape) #shape=[B,Nh,Nx]
        
    def init_hebb(self, eta=None, lam=0.99):
        if eta is None:
            eta = -5./self.w1.shape[1] #eta*d = -5
        self.lam = nn.Parameter(torch.tensor(lam))
        self.eta = nn.Parameter(torch.tensor(eta))         
        
    def update_hebb(self, pre, post):
        """Updates A using a (batched) outer product, i.e. torch.ger(post, pre)
        for each of the elements in the batch
            
        pre.shape = [B,Nx] (pre.unsq.shape=[B,1,Nx])
        post.shape = [B,Nh,1]
        """  
        self.lam.data = torch.clamp(self.lam.data,max=1.)
        self.A = self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1)) #shape=[B,Nh,Nx]
                 
    def forward(self, x, a2, debug=False):
        """
        x.shape = [B,Nx]  
        
        """    
        Batch_size = x.size(0)        
        # b1.shape=[Nh,1] w1.shape=[Nh,Nx] A.shape=[B,Nh,Nx] x.shape=[B,Nx]
        a1_forward = torch.baddbmm(self.b1, self.w1+self.A, x.unsqueeze(2)) #shape = [B,Nh,1]
        # a1_forward.shape=[B,Nh,1] w21.shape=[Nh,Ny] a2.shape=[B,Ny]
        a1 = torch.baddbmm(a1_forward,self.w21.tile((Batch_size,1,1)), a2.unsqueeze(2))
        h = self.f(a1)
        self.update_hebb(x,h)
        
        #b2.shape=[Ny], h.sq.shape=[B,Nh] w2.shape=[Nh,Ny]
        a2 = torch.addmm(self.b2,h.squeeze(dim=2),self.w2) #shape=[B,Ny]
        y = self.fOut(a2)
            
        if debug:
            return a1, h, a2, y
        return a1, h, a2, y

    def evaluate(self, input,output_size):
        batchsize = input.size(1)
        self.reset_state(batchSize = batchsize)
        hiddens = []
        outputs = []
        steps = range(input.size(0))
        for i in steps:
            if i==0:
                a2 = torch.zeros(batchsize,output_size)
            a1,hidden,a2,out = self.forward(input[i],a2)
            hiddens.append(hidden)
            outputs.append(out)
        
        hiddens = torch.stack(hiddens,dim=0)
        outputs = torch.stack(outputs,dim=0)
        return hiddens,outputs        
        
    @torch.no_grad()
    def _monitor(self, trainBatch,validBatch=None, out=None, loss=None, acc=None):
        super(HebbNet, self)._monitor(trainBatch, validBatch, out, loss, acc)
        
        if hasattr(self, 'writer'):
            if self.hist['iter']%10 == 0:
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta', self.eta, self.hist['iter'])
                    
                if self.w2.numel()==1:
                    self.writer.add_scalar('params/w2',self.w2.item(), self.hist['iter'])
                if self.b1.numel()==1:
                    self.writer.add_scalar('params/b1',self.b1.item(), self.hist['iter'])
                if self.b2.numel()==1:
                    self.writer.add_scalar('params/b2',self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1',self.g1.item(), self.hist['iter'])
 
class  Net(nn.Module):
    """ Hebb network model.
    Args:
        input_size: int, input size
        hidden_size1: int, first hidden layer size
        hebb: str, type of hebb: hebb, Anti
    """                      
    def __init__(self,hp,**kwargs):
        super().__init__()
        # hidden size ?
        input_size = hp['n_input']
        hidden_size = hp['hidden_size']
        self.output_size = hp['n_output']
        # _monitor_init?
        
        self.hebb = HebbNet(input_size, hidden_size, self.output_size)
    
    def forward(self,x):
        activity, out = self.hebb.evaluate(x,self.output_size)
        return out, activity
        
                    
def check_dims(W,B=None):
    """Verify that the dimensions of the weight matrices are compatible"""
    dims = [W[0].shape[1]]
    for l in range(len(W)-1):
        assert(W[l].shape[0] == W[l+1].shape[1])
        dims.append(W[l].shape[0])
        if B:
            assert(W[l].shape[0] == B[l].shape[0])
    if B:
        assert(W[-1].shape[0]==B[-1].shape[0])
    dims.append(W[-1].shape[0])
    return dims            
            
def random_weight_init(dims,bias=False):
    W,B = [],[]
    for l in range(len(dims)-1):
        W.append(np.random.randn(dims[l+1], dims[l])/np.sqrt(dims[l]))
        if bias:
            B.append(np.random.randn(dims[l+1]))
        else:
            B.append(np.zeros(dims[l+1]))
        check_dims(W,B)
    return W,B