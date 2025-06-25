 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:38:10 2022

here use the hebbian net

@author: NZ
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import torch.optim as optim
import numpy as np

# class NetworkBase(nn.Module):
#     def __init__(self):
#         """Subclasses must either implement self.loss_fn or override self.average_loss()"""
#         super(NetworkBase,self).__init__()
#         self.eval() #start the module in evaluation mode
#         self.hist = None
#         self.name = self.__class__.__name__
#         self.loss_fn = None
#         self.acc_fn = None
      
#     def evaluate(self,batch):
#         """batch is (X,Y) tuple of Tensors, with X.shape=[T,B,N] or [T,N], Y.shape=[T,1]"""
#         out = torch.empty_like(batch[1])
#         for t,x in enumerate(batch[0]):
#             out[t] = self(x)
#         return out
    
# class StatefulBase(NetworkBase):
#     def evaluate(self,batch,preserveState=False):
#         """NOTE: current state of A will be lost!""" 
#         self.reset_state()
#         out = super(StatefulBase,self).evaluate(batch)
#         return out

class HebbNet(nn.Module):
    def __init__(self, Nx, Nh, Ny, batchSize, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbNet, self).__init__()
        
        W,b = random_weight_init([Nx,Nh,Ny],bias=True)
            
        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float).unsqueeze(1))#shape=[Nh,1] for broadcasting
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float).t()) #shape=[Nh,Ny] pre-transposed for faster matmul
        self.b2 = nn.Parameter(torch.tensor(b[1],dtype=torch.float))
        
        self.f = f
        self.fOut = fOut
        
        self.init_hebb(**hebbArgs) #parameters of Hebbian rule
        
        # self.register_buffer('A',None)
        
        try:
            self.reset_state(batchSize=batchSize)
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in HebbFF.__init__'.format(e))
                            
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
                    
    def forward(self, x, debug=False):
        """
        x.shape = [B,Nx]  
        
        """            
        # b1.shape=[Nh,1] w1.shape=[Nh,Nx] A.shape=[B,Nh,Nx] x.shape=[B,Nx]
        a1 = torch.baddbmm(self.b1, self.w1+self.A, x.unsqueeze(2))
        h = self.f(a1)
        self.update_hebb(x,h)
        
        #b2.shape=[Ny], h.sq.shape=[B,Nh] w2.shape=[Nh,Ny]
        a2 = torch.addmm(self.b2,h.squeeze(dim=2),self.w2)
        y = self.fOut(a2)
            
        if debug:
            return a1, h, a2, y
        return h, y
        
    def evaluate(self, input):
        self.reset_state()
        hiddens = []
        outputs = []
        steps = range(input.size(0))
        for i in steps:
           hidden,out = self.forward(input[i])
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
        output_size = hp['n_output']
        batchSize = hp['batch_size_train']
        # _monitor_init?
        
        self.hebb = HebbNet(input_size, hidden_size, output_size, batchSize)
        
    def forward(self,x):
        activity, out = self.hebb.evaluate(x)
        return out, activity
        
        
def train_dataset(net, trainData, validBatch=None, epochs=100, batchSize=None, earlyStop=True, validStopThres=None, earlyStopValid=False):       
    """trainData is a TensorDataset"""
    net.monitor_init(trainData[:,0,:], validBatch=validBatch)    
    while net.hist['epoch'] < epochs:
        net.hist['epoch'] += 1
        converged = net._train_epoch(trainData, validBatch=validBatch, batchSize=batchSize, earlyStop=earlyStop,validStopThres=validStopThres,earlyStopValid=earlyStopValid)
        if converged:
            print('Converged, stopping early')
            break

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