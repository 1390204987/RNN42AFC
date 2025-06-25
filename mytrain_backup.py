# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:47:53 2021

@author: NaN
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


import  mytask

# import mynetwork
# from mynetwork import Net

import mynetwork1hidden
from mynetwork1hidden import Net

# import mynetworklstm1hidden
# from mynetworklstm1hidden import Net



def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)

def popvec_tensor(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    y_output = y[...,1:-1]
    pref = torch.arange(0, 2*np.pi, 2*np.pi/y_output.shape[-1])
    temp_sum = y_output.sum(axis=-1)
    temp_cos = torch.sum(y_output*torch.cos(pref), axis=-1)/temp_sum
    temp_sin = torch.sum(y_output*torch.sin(pref), axis=-1)/temp_sum    
    loc = torch.atan2(temp_sin, temp_cos)
    return loc

def popvec_2d_tensor(y,n_eachring):
    """Population vector read out along time.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (time,Batch, Units)

    Returns:
        Readout locations: Numpy array (time,Batch,)
    """
    y_max_value,y_max_indices = torch.max(y,dim = 2)
    fixdim = (y_max_indices==0)*1
    fixdim = fixdim.float()
    # fixdim = torch.zeros(np.shape(y_max_indices))
    saccadedim = y_max_indices/n_eachring*2*np.pi
    y_2d = (fixdim,saccadedim)
    return y_2d
    
    
        

def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    y_hat_np = y_hat.detach().numpy()
    if len(y_hat_np.shape) != 3:
        raise ValueError('y_hat_np must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat_np = y_hat_np[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat_np[..., 0]
    y_hat_loc = popvec(y_hat_np[..., 1:])

    # Fixating? Correctly saccading? (at the last time point)
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate? in HD and color target HD this value is False
    should_fix = y_loc < 0 

    # performance
    # perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    perf = np.sum((1-should_fix) * corr_loc * (1-fixating))/len(corr_loc)
    
    return perf

def train(net,device,iepoch,trial,hp,netname):
    running_loss = 0 

    x,y,y_loc,c_mask = trial.x,trial.y,trial.y_loc,trial.c_mask
    x = torch.from_numpy(x).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    y_loc_tensor = torch.from_numpy(y_loc).type(torch.float)
    c_mask = torch.from_numpy(c_mask).type(torch.float)
    inputs = x
    optimizer.zero_grad()   # zero the gradient buffers
    y_hat,activity = net(inputs)

    # h_shaped = torch.reshape(activity, (-1, n_rnn))
    y_shaped = torch.reshape(y, (-1, n_output))
        # y_hat_ shape (n_time*n_batch, n_unit)
    y_hat_shaped = torch.reshape(y_hat,(-1,n_output))
    mask_shaped = torch.reshape(c_mask,(-1,n_output))
    
    loss = torch.mean(torch.sum(torch.square(y_shaped-y_hat_shaped)*mask_shaped,1))

    perf = np.mean(get_perf(y_hat,y_loc))
    # loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    

    print('epoch {}, Loss {:0.4f}, perf {:0.4f}'.format(iepoch, running_loss, perf))
    running_loss = 0  
    
    checkpoint(net,perf, iepoch,hp, netname )
    # torch.save(net, './checkpoint/try2.pt')
    
def checkpoint(model, acc, epoch,hp, outModelName):
    print('saving...')
    state = {
        'state_dict': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state':torch.get_rng_state(),
        'hp':hp
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/{outModelName}.t7')
    
    
seed = 3
ruleset = 'all'
# num_ring = database1.get_num_ring(ruleset)
# n_rule = database1.get_num_rule(ruleset)
# num_ring = 3
num_ring = 1

n_rule = 3
n_eachring = 32
n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
hp = {
            'ruleset': ruleset,
            'rule_start': 1+num_ring*n_eachring,
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            'alpha': 0.2,
            # input noise
            'sigma_x': 0.01,
            # number of units each ring
            'n_eachring': n_eachring,
            'loss_type': 'lsq',
            'dt': 20,
            'rng': np.random.RandomState(seed),
            'easy_task': 1,
            # 'hidden_size1': 64,
            # 'hidden_size2': 64}
            # 'hidden_size3': 32}
            'hidden_size': 64}
              
mode = 'random'

     
net = Net(hp,dt = hp['dt'])
# netname = '1hiddencolorHDsignrestrict'
netname = 'delaysac'
# net = Net(hp)
# Use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

device = 'cpu'
train_trial = mytask.generate_trials('delaysaccade',hp,mode,stim_mod=1,noise_on=True,batch_size = 500)
# train_trial = mytask.generate_trials('coltargdm',hp,mode,stim_mod=1,noise_on=True,batch_size = 500)
# train_trial = mytask.generate_trials('dm',hp,mode,stim_mod=1,noise_on=True,batch_size = 125)
# for iepoch in range(50):
for iepoch in range(2):
    train(net,device,iepoch,train_trial,hp,netname)


# test_trial = mytask.generate_trials('coltargdm',hp,mode,noise_on=True,batch_size = 5)
# test_trial = mytask.generate_trials('dm',hp,mode,noise_on=True,batch_size = 5)
# test_y,test_y_hat,test_loss,test_perf = test(net,test_trial)
    
    