# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:42:42 2024
project parameter like choice corr, noise into dimension reductioned 
 submodule(represent recurrency, feedforwar and feed back) , 
@author: NaN
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import scipy.stats as stats
from scipy.optimize import curve_fit

import mytask
from mytask import generate_trials, rule_name, get_dist

# import mynetworkhebb
# from mynetworkhebb import Net
          
import mynetwork_new3
from mynetwork_new3 import Net

# import mynetwork1hidden
# from mynetwork1hidden import Net

# import mynetwork
# from mynetwork import Net

from mytools import popvec,get_y_direction

from scipy.stats import pearsonr

def _neuralactivity_dm(model_dir, rule, stim_mod, params_list, batch_shape):
    """Base function for computing psychometric performance in 2AFC tasks

    Args:
        model_dir : model name
        rule : task to analyze
        params_list : a list of parameter dictionaries used for the psychometric mode
        batch_shape : shape of each batch. Each batch should have shape (n_rep, ...)
        n_rep is the number of repetitions that will be averaged over

    Return:
        ydatas: list of performances
    """
    print('Starting neural activity analysis of the {:s} task...'.format(rule_name[rule]))
    
    modelparams = torch.load(model_dir)
    state_dict = modelparams["state_dict"]
    hp = modelparams["hp"]
    hp["sigma_x"] = 0 
    net = Net(hp,dt = hp['dt'])
    #remove prefixe "module"
    state_dict = {k.replace("module.",""): v for k, v in state_dict.items()}
    msg = net.load_state_dict(state_dict, strict=False)
    print("Load pretrained model with msg: {}".format(msg))
 
    ydatas = list()
    for params in params_list:
        test_trial = generate_trials(rule,hp,'psychometric',stim_mod, params = params)
        x,y,y_loc,c_mask = test_trial.x,test_trial.y,test_trial.y_loc,test_trial.c_mask
        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.float)    
        c_mask = torch.from_numpy(c_mask).type(torch.float)
        inputs = x
        y_hat,activity = net(inputs)    
        
        if hasattr(net,'hebb'):
            I2H_weight = net.hebb.w1 
            H2O_weight = net.hebb.w2
            effective_weight = \
                {'I2H_weight':I2H_weight,
                 'H2O_weight':H2O_weight}
        else:            
            # H2O_weight = net.fc.effective_weight()
            H2O_weight = net.fc.weight
            H2H_weight = net.rnn.h2h.effective_weight()
            # I2H_weight = net.rnn.input2h.effective_weight()
            I2H_weight = net.rnn.input2h.weight
            effective_weight = \
                {'I2H_weight':I2H_weight,
                 'H2H_weight':H2H_weight,
                 'H2O_weight':H2O_weight}
            
        
    # e_size = net.rnn.e_size
    # return activity, test_trial, state_dict ,y_hat,y_loc,e_size,
    return activity, test_trial, state_dict ,y_hat,y_loc,effective_weight,hp

def neuralactivity_color_dm(model_dir,**kwargs):
    rule = 'coltargdm'
    stim_mod = 1 # 1 is fine task 2 is coarse task
    if stim_mod == 1:
        stim1_coh = np.ones(13)*0.1
        stim1_loc = np.array([-12,-9,-6,-4.5,-3,-1.5,0,1.5,3,4.5,6,9,12])*6/360*np.pi+np.pi
        # stim1_loc = np.array([-25,-24,-23,23,24,25])*6/360*np.pi+np.pi
        n_rep = 50
        unique_n_stim = len(stim1_loc)
    elif stim_mod ==2:
        stim1_coh = np.array([0.5,0.15,0.05,0,0.05,0.15,0.5])*0.05
        stim1_loc = np.array([0,0,0,0,np.pi,np.pi,np.pi])
        n_rep = 300
        unique_n_stim = (len(stim1_coh)-1)*len(stim1_loc)+1
    batch_size = n_rep*unique_n_stim
    batch_shape = (n_rep,unique_n_stim)
    condition_list = {'stim_coh':stim1_coh,'stim_loc':stim1_loc}
    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)
    
    stim1_locs = stim1_loc[ind_stim]
    stim1_strengths = stim1_coh[ind_stim]
    seed = 3
    rng = np.random.RandomState(seed)
    # stim2_locs = rng.choice([np.pi,2*np.pi],(batch_size,))
    stim2_locs = rng.choice([np.pi,0],(batch_size,))
    stim3_locs = (stim2_locs+np.pi)%(2*np.pi)
    
    params_list = list()
    stim1_times = [1000]
    
    for stim1_time in stim1_times:
        params = {'stim1_locs': stim1_locs,
                  'stim1_strengths': stim1_strengths,
                  'stim_time': stim1_time,
                  'stim2_locs': stim2_locs,
                  'stim3_locs': stim3_locs}
        
        params_list.append(params)
        
    if stim_mod == 1:
        xdatas = [stim1_loc]
    elif stim_mod == 2:
        xdatas = [stim1_coh]
    neural_activity,test_trial,state_dict,y_hat,y_loc,effective_weight,hp = _neuralactivity_dm(model_dir, rule,stim_mod, params_list, batch_shape)    
    # neural_activity,test_trial,state_dict,y_hat,y_loc,e_size = _neuralactivity_dm(model_dir, rule,stim_mod, params_list, batch_shape)
    neural_activity = neural_activity.detach().numpy()
    stim1_ons = test_trial.ons
    dt = test_trial.dt
    times_relate = {'stim_ons':stim1_ons,'dt':dt,'stim_dur':stim1_times}  

    y_dir = get_y_direction(y_hat,y_loc) 
    # only analysis trials with the output direction belongs to the given target 
    complete_trial_ind = np.where(np.isnan(y_dir)==False)   
        
    neural_activity = np.squeeze(neural_activity[:,complete_trial_ind,:]) 
    if hp.get("hidden_size1") is None:
        hidden1_size = hp['hidden_size']
    else: 
        hidden1_size = hp['hidden_size1']  
    hidden1_activity = neural_activity[:,:,:hidden1_size]
    if not hp.get("hidden_size2") is None:
        hidden2_size = hp['hidden_size2']        
        hidden2_activity = neural_activity[:,:,hidden1_size:]
        
    stim_on = np.unique(stim1_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    times_relate = {'stim_on':stim_on,'stim_dur':stim_dur,'stim_end':stim_end,
                    'time_steps':time_steps,'time_begin_ind':time_begin_ind,'time_end_ind':time_end_ind}
    return hidden1_activity, hidden2_activity, times_relate

filespath = './checkpoint_batch4'
net = '0000colorhdnet4.t7'
model_dir = filespath+'/'+net
modelparams = torch.load(model_dir)
state_dict = modelparams["state_dict"]
hp = modelparams["hp"]
net = Net(hp,dt = hp['dt'])
#remove prefixe "module"
state_dict = {k.replace("module.",""): v for k, v in state_dict.items()}
msg = net.load_state_dict(state_dict, strict=False)
print("Load pretrained model with msg: {}".format(msg))

H2O_weight = net.fc.weight
H2H_weight = net.rnn.h2h.effective_weight()
# I2H_weight = net.rnn.input2h.effective_weight()
I2H_weight = net.rnn.input2h.weight  
effective_weight = \
    {'I2H_weight':I2H_weight,
     'H2H_weight':H2H_weight,
     'H2O_weight':H2O_weight}
h1_num = hp['hidden_size1']
h2_num = hp['hidden_size2']
wh2h = effective_weight['H2H_weight']
wh2h = wh2h.detach().numpy()
wh1recur = wh2h[0:h1_num,1:h1_num]
wh2recur = wh2h[h1_num:h2_num,h1_num:h2_num]
wh12h2 = wh2h[h1_num:h2_num,0:h1_num]
wh22h1 = wh2h[0:h1_num,h1_num:h2_num]
hidden1_activity,hidden2_activity,times_relate = neuralactivity_color_dm(model_dir)
# project hidden1 and hidden2 activity on relative connection








