# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:07:21 2025
plot input and targeted output and real output
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

# import mynetwork_new3
# from mynetwork_new3 import Net

import mynetwork8
from mynetwork8 import Net

from mytools import popvec,get_y_direction
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import pearsonr


# plt.figure()
# plt.close('all')
THETA = 0.3 * np.pi

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
    
    modelparams = torch.load(model_dir,weights_only=False)
    state_dict = modelparams["state_dict"]
    hp = modelparams["hp"]
    hp["sigma_x"] = 0.01
    hp['sigma_rec1']=0.1
    # hp['sigma_rec2']=0.01
    # hp['fforwardstren']=0.1
    hp['fbackstren']=0.1
    # hp['sigma_x'] = 0.1,
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
            H2O_weight = net.fc.effective_weight()
            # H2O_weight = net.fc.weight
            H2H_weight = net.rnn.h2h.effective_weight()
            I2H_weight = net.rnn.input2h.effective_weight()
            # I2H_weight = net.rnn.input2h.weight
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
        stim1_coh = np.ones(9)*0.8
        stim1_loc = np.array([-12,-6,-1,-0.5,0,0.5,1,6,12])*6/360*np.pi+np.pi
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
    x,y,y_loc,c_mask = test_trial.x,test_trial.y,test_trial.y_loc,test_trial.c_mask

    stim1_ons = test_trial.on
    dt = test_trial.dt
    times_relate = {'stim_ons':stim1_ons,'dt':dt,'stim_dur':stim1_times}  
    
    y_dir = get_y_direction(y_hat,y_loc) 
    # only analysis trials with the output direction belongs to the given target 
    complete_trial_ind = np.where(np.isnan(y_dir)==False)[0]
    
    y_hat = y_hat.detach().numpy()
    
    selected_index = np.random.choice(complete_trial_ind) 
    
    stim1on_time = stim1_ons*dt
    stim1end_time = stim1on_time+stim1_times[0]
    stim1_end = stim1end_time/dt
    n_input_heading = hp['n_input_heading']
    n_input_targcolor = hp['n_input_targcolor']
    n_input_rules = hp['n_input_rules']
    
    # Create a custom colormap: 0 → white, higher values → darker (e.g., blue/black)
    colors = [(1, 1, 1), (0, 0, 0.5)]  # White to dark blue
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    plt.figure()
    plt.imshow(x[:,selected_index,: n_input_heading].T,cmap=cmap)
    plt.xticks(ticks=[stim1_ons,stim1_end], labels=['stim1 on','stim1 end']) 
    plt.savefig("./lunwenfigure/headinginput.svg")
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    # Optional: Keep bottom and left spines but make them thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.figure()
    plt.imshow(x[:,selected_index,n_input_heading:n_input_heading+n_input_targcolor].T,cmap=cmap)
    plt.xticks(ticks=[stim1_ons,stim1_end], labels=['stim1 on','stim1 end']) 
    plt.savefig("./lunwenfigure/colorinput.svg")
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    # Optional: Keep bottom and left spines but make them thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.figure()
    plt.imshow(x[:,selected_index,n_input_heading+n_input_targcolor:].T,cmap=cmap)
    plt.xticks(ticks=[stim1_ons,stim1_end], labels=['stim1 on','stim1 end']) 
    plt.savefig("./lunwenfigure/ruleinput.svg")
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    # Optional: Keep bottom and left spines but make them thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.figure()
    plt.imshow(y[:,selected_index,:].T,cmap=cmap)
    plt.xticks(ticks=[stim1_ons,stim1_end], labels=['stim1 on','stim1 end']) 
    plt.savefig("./lunwenfigure/expectoutput.svg")
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    # Optional: Keep bottom and left spines but make them thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    plt.figure()
    plt.imshow(y_hat[:,selected_index,:].T,cmap=cmap)
    plt.xticks(ticks=[stim1_ons,stim1_end], labels=['stim1 on','stim1 end']) 
    plt.savefig("./lunwenfigure/realoutput.svg")
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    # Optional: Keep bottom and left spines but make them thinner
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
for i in [0]:
    figname_suffix = f'checkgpu/{i}'
    model_dir = './checkpoint/checkgpu.t7'         
    
    neuralactivity_color_dm(model_dir,figname_append=figname_suffix) 