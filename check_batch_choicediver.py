# -*- coding: utf-8 -*-
"""
Created on Tue May 13 08:43:46 2025
for batch calculate choi divergence signal start time
@author: NaN
"""
import os
import re
import pandas as pd
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
import mynetwork4
from mynetwork4 import Net

from mytools import popvec,get_y_direction

from scipy.stats import pearsonr

from tools_divergence import get_divergence

################ Psychometric - Varying Coherence #############################
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


    # get heading per trial
    if len(np.unique(stim1_coh))==1:
        unique_heading = np.unique(stim1_loc)
    else:
        unique_heading = np.unique(np.sign(stim1_loc-np.pi/2)*stim1_coh)


    in_stim = ind_stim[complete_trial_ind]
    heading_per_trial = np.full_like(in_stim,np.nan,dtype=np.float32)
    for i_heading in range(len(unique_heading)): 
        heading_per_trial[np.where(in_stim==i_heading)] = unique_heading[i_heading]

    # get T1 loc per trial(color per trial)
    T1_locs = stim2_locs
    T2_locs = stim3_locs
    T1_locs = T1_locs[complete_trial_ind]
    T2_locs = T2_locs[complete_trial_ind]

    # in lab experiment, define red color on RF(left screen) as 1
    # T1loc_per_trial = -T1_locs# -pi,0 two fixed spacial loc
    T1loc_per_trial = -T1_locs# red target(T1) on left marker as 1 

    # get choice per trial
    # choose T1 mark as 1 choose T2 mark as 0,other direction mark as nan
    y_hat = y_hat.detach().numpy()
    y_hat_end = y_hat[-1]
    y_hat_loc = popvec(y_hat_end[..., 1:])
    y_hat_loc = y_hat_loc[complete_trial_ind]
    choice_per_trial = np.full_like(y_hat_loc,np.nan) 
    choice_per_trial[np.where(np.abs(y_hat_loc-T1_locs)<=0.2*np.pi)] = 1
    choice_per_trial[np.where(np.abs(np.abs(y_hat_loc-T1_locs)-2*np.pi)<=0.2*np.pi)] = 1
    choice_per_trial[np.where(np.abs(y_hat_loc-T2_locs)<=0.2*np.pi)] = 0
    choice_per_trial[np.where(np.abs(np.abs(y_hat_loc-T2_locs)-2*np.pi)<=0.2*np.pi)] = 0

    # get sac direction per trial
    sac_per_trial = y_dir[complete_trial_ind] 


    neural_activity = np.squeeze(neural_activity[:,complete_trial_ind,:]) 
    if hp.get("hidden_size1") is None:
        hidden1_size = hp['hidden_size']
    else: 
        hidden1_size = hp['hidden_size1']
    hidden1_activity = neural_activity[:,:,:hidden1_size]
    if not hp.get("hidden_size2") is None:
        hidden2_size = hp['hidden_size2']        
        hidden2_activity = neural_activity[:,:,hidden1_size:]

    # only use 0heading stimulus condition
    select_0heading = heading_per_trial==np.pi
    sacdir_list = np.sign(sac_per_trial)
    
    h1para = {'color':[0,0,1]}
    if not hp.get("hidden_size2") is None:
        h2para = {'color':[1,0,0]}
    # plt.figure(11,figsize=(10, 16))# plot saccade divergence
    plt.figure(figsize=(10, 6))  # 宽度为10，高度为5
    figname = 'saccade_div'
    plt.title('saccade_div')
    L1_sactime = get_divergence(sacdir_list[select_0heading],neural_activity[:,select_0heading,:hidden1_size],times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1para)
    if not hp.get("hidden_size2") is None:
        L2_sactime = get_divergence(sacdir_list[select_0heading],neural_activity[:,select_0heading,hidden1_size:],times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2para)
        
    
    plt.figure(figsize=(10, 6))
    figname = 'choice_div'
    plt.title('choice_div')
    # plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,:hidden1_size],times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    heading0_choice = choice_per_trial[select_0heading]
    heading0_h1_activity = neural_activity[:,select_0heading,:hidden1_size]
    heading0_h2_activity = neural_activity[:,select_0heading,hidden1_size:]
    L1_choicetime = get_divergence(heading0_choice,heading0_h1_activity,times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1para)
    if not hp.get("hidden_size2") is None:
        # plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,hidden1_size:],times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
        L2_choicetime = get_divergence(heading0_choice,heading0_h2_activity,times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2para)   
    return L1_sactime,L2_sactime,L1_choicetime,L2_choicetime
        
def list_files_in_directory(directory):
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]        
    return files
                      
filespath = './checkpoint_batch13'
nets = list_files_in_directory(filespath)

df = pd.DataFrame(columns=["netid","L1_sactime","L2_sactime", "L1_choicetime", "L2_choicetime"])

for net in nets:
    model_dir = filespath + '/' + net
    L1_sactime,L2_sactime,L1_choicetime,L2_choicetime = neuralactivity_color_dm(model_dir)
    numbersinnetname = re.findall(r'\d+',net)
    netid = numbersinnetname[0]
    
    df = df.append({"netid": netid,
                    "L1_sactime": L1_sactime,
                    "L2_sactime": L2_sactime,
                    "L1_choicetime": L1_choicetime,
                    "L2_choicetime": L2_choicetime},ignore_index=True)

df.to_excel("./encoding_correlation13.xlsx",index=False)
    