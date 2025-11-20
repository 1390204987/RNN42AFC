# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:41:38 2025
compare different batch net behavior difference
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
from roc_tool import rocN

import mytask
from mytask import generate_trials, rule_name, get_dist

# import mynetwork_new3
# from mynetwork_new3 import Net
import mynetwork8
from mynetwork8 import Net

from mytools import popvec,get_y_direction
from myperformance import plot_psychometric_choice

from scipy.stats import pearsonr

from tools_divergence import plot_divergence, plot_conditioned_divergence, get_divergence

from scipy.stats import pearsonr


THETA = 0.3 * np.pi

# From sns.dark_palette("light blue", 3, input="xkcd")
BLUES = [np.array([0.13333333, 0.13333333, 0.13333333, 1.        ]),
         np.array([0.3597078 , 0.47584775, 0.56246059, 1.        ]),
         np.array([0.58431373, 0.81568627, 0.98823529, 1.        ])]


################ Psychometric - Varying Coherence #############################
def _neuralactivity_dm(model_dir, rule, stim_mod, params_list, batch_shape,device):
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
    # hp["sigma_x"] = 0.1
    # hp['sigma_rec1']=0.1
    # hp['sigma_rec2']=0.1
    # # hp['fforwardstren']=0.1
    # hp['fbackstren']=0
    # hp['sigma_x'] = 0.1,
    net = Net(hp,device,dt = hp['dt']).to(device)
    #remove prefixe "module"
    state_dict = {k.replace("module.",""): v for k, v in state_dict.items()}
    msg = net.load_state_dict(state_dict, strict=False)
    print("Load pretrained model with msg: {}".format(msg))
 
    ydatas = list()
    for params in params_list:
        test_trial = generate_trials(rule,hp,device,'psychometric',stim_mod, params = params)
        x,y,y_loc,c_mask = test_trial.x,test_trial.y,test_trial.y_loc,test_trial.c_mask
        # x = torch.from_numpy(x).type(torch.float)
        # y = torch.from_numpy(y).type(torch.float)    
        # c_mask = torch.from_numpy(c_mask).type(torch.float)
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
    return activity, test_trial, state_dict, y_hat,y_loc,effective_weight,hp


def neuralactivity_color_dm(model_dir,device,**kwargs):
    rule = 'coltargdm'
    stim_mod = 2 # 1 is fine task 2 is coarse task
    if stim_mod == 1:
        stim1_coh = np.ones(9)*1.5
        stim1_loc = np.array([-12,-6,-1,-0.5,0,0.5,1,6,12])*6/360*np.pi+np.pi
        # stim1_loc = np.array([-25,-24,-23,23,24,25])*6/360*np.pi+np.pi
        n_rep = 50
        unique_n_stim = len(stim1_loc)
        condition_list = {'stim_coh':stim1_coh,'stim_loc':stim1_loc}
    elif stim_mod ==2:
        unique_stim1_coh = np.array([1,0.5,0.1,0.05,0])
        unique_stim1_loc = np.array([-12,12])*6/360*np.pi+np.pi
        relative_stim1_loc = np.outer(unique_stim1_coh, np.sign(unique_stim1_loc - np.pi)).flatten()+np.pi
        stim1_coh = np.repeat(unique_stim1_coh, 2)
        stim1_loc = np.tile(unique_stim1_loc, 5) 
        relative_stim1_loc = stim1_coh*np.sign(stim1_loc-np.pi)+np.pi
        n_rep = 50
        unique_n_stim = len(unique_stim1_coh)*len(unique_stim1_loc)
        condition_list = {'stim_coh':unique_stim1_coh,'stim_loc':unique_stim1_loc}
    batch_size = n_rep*unique_n_stim
    batch_shape = (n_rep,unique_n_stim)

    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)
    
    stim1_locs = stim1_loc[ind_stim]
    stim1_strengths = stim1_coh[ind_stim]
    seed = 21
    rng = np.random.RandomState(seed)
    # stim2_locs = rng.choice([np.pi,0],(batch_size,))
    half_size = batch_size // 2
    stim2_locs = np.concatenate([np.zeros(half_size), np.ones(half_size) * np.pi])
    rng.shuffle(stim2_locs)
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
        xdatas = [relative_stim1_loc]
    neural_activity,test_trial,state_dict,y_hat,y_loc,effective_weight,hp = _neuralactivity_dm(model_dir, rule,stim_mod, params_list, batch_shape,device)    
    # neural_activity,test_trial,state_dict,y_hat,y_loc,e_size = _neuralactivity_dm(model_dir, rule,stim_mod, params_list, batch_shape)
    neural_activity = neural_activity.detach().cpu().numpy()
    stim1_on = test_trial.on
    dt = test_trial.dt
    times_relate = {'stim_ons':stim1_on,'dt':dt,'stim_dur':stim1_times}  
        
    X = test_trial.x
    y_dir = get_y_direction(y_hat,y_loc) 
    # only analysis trials with the output direction belongs to the given target 
    y_dir = y_dir.detach().cpu().numpy()
    complete_trial_ind = np.where(np.isnan(y_dir)==False)
    y_hat = y_hat.detach().cpu().numpy()
    y_loc = y_loc.detach().cpu().numpy()
    y_hat_end = y_hat[-1]
    y_loc_end = y_loc[-1]
    correct_rate = np.sum((y_loc_end-y_dir)==0)/len(y_dir)
    y_hat_loc = popvec(y_hat_end[..., 1:])
    y_hat_re = np.reshape(y_hat_loc,batch_shape)
    T1_loc = np.reshape(params['stim2_locs'],batch_shape)
    T2_loc = np.reshape(params['stim3_locs'],batch_shape)    
    choose_T1 = (get_dist(y_hat_re-T1_loc)<THETA).sum(axis=0)
    choose_T2 = (get_dist(y_hat_re-T2_loc)<THETA).sum(axis=0)   
    ydatas = list()
    ydatas.append(choose_T1/(choose_T1+choose_T2))
    correct_rate = np.sum((y_loc_end-y_dir)==0)/len(y_dir)
    bias, threshold = plot_psychometric_choice(xdatas,ydatas,
                             labels=[str(t) for t in stim1_times],
                             colors=BLUES,
                             legtitle='Stim. time(ms)', rule=rule,**kwargs)
    
    #plot psychometric according to saccade direction
    y_dir_reshape = np.reshape(y_dir,batch_shape)
    sac_left = (y_dir_reshape == 0).sum(axis=0)
    sac_right = (np.float32(y_dir_reshape) == np.pi).sum(axis=0)
    sac_ydatas = list()
    sac_ydatas.append(sac_left/(sac_left+sac_right))
    sac_bias,sac_threshold = plot_psychometric_choice(xdatas,sac_ydatas,
                             labels=[str(t) for t in stim1_times],
                             colors=BLUES,
                             legtitle='Stim. time(ms)', rule=rule,**kwargs)
    
    return bias, threshold, sac_bias, sac_threshold, xdatas, ydatas, sac_ydatas, correct_rate

def list_files_in_directory(directory):
    files = [file for file in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, file)) and file.endswith('.t7')]        
    return files

filespath = './checkpoint_batchnew2'
# filespath = './check'
nets = list_files_in_directory(filespath)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_behavior_bias = pd.DataFrame()
df_behavior_threshold = pd.DataFrame()
df_behavior_sacbias = pd.DataFrame()
df_behavior_sacthreshold = pd.DataFrame()
df_xdatas = pd.DataFrame()
df_ydatas = pd.DataFrame()
df_sacydatas = pd.DataFrame()
df_correct_rate = pd.DataFrame()

for net in nets:
    model_dir = filespath + '/' + net
    bias, threshold, sac_bias, sac_threshold, xdatas, ydatas, sac_ydatas,correct_rate = neuralactivity_color_dm(model_dir, device)

    numbersinnetname = re.findall(r'\d+', net)
    netid = numbersinnetname[0]

    # 为每个net创建列名
    col_name = f"net_{netid}"
    
    df_correct_rate[col_name] = [correct_rate]
    df_behavior_bias[col_name] = [bias]
    df_behavior_threshold[col_name] = [threshold]
    
    df_behavior_sacbias[col_name] = [sac_bias]
    df_behavior_sacthreshold[col_name] = [sac_threshold]
    
    df_xdatas[col_name] = xdatas[0]
    df_ydatas[col_name] = ydatas[0]
    df_sacydatas[col_name] = sac_ydatas[0]
    
# 保存到Excel的两个不同sheet中
with pd.ExcelWriter("./behavior_batch2.xlsx") as writer:
    df_correct_rate.to_excel(writer, sheet_name='correct_rate', index=False)
    df_behavior_bias.to_excel(writer, sheet_name='behavior_bias', index=False)
    df_behavior_threshold.to_excel(writer, sheet_name='behavior_threshold', index=False)      
    df_behavior_sacbias.to_excel(writer, sheet_name='behavior_sacbias', index=False)
    df_behavior_sacthreshold.to_excel(writer, sheet_name='behavior_sacthreshold', index=False)    
     
    df_xdatas.to_excel(writer, sheet_name='behavior_xdatas', index=False)    
    df_ydatas.to_excel(writer, sheet_name='behavior_ydatas', index=False)   
    df_sacydatas.to_excel(writer, sheet_name='behavior_sacydatas', index=False) 
    
    