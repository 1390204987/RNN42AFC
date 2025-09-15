# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:38:01 2024

check the neural activity in batch

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
import mynetwork8
from mynetwork8 import Net

from mytools import popvec,get_y_direction

from scipy.stats import pearsonr

from tools_selectivity import _selectivity, _z_selectivity,plot_selectivity_corr
from tools_psth import plot_population
from tools_netplot import plot_input2neuron_connectivity,plot_h2h_connectivity,plot_h2output_connectivity
# plt.figure()
# plt.close('all')

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
    # hp["sigma_x"] = 0 
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
    stim_mod = 2 # 1 is fine task 2 is coarse task
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
        unique_heading = np.unique(np.sign(stim1_loc-np.pi/0.5)*stim1_coh)
 
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
    # plot_alltrial_activity(neural_activity,rule=rule)
    # plot_condtitioned_psth(neural_activity,ind_stim,condition_list,times_relate,rule=rule)
    # plot_saccade_divergence(neural_activity,ind_stim,condition_list,times_relate,y_hat,y_loc,rule=rule)
    # heading_selectivity_h1 = _selectivity(heading_per_trial,hidden1_activity,times_relate)
    heading_selectivity_h1 = _z_selectivity(heading_per_trial,choice_per_trial,hidden1_activity,times_relate)
    # saccade_selectivity_h1 = _z_selectivity(sac_per_trial,heading_per_trial,hidden1_activity,times_relate)
    saccade_selectivity_h1 = _selectivity(sac_per_trial,hidden1_activity,times_relate)
    choice_selectivity_h1 = _z_selectivity(choice_per_trial,heading_per_trial,hidden1_activity,times_relate)
    # only use 0heading stimulus condition
    # select_0heading = heading_per_trial==np.pi
    # choice_selectivity_h1 = _selectivity(choice_per_trial[select_0heading],hidden1_activity[:,select_0heading,:],times_relate)
    color_selectivity_h1 = _selectivity(T1loc_per_trial,hidden1_activity,times_relate)
    
    if not hp.get("hidden_size2") is None:
        # heading_selectivity_h2 = _selectivity(heading_per_trial,hidden2_activity,times_relate)
        heading_selectivity_h2 = _z_selectivity(heading_per_trial,choice_per_trial,hidden2_activity,times_relate)
        saccade_selectivity_h2 = _selectivity(sac_per_trial,hidden2_activity,times_relate)
        # saccade_selectivity_h2 = _z_selectivity(sac_per_trial,heading_per_trial,hidden2_activity,times_relate)
        choice_selectivity_h2 = _z_selectivity(choice_per_trial,heading_per_trial,hidden2_activity,times_relate)
        # only use 0heading stimulus condition
        # choice_selectivity_h2 = _selectivity(choice_per_trial[select_0heading],hidden2_activity[:,select_0heading,:],times_relate)
        color_selectivity_h2 = _selectivity(T1loc_per_trial,hidden2_activity,times_relate)
        neural_selectivity = {'heading_selectivity_h1':heading_selectivity_h1,
                              'saccade_selectivity_h1':saccade_selectivity_h1,
                              'choice_selectivity_h1':choice_selectivity_h1,
                              'color_selectivity_h1':color_selectivity_h1,
                              'heading_selectivity_h2':heading_selectivity_h2,
                              'saccade_selectivity_h2':saccade_selectivity_h2,
                              'choice_selectivity_h2':choice_selectivity_h2,
                              'color_selectivity_h2':color_selectivity_h2}
    return neural_selectivity

class CustomException(Exception):
    pass        
def selectivity_corr(var1_selectivity,var2_selectivity,**kwargs):
    useful_neuron_var1_ind = np.where(np.isnan(var1_selectivity[:,0])==False)
    useful_neuron_var2_ind = np.where(np.isnan(var2_selectivity[:,0])==False)
    useful_neuron_ind = np.intersect1d(useful_neuron_var1_ind[0],useful_neuron_var2_ind[0])
    
    try:
        if len(var1_selectivity[useful_neuron_ind, 0]) < 2 or len(var2_selectivity[useful_neuron_ind,0])<2:
            raise CustomException("var1_selectivity has less than 2 elements")
        var1_var2_corr = pearsonr(var1_selectivity[useful_neuron_ind,0],var2_selectivity[useful_neuron_ind,0])
    except CustomException as e:
        var1_var2_corr = (np.nan,np.nan)
        
    return var1_var2_corr

    
def list_files_in_directory(directory):

    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        
    return files
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

filespath = './checkpoint_batch13'
nets = list_files_in_directory(filespath)

df = pd.DataFrame(columns=["netid","layer1cor","layer1p", "layer2cor", "layer2p","lay1color&saccor","lay1color&saccorp","lay2color&saccor","lay2color&saccorp"])

for net in nets:
    model_dir = filespath + '/' + net
    neural_selectivity = neuralactivity_color_dm(model_dir)
    heading_selectivity_h1 = neural_selectivity['heading_selectivity_h1']
    saccade_selectivity_h1 = neural_selectivity['saccade_selectivity_h1']
    choice_selectivity_h1 = neural_selectivity['choice_selectivity_h1']
    color_selectivity_h1 = neural_selectivity['color_selectivity_h1']
    heading_selectivity_h2 = neural_selectivity['heading_selectivity_h2']
    saccade_selectivity_h2 = neural_selectivity['saccade_selectivity_h2']
    choice_selectivity_h2 = neural_selectivity['choice_selectivity_h2']
    color_selectivity_h2 = neural_selectivity['color_selectivity_h2'] 
    heading_choice_corr1 = selectivity_corr(heading_selectivity_h1,choice_selectivity_h1)   
    heading_choice_corr2 = selectivity_corr(heading_selectivity_h2,choice_selectivity_h2)
    color_sac_corr1 = selectivity_corr(color_selectivity_h1,saccade_selectivity_h1)
    color_sac_corr2 = selectivity_corr(color_selectivity_h2,saccade_selectivity_h2)
    mean_abschosele_h1 = np.mean(np.abs(choice_selectivity_h1[:,0]))
    mean_abschosele_h2 = np.mean(np.abs(choice_selectivity_h2[:,0]))
    numbersinnetname = re.findall(r'\d+',net)
    netid = numbersinnetname[0]
    
    df = df.append({"netid": netid,
                    "layer1cor": heading_choice_corr1[0],
                    "layer1p": heading_choice_corr1[1],
                    "layer2cor": heading_choice_corr2[0],
                    "layer2p": heading_choice_corr2[1],
                    "lay1color&saccor": color_sac_corr1[0],
                    "lay1color&saccorp":color_sac_corr1[1],
                    "lay2color&saccor": color_sac_corr2[0],
                    "lay2color&saccorp":color_sac_corr2[1]},ignore_index=True)
    
df.to_excel("./encoding_correlation13.xlsx",index=False)
    
    
    
    
    