# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:09:24 2022

@author: NaN
"""
"""
This file contains functions that check the neuron activity
"""

import numpy as np
import matplotlib.pyplot as plt
from SetFigure import SetFigure
import torch
import torch.nn as nn
import torch.optim as optim

import scipy.stats as stats
from scipy.optimize import curve_fit

import mytask
from mytask import generate_trials, rule_name, get_dist

# import mynetworkhebb
# from mynetworkhebb import Net
          
# import mynetwork_new3
# from mynetwork_new3 import Net
import mynetwork8
from mynetwork8 import Net

# import mynetwork1hidden
# from mynetwork1hidden import Net

# import mynetwork
# from mynetwork import Net`

from mytools import popvec,get_y_direction
from myperformance import plot_psychometric_choice
from tools_divergence import plot_divergence, plot_conditioned_divergence, get_divergence

from scipy.stats import pearsonr

from tools_selectivity import _selectivity, _z_selectivity,plot_selectivity_corr
from tools_psth import plot_population
from tools_netplot import plot_input2neuron_connectivity,plot_h2h_connectivity,plot_h2output_connectivity,plot_forward_connectivity
# plt.figure()
# plt.close('all')
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
    # hp['fforwardstren']=1
    hp['fbackstren']=0
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

def neuralactivity_delaysac(model_dir,device,**kwargs):
    rule = 'delaysaccade'
    stim_mod = 1
    stim_loc = np.linspace(0,315,num=8)/180*np.pi
    n_rep = 8
    unique_n_stim = len(stim_loc)
    batch_size = n_rep*unique_n_stim
    batch_shape = (n_rep,unique_n_stim)
    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)
    
    stim_locs = stim_loc[ind_stim]
    
    params_list = list()
    stim_times = [1200]
    
    for stim_time in stim_times:
        params = {'stim_locs':stim_locs,
                  'stim_time':stim_time}
        
        params_list.append(params)
        
    neural_activity,test_trial,state_dict,y_hat,y_loc,effective_weight,hp = _neuralactivity_dm(
        model_dir, rule,stim_mod, params_list, batch_shape,device)
    neural_activity = neural_activity.detach().numpy()
    stim_ons = test_trial.ons
    dt = test_trial.dt
    times_relate = {'stim_ons':stim_ons,'dt':dt,'stim_dur':stim_times}
    
    input_list = ind_stim
    # plot_input2neuron_connectivity(state_dict,heading_selectivity=None,rule=rule,figname_append = kwargs['figname_append'])    
    # plot_h2h_connectivity(state_dict,heading_selectivity=None,saccade_selectivity=None,rule=rule,figname_append = kwargs['figname_append'])    
    plot_h2output_connectivity(effective_weight,[],rule_name,rule=rule,figname_append = kwargs['figname_append'])    
    # plot_population(neural_activity,input_list,times_relate)
    
def neuralactivity_dm(model_dir,device,**kwargs):
    rule = 'dm'
    stim_mod = 1   # 1 is fine task 2 is coarse task
    if stim_mod == 1:
        stim_coh = np.ones(8)*1.8
        stim_loc = np.array([-12,-6,-3,0,1,3,6,12])*3/360*np.pi+np.pi
        # stim_loc = np.array([-6,-4.5,-4,-3,-2,-1.5,0])*3/360*np.pi+np.pi
        n_rep = 50
        unique_n_stim = len(stim_loc)
    elif stim_mod == 2:
        stim_coh = np.array([0.05, 0.15, 0.5, 0, 0.05, 0.15, 0.5])*0.05
        stim_loc = np.array([0, 0, 0, 0, np.pi, np.pi, np.pi])
        n_rep = 300
        unique_n_stim = (len(stim_coh)-1)*len(stim_loc)+1
    batch_size = n_rep*unique_n_stim
    batch_shape =  (n_rep,unique_n_stim)
    condition_list = {'stim_coh':stim_coh,'stim_loc':stim_loc}
    
    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)   
    
    stim_locs = stim_loc[ind_stim]
    stim_strengths = stim_coh[ind_stim]
    
    params_list = list()
    stim_times = [1000]
    
    for stim_time in stim_times:
        params = {'stim_locs': stim_locs,
                  'stim_strengths': stim_strengths,
                  'stim_time': stim_time}
        
        params_list.append(params)
    
    if stim_mod == 1:
        xdatas = [stim_loc]
    elif stim_mod == 2:
        xdatas = [stim_coh]
        
        
    neural_activity,test_trial,state_dict,y_hat,y_loc,effective_weight,hp = _neuralactivity_dm(model_dir, rule,stim_mod, params_list, batch_shape)
    # neural_activity,test_trial,state_dict,y_hat,y_loc,e_size = _neuralactivity_dm(model_dir, rule,stim_mod, params_list, batch_shape)
    neural_activity = neural_activity.detach().numpy()
    stim_ons = test_trial.on
    dt = test_trial.dt
    times_relate = {'stim_ons':stim_ons,'dt':dt,'stim_dur':stim_times}
    
    y_dir = get_y_direction(y_hat,y_loc) 
    # only analysis trials with the output direction belongs to the given target 
    complete_trial_ind = np.where(np.isnan(y_dir)==False)
    y_hat = y_hat.detach().numpy()
    y_hat_end = y_hat[-1]
    y_hat_loc = popvec(y_hat_end[..., 1:])
    y_hat_re = np.reshape(y_hat_loc,batch_shape)
    T2_loc = np.zeros(batch_shape)
    T1_loc = T2_loc + np.pi 
    choose_T1 = (get_dist(y_hat_re-T1_loc)<THETA).sum(axis=0)
    choose_T2 = (get_dist(y_hat_re-T2_loc)<THETA).sum(axis=0)   
    ydatas = list()
    ydatas.append(choose_T1/(choose_T1+choose_T2))
    plot_psychometric_choice(xdatas,ydatas,
                             labels=[str(t) for t in stim_times],
                             colors=BLUES,
                             legtitle='Stim. time(ms)', rule=rule,**kwargs)
    
    # get heading per trial
    if len(np.unique(stim_coh))==1:
        unique_heading = np.unique(stim_loc)
    else:
        unique_heading = np.unique(np.sign(stim_loc)*stim_coh)
 
    in_stim = ind_stim[complete_trial_ind]
    heading_per_trial = np.full_like(in_stim,np.nan,dtype=np.float32)
    for i_heading in range(len(unique_heading)): 
        heading_per_trial[np.where(in_stim==i_heading)] = unique_heading[i_heading]
        
    # get sac direction per trial 
    # original for heading<0,the pi direction is correct choice
    sac_per_trial = y_dir[complete_trial_ind] 
    
    neural_activity = np.squeeze(neural_activity[:,complete_trial_ind,:]) 
    if hp.get('hidden_size1'):
        hidden1_size = hp['hidden_size1']
    else:
        hidden1_size = hp['hidden_size'] 
    hidden1_activity = neural_activity[:,:,:hidden1_size]
    if not hp.get("hidden_size2") is None:
        hidden2_size = hp['hidden_size2']
        hidden2_activity = neural_activity[:,:,hidden1_size:]
    # plot_alltrial_activity(neural_activity,rule=rule,figname_append = kwargs['figname_append'])
    # plot_condtitioned_psth(neural_activity,ind_stim,condition_list,times_relate,rule=rule,figname_append = kwargs['figname_append'])
    heading_selectivity_h1 = _z_selectivity(heading_per_trial,sac_per_trial,hidden1_activity,times_relate)
    # saccade_selectivity_h1 = _z_selectivity(sac_per_trial,heading_per_trial,hidden1_activity,times_relate)    # plot_corr_heading_saccade(heading_selectivity,saccade_selectivity,rule=rule,figname_append = kwargs['figname_append'])
    # only use 0 heading stimulus condition
    select_0heading = heading_per_trial==unique_heading[3]
    saccade_selectivity_h1 = _selectivity(sac_per_trial[select_0heading],hidden1_activity[:,select_0heading,:],times_relate)
    if not hp.get("hidden_size2") is None:
        heading_selectivity_h2 = _z_selectivity(heading_per_trial,sac_per_trial,hidden2_activity,times_relate)
        # only use 0 heading stimulus condition
        select_0heading = heading_per_trial==unique_heading[3]
        saccade_selectivity_h2 = _selectivity(sac_per_trial[select_0heading],hidden2_activity[:,select_0heading,:],times_relate)    # plot_corr_heading_saccade(heading_selectivity,saccade_selectivity,rule=rule,figname_append = kwargs['figname_append'])    
    h1para = {'color':[0,0,1],'marker':'.','text_y':0.8} #blue
    if not hp.get("hidden_size2") is None:
        h2para = {'color':[1,0,0],'marker':'.','text_y':0.6} #red
      
    plt.figure()
    figname = 'heading_vs_sac'
    plt.xlabel('heading selectivity')
    plt.ylabel('saccade selectivity')
    plt.title('heading selectivity vs. saccade selectivity')
    plot_selectivity_corr(heading_selectivity_h1,saccade_selectivity_h1,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(heading_selectivity_h2,saccade_selectivity_h2,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
    # plot_saccade_divergence(neural_activity,ind_stim,condition_list,times_relate,y_hat,y_loc,rule=rule,figname_append = kwargs['figname_append'])

    # E_neural_activity = neural_activity[:, :, :e_size]
    plot_input2neuron_connectivity(effective_weight,[],rule_name,rule=rule,figname_append = kwargs['figname_append']) 
    if not effective_weight.get("H2H_weight") is None:
        plot_h2h_connectivity(effective_weight,[],[],rule_name,rule=rule,figname_append = kwargs['figname_append'])    
    plot_h2output_connectivity(effective_weight,[],rule_name,rule=rule,figname_append = kwargs['figname_append'])    
    heading0_saccade = sac_per_trial[select_0heading]
    heading0_h1_activity = neural_activity[:,select_0heading,:hidden1_size]
    heading0_h2_activity = neural_activity[:,select_0heading,hidden1_size:]
    plt.figure(figsize=(10, 6))
    figname = 'choice_div'
    plt.title('choice_div')
    plot_divergence(heading0_saccade,heading0_h1_activity,times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1para)
    if not hp.get("hidden_size2") is None:
        # plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,hidden1_size:],times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
        plot_divergence(heading0_saccade,heading0_h2_activity,times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2para)

def neuralactivity_color_dm(model_dir,device,**kwargs):
    rule = 'coltargdm'
    stim_mod = 2# 1 is fine task 2 is coarse task
    if stim_mod == 1:
        stim1_coh = np.ones(9)*1.5
        stim1_loc = np.array([-12,-6,-1,-0.5,0,0.5,1,6,12])*6/360*np.pi+np.pi
        # stim1_loc = np.array([-25,-24,-23,23,24,25])*6/360*np.pi+np.pi
        n_rep = 50
        unique_n_stim = len(stim1_loc)
        condition_list = {'stim_coh':stim1_coh,'stim_loc':stim1_loc}
    elif stim_mod ==2:
        unique_stim1_coh = np.array([1,0.5,0.1,0.05,0])
        unique_stim1_coh  = unique_stim1_coh*1
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
    plot_psychometric_choice(xdatas,ydatas,
                             labels=[str(t) for t in stim1_times],
                             colors=BLUES,
                             legtitle='Stim. time(ms)', rule=rule, correct_rate=correct_rate,**kwargs)
    plt.savefig("./lunwenfigure/psy_color.svg")
    #plot psychometric according to saccade direction
    y_dir_reshape = np.reshape(y_dir,batch_shape)
    sac_left = (y_dir_reshape == 0).sum(axis=0)
    sac_right = (np.float32(y_dir_reshape) == np.pi).sum(axis=0)
    sac_ydatas = list()
    sac_ydatas.append(sac_left/(sac_left+sac_right))
    plot_psychometric_choice(xdatas,sac_ydatas,
                             labels=[str(t) for t in stim1_times],
                             colors=BLUES,
                             legtitle='Stim. time(ms)', rule=rule,**kwargs)
    plt.savefig("./lunwenfigure/psy_dir.svg")
    #plot psychometric according to saccaade direction respect T1 loc
    # plot_psychometric_choice(T1_loc,sac_ydatas,
    #                          labels=[str(t) for t in stim1_times],
    #                          colors=BLUES,
    #                          legtitle='Stim. time(ms)', rule=rule,**kwargs)
    
    # get heading per trial
    if len(np.unique(stim1_coh))==1:
        unique_heading = np.unique(stim1_loc)
    else:
        unique_heading = relative_stim1_loc
 
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
    T1loc_per_trial = T1_locs# red target(T1) on left marker as 1 
    # get choice per trial
    # choose T1 mark as 1 choose T2 mark as 0,other direction mark as nan

    y_hat_loc = y_hat_loc[complete_trial_ind]
    choice_per_trial = np.full_like(y_hat_loc,np.nan) 
    choice_per_trial[get_dist(y_hat_loc-T1_locs)<THETA] = 1   
    choice_per_trial[get_dist(y_hat_loc-T2_locs)<THETA] = 0
    
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
    target_headings = np.array([-0.5, 0, 0.5]) * (6/360) * np.pi + np.pi  # 转换为弧度
    select_0heading = np.any(np.isclose(heading_per_trial, target_headings[:, None]), axis=0)
    # select_0heading = heading_per_trial==np.array([-0.5,0,0.5])*6/360*np.pi+np.pi
    select_T10 = T1loc_per_trial==0
    select_sac0 = sac_per_trial==0
    # plot_alltrial_activity(neural_activity,rule=rule)
    # plot_condtitioned_psth(neural_activity,ind_stim,condition_list,times_relate,rule=rule)
    # plot_saccade_divergence(neural_activity,ind_stim,condition_list,times_relate,y_hat,y_loc,rule=rule)
    # heading_selectivity_h1 = _selectivity(heading_per_trial,hidden1_activity,times_relate)
    heading_selectivity_h1 = _z_selectivity(heading_per_trial,choice_per_trial,hidden1_activity,times_relate)
    saccade_selectivity_h1 = _z_selectivity(sac_per_trial,heading_per_trial,hidden1_activity,times_relate)
    # saccade_selectivity_h1 = _selectivity(sac_per_trial,hidden1_activity,times_relate)
    # saccade_selectivity_h1 = _selectivity(sac_per_trial[select_0heading&select_T10],hidden1_activity[:,select_0heading&select_T10,:],times_relate)
    # choice_selectivity_h1 = _z_selectivity(choice_per_trial,heading_per_trial,hidden1_activity,times_relate)

    heading0_choice = choice_per_trial[select_0heading]
    choice_selectivity_h1 = _selectivity(heading0_choice,hidden1_activity[:,select_0heading,:],times_relate)
    # color_selectivity_h1 = _selectivity(T1loc_per_trial[select_0heading&select_sac0],hidden1_activity[:,select_0heading&select_sac0,:],times_relate)
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
    
    h1para = {'color':[0,0,1],'marker':'.','text_y':0.8}
    if not hp.get("hidden_size2") is None:
        h2para = {'color':[1,0,0],'marker':'.','text_y':0.6}
        
    # plt.close('all')    
    plt.figure()
    figname = 'heading_vs_sac'
    plt.xlabel('heading selectivity')
    plt.ylabel('saccade selectivity')
    plt.title('heading selectivity vs. saccade selectivity')
    plot_selectivity_corr(heading_selectivity_h1,saccade_selectivity_h1,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(heading_selectivity_h2,saccade_selectivity_h2,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
    
    plt.figure()
    figname = 'heading_vs_choice'
    plt.xlabel('heading selectivity')
    plt.ylabel('choice selectivity')
    plt.title('heading selectivity vs. choice selectivity')
    plot_selectivity_corr(heading_selectivity_h1,choice_selectivity_h1,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(heading_selectivity_h2,choice_selectivity_h2,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
    
    plt.figure()
    figname = 'heading_vs_color'
    plt.xlabel('heading selectivity')
    plt.ylabel('color selectivity')
    plt.title('heading selectivity vs. color selectivity')
    plot_selectivity_corr(heading_selectivity_h1,color_selectivity_h1,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(heading_selectivity_h2,color_selectivity_h2,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)

    plt.figure()
    figname = 'sac_vs_color'
    plt.xlabel('sac selectivity')
    plt.ylabel('color selectivity')
    plt.title('sac selectivity vs. color selectivity')    
    plot_selectivity_corr(saccade_selectivity_h1,color_selectivity_h1,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(saccade_selectivity_h2,color_selectivity_h2,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
    
    #plot left sac prefer neuron's heading&color cor
    plt.figure()
    figname = 'rightsac_color_vs_sac'
    plt.xlabel('heading selectivity')
    plt.ylabel('color selectivity')
    plt.title('right sac prefer')
    selecth1 = saccade_selectivity_h1[:,0]>0
    selecth2 = saccade_selectivity_h2[:,0]>0
    plot_selectivity_corr(heading_selectivity_h1[selecth1,:],color_selectivity_h1[selecth1,:],rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(heading_selectivity_h2[selecth2,:],color_selectivity_h2[selecth2,:],rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
 
    #plot left sac prefer neuron's heading&color cor
    plt.figure()
    figname = 'leftsac_color_vs_sac'
    plt.xlabel('heading selectivity')
    plt.ylabel('color selectivity')
    plt.title('left sac prefer')
    selecth1 = saccade_selectivity_h1[:,0]<0
    selecth2 = saccade_selectivity_h2[:,0]<0
    plot_selectivity_corr(heading_selectivity_h1[selecth1,:],color_selectivity_h1[selecth1,:],rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h1para)
    if not hp.get("hidden_size2") is None:
        plot_selectivity_corr(heading_selectivity_h2[selecth2,:],color_selectivity_h2[selecth2,:],rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],para=h2para)
 
    # E_neural_activity = neural_activity[:, :, :e_size]
    # plot_input2neuron_connectivity(state_dict,heading_selectivity,rule=rule,figname_append = kwargs['figname_append'])  
    # plt.figure(7)
    plot_input2neuron_connectivity(effective_weight,[],rule_name,rule=rule,figname_append = kwargs['figname_append']) 
    plt.savefig("./lunwenfigure/input2hidden.svg")
    if not effective_weight.get("H2H_weight") is None:
        # plt.figure()
        plot_h2h_connectivity(effective_weight,[],[],rule_name,rule=rule,figname_append = kwargs['figname_append'])
        # plot_h2h_connectivity(effective_weight,np.vstack((heading_selectivity_h1,heading_selectivity_h2)),np.vstack((saccade_selectivity_h2,saccade_selectivity_h2)),rule_name,rule=rule,figname_append = kwargs['figname_append'])
        plt.savefig("./lunwenfigure/h2h.svg")
        # plot_forward_connectivity(effective_weight,hidden1_size,heading_selectivity_h1,saccade_selectivity_h2,rule_name,rule=rule,figname_append = kwargs['figname_append'])
    # plt.figure()
    plot_h2output_connectivity(effective_weight,[],rule_name,rule=rule,figname_append = kwargs['figname_append'])   
    plt.savefig("./lunwenfigure/h2output.svg")
    # plt.figure()# plot neural activity according to saccade direction
    # sacdir_col_list = np.sign(sac_per_trial)*2+np.sign(T1loc_per_trial)
    sacdir_list = np.sign(sac_per_trial)
    stim_on = test_trial.on
    dt = test_trial.dt
    # stim_times = [1000]
    times_relate = {'stim_on':stim_on,'dt':dt,'stim_dur':stim1_times}
    plot_population(neural_activity[:,:,:hidden1_size],sacdir_list,times_relate)
    plt.suptitle('hidden1_sac')
    SetFigure()
    plt.savefig("./lunwenfigure/h1population_sac.svg")
    plot_population(neural_activity[:,:,hidden1_size:],sacdir_list,times_relate)
    plt.suptitle('hidden2_sac') 
    SetFigure()
    plt.savefig("./lunwenfigure/h2population_sac.svg")
    plot_population(neural_activity[:,:,:hidden1_size],choice_per_trial,times_relate)
    plt.suptitle('hidden1_abschoice')
    SetFigure()
    plt.savefig("./lunwenfigure/h1population_choice.svg")
    plot_population(neural_activity[:,:,hidden1_size:],choice_per_trial,times_relate)
    plt.suptitle('hidden2_abschoice')
    SetFigure()
    plt.savefig("./lunwenfigure/h2population_choice.svg")
    
    

    h1abs_para = {'color': [x/255 + y*0.75*(1-1*2/4) for x, y in zip([139,10,80], [1,1,1])]}
    h1sac_para = {'color': [x/255 + y*0.75*(1-1*2/4) for x, y in zip([0,0,139], [1,1,1])]}
    if not hp.get("hidden_size2") is None:
        h2abs_para = {'color': [x/255 for x in [139,10,80]]}
        h2sac_para = {'color': [x/255 for x in [0,0,139]]}
    # plt.figure(11,figsize=(10, 16))# plot saccade divergence
    plt.figure(figsize=(10, 6))  # 宽度为10，高度为5
    figname = 'saccade_div'
    plt.title('saccade_div')
    # L1_sactime = get_divergence(sacdir_list[select_0heading],neural_activity[:,select_0heading,:hidden1_size],times_relate,rule_name,rule=rule,
                                # figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1sac_para)
    # plot_divergence(sacdir_list[select_0heading],neural_activity[:,select_0heading,:hidden1_size],times_relate,rule_name,
                    # rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1sac_para,doplot=1)
    L1_sactime = plot_conditioned_divergence(sacdir_list,heading_per_trial,neural_activity[:,:,:hidden1_size],times_relate,rule_name,rule=rule,figname=figname,
                    figname_append = kwargs['figname_append'],iarea=1,para=h1sac_para,doplot=1)
    # plot_conditioned_divergence(sacdir_list,heading_per_trial,neural_activity[:,:,:hidden1_size],times_relate,rule_name,rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1abs_para)

    SetFigure(15)
    plt.show()
    if not hp.get("hidden_size2") is None:
        # plot_divergence(sacdir_list[select_0heading],neural_activity[:,select_0heading,hidden1_size:],times_relate,rule_name,
                        # rule=rule,figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2sac_para,doplot=1)
        SetFigure(15)
        plt.show()
        L2_sactime = plot_conditioned_divergence(sacdir_list,heading_per_trial,neural_activity[:,:,hidden1_size:],times_relate,rule_name,rule=rule,
                        figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2sac_para,doplot=1)
    plt.savefig("./lunwenfigure/sac.svg")
    
    plt.figure(figsize=(10, 6))
    figname = 'choice_div'
    plt.title('choice_div')
    L1_choicetime = plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,:hidden1_size],times_relate,rule_name,rule=rule,
                                figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1abs_para,doplot=1)
    heading0_choice = choice_per_trial[select_0heading]
    heading0_h1_activity = neural_activity[:,select_0heading,:hidden1_size]
    heading0_h2_activity = neural_activity[:,select_0heading,hidden1_size:]
    # plot_divergence(heading0_choice,heading0_h1_activity,times_relate,rule_name,rule=rule,
                    # figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1abs_para,doplot=1)
    SetFigure(15)
    plt.show()
    L1_choicetime = get_divergence(heading0_choice,heading0_h1_activity,times_relate,rule_name,rule=rule,
                                   figname=figname,figname_append = kwargs['figname_append'],iarea=1,para=h1abs_para,doplot=1)
    if not hp.get("hidden_size2") is None:
        L2_choicetime = plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,hidden1_size:],times_relate,rule_name,rule=rule,
                                    figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2abs_para,doplot=1)
        # plot_divergence(heading0_choice,heading0_h2_activity,times_relate,rule_name,rule=rule,
                        # figname=figname,figname_append = kwargs['figname_append'],iarea=2,para=h2abs_para,doplot=1)
        SetFigure(15)
        plt.show()
        # L2_choicetime = get_divergence(heading0_choice,heading0_h2_activity,times_relate,rule_name,rule=rule,figname=figname,
                                       # figname_append = kwargs['figname_append'],iarea=2,para=h2abs_para,doplot=1)   
    plt.savefig("./lunwenfigure/abschoice.svg")
def plot_corr_heading_color_sac(heading_selectivity,saccade_selectivity,color_selectivity,**kwargs):
    import seaborn as sns
    
    heading_negative_color_positive = (heading_selectivity[:,0]<0)&(color_selectivity[:,0]>0)
    heading_positive_color_positive = (heading_selectivity[:,0]>0)&(color_selectivity[:,0]>0)
    heading_negative_color_negative = (heading_selectivity[:,0]<0)&(color_selectivity[:,0]<0)
    heading_positive_color_negative = (heading_selectivity[:,0]>0)&(color_selectivity[:,0]<0)
    classified_sac_selectivity = [abs(saccade_selectivity[heading_negative_color_positive,0]),
                                  abs(saccade_selectivity[heading_positive_color_positive,0]),
                                  abs(saccade_selectivity[heading_negative_color_negative,0]),
                                  abs(saccade_selectivity[heading_positive_color_negative,0])]
    plt.figure()    
    sns.boxplot( data=classified_sac_selectivity)
    sns.swarmplot(data=classified_sac_selectivity)
    plt.xlabel('classes')
    plt.ylabel('abs sac selectivity')
    plt.title('sac selectivity with heading&color selectivity')    
    if 'figname_append' in kwargs:
        figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+kwargs['figname_append']+'/sac_heading&color'
    else:  
        figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/sac_heading&color'
    plt.savefig(figname+'.png', transparent=True)     
    

# model_dir = './checkpoint/1hiddenHDsignrestrict.t7'           
# neuralactivity_dm(model_dir,figname_append='readoutsign_abs') 

# model_dir = './checkpoint/1hiddencolorHDtask.t7'          
# neuralactivity_color_dm(model_dir) 

# model_dir = './checkpoint/continue2hiddennet2.t7'           
# neuralactivity_color_dm(model_dir,figname_append='debug') 14
# neuralactivity_dm(model_dir,figname_append='debug') 
# model_dir = './checkpoint/delaysac.t7'           
# neuralactivity_delaysac(model_dir,figname_append='delaysac') 
# model_dir = './checkpoint/delaysac2hidden.t7'           
# neuralactivity_delaysac(model_dir,figname_append='2hidden')

# model_dir = './checkpoint/2AFC802hiddennet2keep.t7'    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i in [0]:
    figname_suffix = f'checkgpu/{i}'
    model_dir = './checkpoint/color2h1.t7'        
    # model_dir = './checkpoint/onlyfeedforward.t7'      
    # model_dir = './checkpoint_batchnew1/4411colorhdnet8.t7'     
    # model_dir = './checkpoint_scaled/4411colorhdnet8_feedback_x2.t7'  
    # model_dir = 'I:/model/data_simulation/neuralactivityinputtask/checkpoint_batchnew1/0421colorhdnet4.t7'
    neuralactivity_color_dm(model_dir,device,figname_append=figname_suffix) 
    # psychometric_color_dm(model_dir,figname_append=figname_suffix)
    # neuralactivity_dm(model_dir,figname_append='continue2AFC2hidden2') 

# model_dir = './checkpoint/dm2hiddennet2.t7'           
# neuralactivity_dm(model_dir,figname_append='dm2hidden2') 
# model_dir = './checkpoint/continue2hiddennet2.t7'           
# neuralactivity_color_dm(model_dir,figname_append='continue2hidden2') 
# neuralactivity_dm(model_dir,figname_append='continue2hidden2') 
