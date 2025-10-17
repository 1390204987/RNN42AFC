# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 11:35:32 2025
plot each net roc or temporal activity together
@author: NaN
"""
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
from roc_tool import rocN

import mytask
from mytask import generate_trials, rule_name, get_dist

# import mynetwork_new3
# from mynetwork_new3 import Net
import mynetwork8
from mynetwork8 import Net

from mytools import popvec,get_y_direction

from scipy.stats import pearsonr

from tools_divergence import plot_divergence, plot_conditioned_divergence, get_divergence

from scipy.stats import pearsonr


THETA = 0.3 * np.pi


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


    
    sacdir_list = np.sign(sac_per_trial)
    stim_on = test_trial.on
    dt = test_trial.dt
    stim_times = [1000]
    times_relate = {'stim_on':stim_on,'dt':dt,'stim_dur':stim_times}
     
    h1abs_para = {'color': [x/255 for x in [139,10,80]]}
    h1sac_para = {'color': [x/255 for x in [0,0,139]]}
    if not hp.get("hidden_size2") is None:
        h2abs_para = {'color': [x/255 + y*0.75*(1-1*2/4) for x, y in zip([139,10,80], [1,1,1])]}
        h2sac_para = {'color': [x/255 + y*0.75*(1-1*2/4) for x, y in zip([0,0,139], [1,1,1])]}

    figname = 'saccade_div'

    L1_sactime,L1_sactemporal,L1_shuffle_sac = plot_conditioned_divergence(sacdir_list,heading_per_trial,neural_activity[:,:,:hidden1_size],times_relate,rule_name,rule=rule,figname=figname,
                    iarea=1,para=h1sac_para,doplot=0)


    if not hp.get("hidden_size2") is None:


        L2_sactime,L2_sactemporal,L2_shuffle_sac = plot_conditioned_divergence(sacdir_list,heading_per_trial,neural_activity[:,:,hidden1_size:],times_relate,rule_name,rule=rule,
                        figname=figname,iarea=2,para=h2sac_para,doplot=0)
# 
    L1_choicetime,_,_ = plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,:hidden1_size],times_relate,rule_name,rule=rule,
                                figname=figname,iarea=1,para=h1abs_para,doplot=0)
    heading0_choice = choice_per_trial[select_0heading]
    heading0_h1_activity = neural_activity[:,select_0heading,:hidden1_size]
    heading0_h2_activity = neural_activity[:,select_0heading,hidden1_size:]

    if not hp.get("hidden_size2") is None:
        L2_choicetime,_,_ = plot_conditioned_divergence(choice_per_trial,heading_per_trial,neural_activity[:,:,hidden1_size:],times_relate,rule_name,rule=rule,
                                    figname=figname,iarea=2,para=h2abs_para,doplot=0)

    stim_times = np.array(stim_times)
    fir_rate = np.mean(neural_activity[stim_on:int(stim_on+stim_times[0]/dt),:,:],axis=0 )

    n_units = fir_rate.shape[1]
    ROC_sac = np.full(n_units, np.nan)
    # z score for heading
    zfir_rate = np.full_like(fir_rate,np.nan)
    for iheading in unique_heading:
        select = np.isclose(heading_per_trial, iheading, rtol=1e-5, atol=1e-8)
        zfir_rate[select,:] = (fir_rate[select,:]-np.mean(fir_rate[select,:],axis=0))/np.std(fir_rate[select,:],axis=0)

    select_left_sac = sacdir_list==0
    select_right_sac = sacdir_list==1
    fir_Lsac = zfir_rate[select_left_sac,:]
    fir_Rsac = zfir_rate[select_right_sac,:]
    
    # 对每个单元分别计算ROC
    for i in range(n_units):
        # 提取当前单元的数据（第一个维度的所有元素）
        x = fir_Lsac[:, i]
        y = fir_Rsac[:, i]
        
        # 计算ROC（只获取AUC值，忽略其他返回值）
        roc_auc, _, _ = rocN(x, y)
        ROC_sac[i] = roc_auc
    
    L1_sacROC = ROC_sac[:hidden1_size] 
    L2_sacROC = ROC_sac[hidden1_size:]
    
    return L1_sacROC,L2_sacROC,L1_sactemporal,L2_sactemporal,L1_shuffle_sac,L2_shuffle_sac

def list_files_in_directory(directory):
    files = [file for file in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, file)) and file.endswith('.t7')]        
    return files

filespath = './checkpoint_batchnew2_1'
# filespath = './check'
nets = list_files_in_directory(filespath)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# df = pd.DataFrame(columns=["netid","L1_sacROC","L2_sacROC"])

# 创建两个DataFrame来分别存储L1和L2 roc数据
df_L1 = pd.DataFrame()
df_L2 = pd.DataFrame()

# 创建两个DataFrame来分别存储L1和L2 temporal数据
df_L1temporal = pd.DataFrame()
df_L2temporal = pd.DataFrame()
df_L1shu_temporal = pd.DataFrame()
df_L2shu_temporal = pd.DataFrame()

for net in nets:
    model_dir = filespath + '/' + net
    L1_sacROC, L2_sacROC,L1_sactemporal,L2_sactemporal,L1_shuffle_sac,L2_shuffle_sac = neuralactivity_color_dm(model_dir, device)
    numbersinnetname = re.findall(r'\d+', net)
    netid = numbersinnetname[0]
    
    # 为每个net创建列名
    col_name = f"net_{netid}"
    
    # 将L1_sacROC和L2_sacROC分别保存为DataFrame的列
    df_L1[col_name] = L1_sacROC
    df_L2[col_name] = L2_sacROC
    # 将L1_temporal和L2_temporal分别保存为DataFrame的列
    df_L1temporal[col_name] = L1_sactemporal
    df_L2temporal[col_name] = L2_sactemporal
    df_L1shu_temporal[col_name] = L1_shuffle_sac
    df_L2shu_temporal[col_name] = L2_shuffle_sac
    
    
# 保存到Excel的两个不同sheet中
with pd.ExcelWriter("./sacROC_batch6.xlsx") as writer:
    df_L1.to_excel(writer, sheet_name='L1_sacROC', index=False)
    df_L2.to_excel(writer, sheet_name='L2_sacROC', index=False)
    
# 保存到Excel的两个不同sheet中
with pd.ExcelWriter("./sactemporal_batch6.xlsx") as writer:
    df_L1temporal.to_excel(writer, sheet_name='L1_sactemporal', index=False)
    df_L2temporal.to_excel(writer, sheet_name='L2_sactemporal', index=False)
    df_L1shu_temporal.to_excel(writer, sheet_name='shu_L1_sactemporal', index=False)
    df_L2shu_temporal.to_excel(writer, sheet_name='shu_L2_sactemporal', index=False)

print(f"保存完成！")
print(f"L1_sacROC sheet: {df_L1.shape[1]}个net, 每个net {df_L1.shape[0]}个值")
print(f"L2_sacROC sheet: {df_L2.shape[1]}个net, 每个net {df_L2.shape[0]}个值")













