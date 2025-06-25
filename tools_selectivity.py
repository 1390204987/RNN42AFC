# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:15:09 2022
for selectivity analysis
@author: NZ
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import os

def _selectivity(var_list,neural_activity,times_relate):
    
    dt = times_relate['dt']    
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]

    firing_rate = np.squeeze(np.mean(neural_activity[time_begin_ind:time_end_ind,:,:],0))

    selectivity = np.full((firing_rate.shape[1],2),np.nan,dtype=np.float32)
    for i_neuron in range(firing_rate.shape[1]):
        y = firing_rate[:,i_neuron]
        selectivity[i_neuron,:] = pearsonr(var_list,y)
    
    return selectivity

def _z_selectivity(var_list,z_var_list,neural_activity,times_relate):
    
    dt = times_relate['dt']    
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    
    firing_rate = np.squeeze(np.mean(neural_activity[time_begin_ind:time_end_ind,:,:],0))

    Z_firing_rate = np.full_like(firing_rate,0)
    unique_z_var = np.unique(z_var_list)
    for i_z_var in range(len(unique_z_var)):
        select_ind = np.where(z_var_list==unique_z_var[i_z_var])
        rerange_firing_rate = np.squeeze(firing_rate[select_ind,:])
        Z_firing_rate[select_ind[0],:] = (rerange_firing_rate
             -np.mean(rerange_firing_rate,axis=0))/np.std(rerange_firing_rate,axis=0)
    Z_firing_rate[np.isnan(Z_firing_rate)]=0
    z_selectivity = np.full((Z_firing_rate.shape[1],2),np.nan,dtype=np.float32)
    for i_neuron in range(Z_firing_rate.shape[1]):
        y = Z_firing_rate[:,i_neuron]
        # only use non-zero heading condition
        select_var_list = var_list[z_var_list!=np.pi]
        # select_y = y[var_list~=0]
        z_selectivity[i_neuron,:] = pearsonr(var_list,y)
        
    return z_selectivity

def plot_selectivity_corr(var1_selectivity,var2_selectivity,rule_name,figname,**kwargs):
    useful_neuron_var1_ind = np.where(np.isnan(var1_selectivity[:,0])==False)
    useful_neuron_var2_ind = np.where(np.isnan(var2_selectivity[:,0])==False)
    useful_neuron_ind = np.intersect1d(useful_neuron_var1_ind[0],useful_neuron_var2_ind[0])
    var1_var2_corr = pearsonr(var1_selectivity[useful_neuron_ind,0],var2_selectivity[useful_neuron_ind,0])
    plot_para = kwargs['para']
    text_y = plot_para['text_y']
    plt.scatter(var1_selectivity[:,0],var2_selectivity[:,0],color=plot_para['color'],marker=plot_para['marker'])
    if var1_var2_corr[1]<0.05:
        plt.text(0,text_y-0.1,'p='+str(f"{var1_var2_corr[1]:.1E}"),color=plot_para['color'])
        plt.text(0,text_y,'corr='+str(var1_var2_corr[0]).split('.')[0]+'.'+str(var1_var2_corr[0]).split('.')[1][:2],color=plot_para['color'])
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    if 'figname_append' in kwargs:
        pathname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/'
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/'+figname
    else:     
        pathname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+figname
    
    os.makedirs(pathname,exist_ok=True)
    plt.savefig(figname+'.png', transparent=True) 