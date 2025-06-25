# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 09:26:02 2022

for plot condtitioned or non-conditioned psth

@author: NZ
"""
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import ttest_ind 
import matplotlib.pyplot as plt
from mytools import popvec,get_y_direction
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_population(neural_activity,var_list,times_relate):
    '''
    for memeory saccade activity analysis
    get the prefer direction of each neuron(vector sum), 
    rank these neuron according there prefer angle,
    plot population activity change along time in a trial.
    '''    
    dt = times_relate['dt']
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt

    # mark the stimulus start and end 
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    
    spike_rates = np.mean(neural_activity[time_begin_ind:time_end_ind,:,:],0)
    # normalize neural activity  (x-min(min(x)))/max(max(x)-min(min(x)))
    # norm_neural_activity= (neural_activity-np.ndarray.min(np.ndarray.min(neural_activity,axis=0),axis=0))/(
    #     np.ndarray.max(np.ndarray.max(neural_activity,axis=0),axis=0)-np.ndarray.min(np.ndarray.min(neural_activity,axis=0),axis=0))
    norm_neural_activity = (neural_activity-np.ndarray.min(neural_activity,axis=0))/(
        np.ndarray.max(neural_activity,axis=0)-np.ndarray.min(neural_activity,axis=0))
    
    unique_var = np.unique(var_list)  
    # spike_rates_array = np.full((len(unique_var),int(len(var_list)/len(unique_var)),spike_rates.shape[1]),np.nan)
    spike_rates_array = []
    mean_spike_rates = np.full((len(unique_var),spike_rates.shape[1]),np.nan)
    for icondition in range(len(unique_var)):
        select = var_list == unique_var[icondition]
        spike_rates_array.append(spike_rates[select,:])
        mean_spike_rates[icondition,:] = np.mean(spike_rates[select,:],0) 
    #select condition selective neuron
    if len(spike_rates_array)>2:
        # do one way anova analysis
        [F_value,P_value] = f_oneway(spike_rates_array[0],spike_rates_array[1],spike_rates_array[2],spike_rates_array[3])
    else:
        [F_value,P_value] = ttest_ind(spike_rates_array[0],spike_rates_array[1])
    selective_neuron = P_value<0.05
    mean_spike_rates = mean_spike_rates[:,selective_neuron]
    #calculate prefer angle
    pref = np.arange(0,2*np.pi,2*np.pi/(len(unique_var)))
    temp_sum = mean_spike_rates.sum(axis=0)
    temp_cos = np.sum(mean_spike_rates.T*np.cos(pref),axis=-1)/temp_sum
    temp_sin = np.sum(mean_spike_rates.T*np.sin(pref),axis=-1)/temp_sum
    pref_angle = np.arctan2(temp_sin,temp_cos)
    pref_angle = np.mod(pref_angle,2*np.pi)
    neural_rank = np.argsort(pref_angle)
    neural_rank_value = np.sort(pref_angle)
    # plot_neuron = np.isnan(neural_rank_value)==False
    each_mean_activity = []
    unique_var_num = len(unique_var)
    fig, axs = plt.subplots(1, unique_var_num, figsize=(4*unique_var_num, 4))  # 画布大小可调整
    #plot 1 trial response & 1condition response
    for i_cond in range(len(unique_var)):
        select_condition = unique_var[i_cond]
        select = np.where(var_list == select_condition)
        show_norm_neu_act = norm_neural_activity[:,:,selective_neuron]
        # activity_matrix = np.squeeze(show_norm_neu_act[:,neural_rank])
        mean_activity_matrix = np.nanmean(show_norm_neu_act[:,select[0],:],1)
        each_mean_activity.append(mean_activity_matrix)
        ax = axs[i_cond] if unique_var_num > 1 else axs
        im = ax.imshow(mean_activity_matrix[:, neural_rank].T, origin='lower', aspect='auto',cmap='viridis')
        # 设置 xticks 和标签
        ax.set_xticks([time_begin_ind, time_end_ind])
        ax.set_xticklabels(['stim1 on', 'stim1 end'], fontfamily='Arial')        
        # 设置短刻度线（tick length）
        ax.tick_params(axis='x', direction='out', length=4, width=1,bottom=True)  # 长度4pt，粗细1pt

        ax.set_title(f'Condition {select_condition}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron')
        # 去掉图中网格线
        ax.grid(False)
    # 只加一个 colorbar 右边
    divider = make_axes_locatable(axs[-1])  # 最右侧子图
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 调整大小和间距
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()
    # plot all neural mean activity under each condtion with different color
    # for i_cond in range(len(each_mean_activity)):
    #     plt.plot(np.mean(each_mean_activity[i_cond],1))
    # plt.plot([time_end_ind,time_end_ind],[0,0.4])
    # plt.plot([time_begin_ind,time_begin_ind],[0,0.4])
    # plt.imshow()

def plot_alltrial_activity(neural_activity,rule_name,**kwargs):
    
    trial_num = range(neural_activity.shape[1])

    for i_neuron in range(neural_activity.shape[2]):
        plt.figure()
        for i_trial in trial_num:
            _ = plt.plot(neural_activity[:,i_trial,i_neuron])
            plt.xlabel('Time step')
            plt.ylabel('Activity')
            plt.title('neuron:{i_neuron}'.format(i_neuron = i_neuron))
        plt.show
        if 'figname_append' in kwargs:
            figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+kwargs['figname_append']+'/alltrialactivity_neuron'+str(i_neuron)
        else:
            figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/alltrialactivity_neuron'+str(i_neuron)
        plt.savefig(figname+'.png', transparent=True)

def plot_condtitioned_psth(neural_activity,ind_stim,condition_list,times_relate,rule_name,**kwargs):
    condition_num = np.unique(ind_stim)
    mean_neural_activity = []
    sem_neural_activity = []
    upper_neural_activity = []
    lower_neural_activity = []
    stim_coh = condition_list['stim_coh']    
    stim_loc = condition_list['stim_loc']    
    
    heading_colors = [(0	,0.545098039,0.270588235),
                     (0,0.803921569,0.4),
                     (0,0.933333333,0.462745098),
                     (1.00,1.00,0.00 ),
                     (0.94,0.50,0.50 ),
                     (1.00,0.27,0.00 ),
                     (0.545098039,0.101960784,0.101960784)]
    
    stim_ons = times_relate['stim_ons']
    dt = times_relate['dt']
    stim_durs = times_relate['stim_dur']
    stim_off = 0 + stim_durs[0]
    x = np.arange(neural_activity.shape[0])*dt-stim_ons[0]
    # x = np.arange(neural_activity.shape[0])*dt-stim_ons
    ymax = np.zeros([len(condition_num),neural_activity.shape[2]])
    ymin = np.zeros([len(condition_num),neural_activity.shape[2]])

    for i_condition in range(len(condition_num)):
        select_trials = np.where(ind_stim==condition_num[i_condition])
        select_neural_activity = np.squeeze(neural_activity[:,select_trials,:])
        # select_neural_activity = np.take(neural_activity,select_trials)
        mean_neural_activity.append(np.mean(select_neural_activity,1))
        sem_neural_activity.append(np.std(select_neural_activity,1)/np.sqrt(len(select_trials)))
        upper_neural_activity.append(mean_neural_activity[i_condition]+sem_neural_activity[i_condition])
        lower_neural_activity.append(mean_neural_activity[i_condition]-sem_neural_activity[i_condition])
        ymax[i_condition,:]=np.max(upper_neural_activity[i_condition],axis = 0)
        ymin[i_condition,:]=np.min(lower_neural_activity[i_condition],axis = 0)
    for i_neuron in range(neural_activity.shape[2]):
        plt.figure()
        for i_condition in range(len(condition_num)):
            _ = plt.plot(x,mean_neural_activity[i_condition][:,i_neuron],color=heading_colors[i_condition])
            _ = plt.plot(x,upper_neural_activity[i_condition][:,i_neuron],color=heading_colors[i_condition])
            _ = plt.plot(x,lower_neural_activity[i_condition][:,i_neuron],color=heading_colors[i_condition])
            _ = plt.fill_between(x,lower_neural_activity[i_condition][:,i_neuron],upper_neural_activity[i_condition][:,i_neuron],
                                  facecolor=heading_colors[i_condition],alpha=0.8)
        plt.xlabel('Time step')
        plt.ylabel('mean Activity')
        plt.title('neuron:{i_neuron}'.format(i_neuron = i_neuron))
        plt.plot([0,0],[np.min(ymin[:,i_neuron]),np.max(ymax[:,i_neuron])],color='k')
        plt.plot([stim_off,stim_off],[np.min(ymin[:,i_neuron]),np.max(ymax[:,i_neuron])],color='k')
        plt.show
        if 'figname_append' in kwargs:
            figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+kwargs['figname_append']+'/psth_neuron'+str(i_neuron)
        else:
            figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/psth_neuron'+str(i_neuron)
        plt.savefig(figname+'.png', transparent=True)    
        
def plot_saccade_divergence(neural_activity,ind_stim,condition_list,times_relate,y_hat,y_loc,rule_name,**kwargs):
    condition_num = np.unique(ind_stim)

    stim_coh = condition_list['stim_coh']    
    stim_loc = condition_list['stim_loc']    
    
    if len(np.unique(stim_coh))==1:
        unique_heading = np.unique(stim_loc)
    else:
        unique_heading = np.unique(np.sign(stim_loc)*stim_coh)
        
    y_dir = get_y_direction(y_hat,y_loc) 
    
    # only analysis trials with the output direction belongs to the given target 
    complete_trial_ind = np.where(np.isnan(y_dir)==False)
    
    in_stim = ind_stim[complete_trial_ind]
    heading = np.full_like(in_stim,np.nan,dtype=np.float32)
    for i_heading in range(len(unique_heading)):        
        heading[np.where(in_stim==i_heading)] = unique_heading[i_heading]
        
    sac_dir = y_dir[complete_trial_ind]  
    unique_sac_dir = np.unique(sac_dir)
    
    neural_activity = np.squeeze(neural_activity[:,complete_trial_ind,:])        
    
    # z_score for declear the effect of heading effects on saccade selectivity calculation

    
    # shift time window
    dt = times_relate['dt']
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_start = stim_on
    time_end = stim_end+500
    window_interval = 40 #ms
    step = 20 # ms
    steps = range(int(time_start[0]+window_interval/2),int(time_end[0]-window_interval/2),step)
    Z_neural_activity = np.full((len(steps),neural_activity.shape[1],neural_activity.shape[2]),np.nan)
    
    for i_heading in range(len(unique_heading)):
        i = 0
        for i_step in steps:
            select_ind = np.where(heading==unique_heading[i_heading])
            rerange_neural_activity = np.squeeze(np.mean(neural_activity[int((i_step-window_interval/2)/dt):int((i_step+window_interval/2)/dt),select_ind,:],axis=0))
            # Z_neural_activity[i,select_ind[0],:]=(rerange_neural_activity-np.mean(rerange_neural_activity,axis=0))/np.std(rerange_neural_activity,axis=0)
            Z_neural_activity[i,select_ind[0],:] = rerange_neural_activity
            i = i+1
    Z_neural_activity[np.isnan(Z_neural_activity)] = 0
    leftsac = unique_sac_dir[0]
    rightsac = unique_sac_dir[1]
    select_leftsac_ind = np.where(sac_dir==leftsac) 
    select_rightsac_ind = np.where(sac_dir==rightsac)       
    Z_left_sac_activity = Z_neural_activity[:,select_leftsac_ind[0],:]
    Z_right_sac_activity = Z_neural_activity[:,select_rightsac_ind[0],:]
    mean_left_sac_activity = np.squeeze(np.mean(Z_left_sac_activity,axis=1))
    mean_right_sac_activity = np.squeeze(np.mean(Z_right_sac_activity,axis=1))
    
    times = np.array(steps)
    for i_neuron in range(mean_left_sac_activity.shape[1]):
        plt.figure()
        _ = plt.plot(times,mean_left_sac_activity[:,i_neuron],color='green')
        _ = plt.plot(times,mean_right_sac_activity[:,i_neuron],color='red')
        ymax = np.max([np.max(mean_left_sac_activity[:,i_neuron]),np.max(mean_right_sac_activity[:,i_neuron])])
        ymin = np.min([np.min(mean_left_sac_activity[:,i_neuron]),np.min(mean_right_sac_activity[:,i_neuron])])
        
        plt.xlabel('Time step')
        plt.ylabel('mean Activity')
        plt.title('neuron:{i_neuron}'.format(i_neuron = i_neuron))
        plt.plot([stim_on,stim_on],[ymin,ymax],color='k')
        plt.plot([stim_end,stim_end],[ymin,ymax],color='k')
        # plt.show
        if 'figname_append' in kwargs:
            figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+kwargs['figname_append']+'/sac_divergence_neuron'+str(i_neuron)
        else:          
            figname = './neuralfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/sac_divergence_neuron'+str(i_neuron)
        # plt.savefig(figname+'.png', transparent=True)
