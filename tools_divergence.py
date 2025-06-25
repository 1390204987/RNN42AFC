# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:20:36 2024

@author: NaN
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr,ttest_ind
import os
from guassian_smooth import guass_smooth



from scipy.stats import pearsonr, ttest_ind
import numpy as np
def get_divergence(var_list, neural_activity, times_relate, rule_name, figname, iarea, **kwargs):
    dt = times_relate['dt']    
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons) * dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_dur[0] if isinstance(stim_dur, (list, np.ndarray)) else stim_dur
    
    time_steps = np.arange(neural_activity.shape[0]) * dt - stim_on
    time_begin_ind = np.where(time_steps >= 0)[0][0]
    time_end_ind = np.where(time_steps <= stim_end)[0][-1]
    trial_start = -stim_on
    trial_end = time_steps[-1]
    
    unique_var = np.unique(var_list)
    unit_num = neural_activity.shape[2]

    prefer_zactivity = []
    none_zactivity = []
    
    for iunit in range(unit_num):
        firing_rate = np.squeeze(np.mean(neural_activity[time_begin_ind:time_end_ind, :, iunit], 0))
        rr = pearsonr(var_list, firing_rate)
        prefer = unique_var[1] if rr[0] > 0 else unique_var[0]
        none = unique_var[0] if rr[0] > 0 else unique_var[1]

        select_prefer = var_list == prefer
        select_none = var_list == none

        align_markers = [trial_start, trial_end]
        t_centers,step_size, neural_act_smooth = guass_smooth(align_markers, neural_activity[:, :, iunit])
        base = np.mean(neural_act_smooth, axis=1, keepdims=True)
        gain = np.std(neural_act_smooth, keepdims=True) + 1e-6
        z_neural_activity = (neural_act_smooth - base) / gain

        prefer_zactivity.append(z_neural_activity[:, select_prefer])
        none_zactivity.append(z_neural_activity[:, select_none])
        
    divergence = np.zeros((unit_num, len(t_centers)))
    for iunit in range(unit_num):
        divergence[iunit, :] = np.nanmean(prefer_zactivity[iunit], axis=1) - np.nanmean(none_zactivity[iunit], axis=1)
    meandivergence = np.nanmean(divergence, axis=0)
    sem_divergence = np.nanstd(divergence, axis=0) / np.sqrt(unit_num)
    
    # Permutation testing (10 times)
    divergence_shuffle_all = np.zeros((10, unit_num, len(t_centers)))
    for n in range(10):
        var_list_shuffle = np.random.permutation(var_list)
        for iunit in range(unit_num):
            firing_rate = np.squeeze(np.mean(neural_activity[time_begin_ind:time_end_ind, :, iunit], 0))
            rr_shuffle = pearsonr(var_list_shuffle, firing_rate)
            prefer_shuffle = unique_var[1] if rr_shuffle[0] > 0 else unique_var[0]
            none_shuffle = unique_var[0] if rr_shuffle[0] > 0 else unique_var[1]

            select_prefer_shuffle = var_list_shuffle == prefer_shuffle
            select_none_shuffle = var_list_shuffle == none_shuffle

            align_markers = [trial_start, trial_end]
            _,_,neural_act_smooth = guass_smooth(align_markers, neural_activity[:, :, iunit])
            base = np.mean(neural_act_smooth, axis=1, keepdims=True)
            gain = np.std(neural_act_smooth, keepdims=True) + 1e-6
            z_neural_activity = (neural_act_smooth - base) / gain

            pref_z = z_neural_activity[:, select_prefer_shuffle]
            none_z = z_neural_activity[:, select_none_shuffle]
            divergence_shuffle_all[n, iunit, :] = np.nanmean(pref_z, axis=1) - np.nanmean(none_z, axis=1)
            
    # Mean across units, then mean across permutations
    meandivergence_shuffle_100 = np.nanmean(divergence_shuffle_all, axis=0)  # shape (unit, time)
    meandivergence_shuffle_avg = np.nanmean(meandivergence_shuffle_100, axis=0)  # shape (time,)
    sem_divergence_shuffle = np.nanstd(meandivergence_shuffle_100, axis=0) / np.sqrt(unit_num)
        
    # Perform t-test between real and permuted divergence
    t_stat, p_val_two_sided = ttest_ind(divergence, meandivergence_shuffle_100, nan_policy='omit')
    # 手动转换为单侧 p 值（greater）
    p_val_greater = p_val_two_sided / 2  # 双侧 p 值的一半
    p_val_greater = np.where(t_stat > 0, p_val_greater, 1 - p_val_greater)  # 确保 t_stat > 0 时才认为显著
    significant_time_points = t_centers[p_val_greater < 0.001]
    # 使用函数找到符合条件的连续时间段
    continuous_segments = find_continuous_segments(significant_time_points,step_size)
    if isinstance(continuous_segments, float) and np.isnan(continuous_segments):
        start_time = np.nan
    else:
        start_time = continuous_segments[0][0]
    return start_time

def plot_divergence(var_list, neural_activity, times_relate, rule_name, figname, iarea, **kwargs):
    plot_para = kwargs.get('para', {})
    dt = times_relate['dt']    
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons) * dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_dur[0] if isinstance(stim_dur, (list, np.ndarray)) else stim_dur

    time_steps = np.arange(neural_activity.shape[0]) * dt - stim_on
    time_begin_ind = np.where(time_steps >= 0)[0][0]
    time_end_ind = np.where(time_steps <= stim_end)[0][-1]
    trial_start = -stim_on
    trial_end = time_steps[-1]

    unique_var = np.unique(var_list)
    unit_num = neural_activity.shape[2]

    prefer_zactivity = []
    none_zactivity = []

    for iunit in range(unit_num):
        firing_rate = np.squeeze(np.mean(neural_activity[time_begin_ind:time_end_ind, :, iunit], 0))
        rr = pearsonr(var_list, firing_rate)
        prefer = unique_var[1] if rr[0] > 0 else unique_var[0]
        none = unique_var[0] if rr[0] > 0 else unique_var[1]

        select_prefer = var_list == prefer
        select_none = var_list == none

        align_markers = [trial_start, trial_end]
        t_centers,step_size,neural_act_smooth = guass_smooth(align_markers, neural_activity[:, :, iunit])
        base = np.mean(neural_act_smooth, axis=1, keepdims=True)
        gain = np.std(neural_act_smooth, keepdims=True) + 1e-6
        z_neural_activity = (neural_act_smooth - base) / gain

        prefer_zactivity.append(z_neural_activity[:, select_prefer])
        none_zactivity.append(z_neural_activity[:, select_none])

    divergence = np.zeros((unit_num, len(t_centers)))
    for iunit in range(unit_num):
        divergence[iunit, :] = np.nanmean(prefer_zactivity[iunit], axis=1) - np.nanmean(none_zactivity[iunit], axis=1)
    meandivergence = np.nanmean(divergence, axis=0)
    sem_divergence = np.nanstd(divergence, axis=0) / np.sqrt(unit_num)
    
    # Permutation testing (100 times)
    divergence_shuffle_all = np.zeros((10, unit_num, len(t_centers)))
    for n in range(10):
        var_list_shuffle = np.random.permutation(var_list)
        for iunit in range(unit_num):
            firing_rate = np.squeeze(np.mean(neural_activity[time_begin_ind:time_end_ind, :, iunit], 0))
            rr_shuffle = pearsonr(var_list_shuffle, firing_rate)
            prefer_shuffle = unique_var[1] if rr_shuffle[0] > 0 else unique_var[0]
            none_shuffle = unique_var[0] if rr_shuffle[0] > 0 else unique_var[1]

            select_prefer_shuffle = var_list_shuffle == prefer_shuffle
            select_none_shuffle = var_list_shuffle == none_shuffle

            align_markers = [trial_start, trial_end]
            _,_, neural_act_smooth = guass_smooth(align_markers, neural_activity[:, :, iunit])
            base = np.mean(neural_act_smooth, axis=1, keepdims=True)
            gain = np.std(neural_act_smooth, keepdims=True) + 1e-6
            z_neural_activity = (neural_act_smooth - base) / gain

            pref_z = z_neural_activity[:, select_prefer_shuffle]
            none_z = z_neural_activity[:, select_none_shuffle]
            divergence_shuffle_all[n, iunit, :] = np.nanmean(pref_z, axis=1) - np.nanmean(none_z, axis=1)

    # Mean across units, then mean across permutations
    meandivergence_shuffle_100 = np.nanmean(divergence_shuffle_all, axis=0)  # shape (unit, time)
    meandivergence_shuffle_avg = np.nanmean(meandivergence_shuffle_100, axis=0)  # shape (time,)
    sem_divergence_shuffle = np.nanstd(meandivergence_shuffle_100, axis=0) / np.sqrt(unit_num)

    # Perform t-test between real and permuted divergence
    t_stat, p_val = ttest_ind(divergence, meandivergence_shuffle_100, nan_policy='omit')

    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.errorbar(t_centers, meandivergence, yerr=sem_divergence, label='Original Mean Divergence', color=plot_para['color'])
    # plt.errorbar(t_centers, meandivergence_shuffle_avg, yerr=sem_divergence_shuffle, label='Shuffled Mean Divergence', linestyle='--', color=plot_para['color'])
    # 设置seaborn样式
    sns.set(style='whitegrid')
        
    # 原始数据
    plt.errorbar(t_centers, meandivergence, yerr=sem_divergence, 
                 label='Original Mean Divergence', 
                 color=plot_para['color'],
                 linewidth=2)
    
    # Shuffle数据
    plt.errorbar(t_centers, meandivergence_shuffle_avg, yerr=sem_divergence_shuffle, 
                 label='Shuffled Mean Divergence', 
                 linestyle='--', 
                 color=plot_para['color'],
                 linewidth=2)
    
    # 添加填充区域（可选）
    plt.fill_between(t_centers, 
                     meandivergence - sem_divergence,
                     meandivergence + sem_divergence,
                     color=plot_para['color'], alpha=0.1)
    
    plt.fill_between(t_centers,
                     meandivergence_shuffle_avg - sem_divergence_shuffle,
                     meandivergence_shuffle_avg + sem_divergence_shuffle,
                     color=plot_para['color'], alpha=0.1)
    
    # 美化图形
    plt.grid(False)
    plt.legend(frameon=False)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Divergence Comparison: Original vs Shuffled')
    sns.despine()  # 移除顶部和右侧的轴线
    plt.tight_layout()
    # Mark significant p-values
    significant_time_points = t_centers[p_val < 0.001]
    y_loc = np.zeros(significant_time_points.size)-iarea*0.01
    plt.scatter(significant_time_points, y_loc, color=plot_para['color'], marker='*', label='p < 0.001')

    
    plt.axvline(x=0, color='r', linestyle='--', label='Stim Onset')
    plt.axvline(x=stim_end, color='g', linestyle='--', label='Stim End')
    plt.xlabel('Time (s)')
    plt.ylabel('Divergence')
    # plt.title(f'Divergence between Preferred and Non-preferred Stimuli (p-value: {p_val:.4f})')
    plt.show()
    
    if 'figname_append' in kwargs:
        pathname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/'
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/'+figname
    else:     
        pathname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+figname
    
    os.makedirs(pathname,exist_ok=True)
    plt.savefig(figname+'.png', transparent=True) 

def plot_conditioned_divergence(var_list1,var_list2,neural_activity,times_relate,rule_name,figname,iarea,**kwargs):
    plot_para = kwargs['para']
    dt = times_relate['dt']    
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    
    unique_var = np.unique(var_list1)
    unit_num = neural_activity.shape[2]
    divergence = np.zeros((unit_num, len(time_steps)))
    meandivergence = np.zeros((1, len(time_steps)))
    
    var_list_shuffle = np.random.permutation(var_list1)
    divergence_shuffle = np.zeros((unit_num, len(time_steps)))
    meandivergence_shuffle = np.zeros((1, len(time_steps)))
    for iunit in range(unit_num):
        firing_rate = np.squeeze(np.nanmean(neural_activity[time_begin_ind:time_end_ind,:,iunit],0))
        rr = pearsonr(var_list1,firing_rate)
        if rr[0]>0:
            prefer = unique_var[1]
            none = unique_var[0]
        else:
            prefer = unique_var[0]
            none = unique_var[1]
        rr_shuffle = pearsonr(var_list_shuffle,firing_rate)
        if rr_shuffle[0]>0:
            prefer_shuffle = unique_var[1]
            none_shuffle = unique_var[0]
        else:
            prefer_shuffle = unique_var[0]
            none_shuffle = unique_var[1]    
            
        select_prefer = var_list1==prefer
        select_none = var_list1==none
        
        # base = np.min(neural_activity[:, :, iunit])
        # gain = np.max(neural_activity[:, :, iunit]) - base
        # z_neural_activity = (neural_activity[:,:,iunit]-base)/gain
        unique_var2 = np.unique(var_list2)
        z_neural_activity = np.zeros_like(neural_activity)
        for ivar2 in unique_var2:
            select = var_list2==ivar2
            selected_activity = neural_activity[:, select, iunit]
            # Calculate mean and std for the selected activity
            mean_activity = np.nanmean(selected_activity, axis=1, keepdims=True)
            std_activity = np.nanstd(selected_activity, axis=1, keepdims=True)
            z_neural_activity[:, select, iunit] = (selected_activity - mean_activity) / std_activity
        
        prefer_zactivity = z_neural_activity[:,select_prefer,iunit]
        none_zactivity = z_neural_activity[:,select_none,iunit]
        divergence[iunit, :] = np.nanmean(prefer_zactivity, axis=1) - np.nanmean(none_zactivity, axis=1)
        
        select_prefer_shuffle = var_list_shuffle==prefer_shuffle
        select_none_shuffle = var_list_shuffle==none_shuffle

        prefer_zactivity_shuffle = z_neural_activity[:,select_prefer_shuffle,iunit]
        none_zactivity_shuffle = z_neural_activity[:,select_none_shuffle,iunit]
        divergence_shuffle[iunit, :] = np.mean(prefer_zactivity_shuffle, axis=1) - np.mean(none_zactivity_shuffle, axis=1)
        
    meandivergence = np.nanmean(divergence, axis=0)
    meandivergence_shuffle = np.nanmean(divergence_shuffle, axis=0)
    sem_divergence = np.nanstd(divergence, axis=0) / np.sqrt(unit_num)
    sem_divergence_shuffle = np.nanstd(divergence_shuffle, axis=0) / np.sqrt(unit_num)
 
        # Perform t-test between original and shuffled mean divergence
    t_stat, p_val = ttest_ind(divergence, divergence_shuffle, nan_policy='omit')
    
    
    # Plotting
    # plt.figure(figsize=(10, 6))
    plt.errorbar(time_steps, meandivergence, yerr=sem_divergence, label='Original Mean Divergence', color=plot_para['color'])
    plt.errorbar(time_steps, meandivergence_shuffle, yerr=sem_divergence_shuffle, label='Shuffled Mean Divergence', linestyle='--', color=plot_para['color'])
    
    # Mark significant p-values
    significant_time_points = time_steps[p_val < 0.001]
    
    y_loc = np.zeros(significant_time_points.size)-iarea*0.001
    plt.scatter(significant_time_points, y_loc, color=plot_para['color'], marker='*', label='p < 0.01')

    
    plt.axvline(x=0, color='r', linestyle='--', label='Stim Onset')
    plt.axvline(x=stim_end, color='g', linestyle='--', label='Stim End')
    plt.xlabel('Time (s)')
    plt.ylabel('Divergence')
    # plt.title(f'Divergence between Preferred and Non-preferred Stimuli (p-value: {p_val:.4f})')
    # plt.legend()
    # plt.show()
    if 'figname_append' in kwargs:
        pathname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/'
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/'+figname
    else:     
        pathname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+figname
    
    os.makedirs(pathname,exist_ok=True)
    plt.savefig(figname+'.png', transparent=True) 
    
# 找到连续的显著时间点区间
def find_continuous_segments(time_points,step_size,min_length=5):
    if len(time_points) == 0:
        return np.nan
    
    # 先对时间点排序
    sorted_times = np.sort(time_points)
    
    # 计算时间点之间的差异
    diffs = np.diff(sorted_times)
    
    # 找到不连续的点（假设采样间隔相同，差异大于一个间隔的点就是不连续点）
    # 这里假设时间点是等间隔采样的，如果不等间隔需要调整阈值
    break_points = np.where(diffs > 1.5 * step_size)[0] + 1
    
    # 分割成连续段
    segments = np.split(sorted_times, break_points)
    
    # 筛选长度>=min_length的段
    valid_segments = [seg for seg in segments if len(seg) >= min_length]
    
    if not valid_segments:  # 如果没有符合条件的连续段
        return np.nan
    else:
        return valid_segments
    
    