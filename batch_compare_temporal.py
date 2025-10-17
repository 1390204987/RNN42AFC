# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:48:41 2025

plot two batch output temporal difference

@author: NaN

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
def simple_temporal_plot_combined(file1, file2,times_relate):
    """简洁版本的时间序列图 - 合并显示"""
    
    # 加载数据
    df1_L1_temp = pd.read_excel(file1, sheet_name='L1_sactemporal', index_col=0)
    df1_L2_temp = pd.read_excel(file1, sheet_name='L2_sactemporal', index_col=0)
    df1_L1_shu = pd.read_excel(file1, sheet_name='shu_L1_sactemporal', index_col=0)
    df1_L2_shu = pd.read_excel(file1, sheet_name='shu_L2_sactemporal', index_col=0)
    
    df2_L1_temp = pd.read_excel(file2, sheet_name='L1_sactemporal', index_col=0)
    df2_L2_temp = pd.read_excel(file2, sheet_name='L2_sactemporal', index_col=0)
    df2_L1_shu = pd.read_excel(file2, sheet_name='shu_L1_sactemporal', index_col=0)
    df2_L2_shu = pd.read_excel(file2, sheet_name='shu_L2_sactemporal', index_col=0)
    
    time_points = np.arange(len(df1_L1_temp))
    time_points_ms = time_points * times_relate['dt']-times_relate['stim_on']
    
    # 创建两个图形：L1和L2
    fig, ax1 = plt.subplots(figsize=(8, 6))  # 创建图形和坐标轴 
    # L1_sactemporal - 两个文件在同一坐标轴
    plot_combined_temporal(ax1, time_points_ms, 
                          df1_L1_temp, df1_L1_shu, 'File1',
                          df2_L1_temp, df2_L1_shu, 'File2',
                          'L1_sactemporal')
    # 在stim1 on和off时间点画竖线
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Stim On')
    ax1.axvline(x=times_relate['stim_dur'], color='gray', linestyle='--', alpha=0.7, label='Stim Off')
    plt.savefig("./lunwenfigure/cutfeedback_aftertrain.svg")
    
    fig, ax2 = plt.subplots(figsize=(8, 6))  # 创建图形和坐标轴
    # L2_sactemporal - 两个文件在同一坐标轴
    plot_combined_temporal(ax2, time_points_ms,
                          df1_L2_temp, df1_L2_shu, 'File1',
                          df2_L2_temp, df2_L2_shu, 'File2',
                          'L2_sactemporal')
    # 在stim1 on和off时间点画竖线
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Stim On')
    ax1.axvline(x=times_relate['stim_dur'], color='gray', linestyle='--', alpha=0.7, label='Stim Off')
    
    plt.show()

def plot_combined_temporal(ax, time_points, temp_data1, shu_data1, label1, temp_data2, shu_data2, label2, title):
    """在同一坐标轴上绘制两个文件的时间序列"""
    f1 = {'color': [x/255 + y*0.75*(1-1*2/4) for x, y in zip([0,0,139], [1,1,1])],'linestyle':'-'}
    f2 = {'color': [x/255 + y*0.75*(1-1*2/4) for x, y in zip([0,0,139], [1,1,1])],'linestyle':'-'}
    # 计算均值
    temp_mean1 = temp_data1.mean(axis=1)
    shu_mean1 = shu_data1.mean(axis=1)
    temp_mean2 = temp_data2.mean(axis=1)
    shu_mean2 = shu_data2.mean(axis=1)
    
    # 计算标准差（用于阴影区域）
    temp_std1 = temp_data1.std(axis=1) / np.sqrt(temp_data1.shape[1])
    shu_std1 = shu_data1.std(axis=1) / np.sqrt(temp_data1.shape[1])
    temp_std2 = temp_data2.std(axis=1) / np.sqrt(temp_data1.shape[1])
    shu_std2 = shu_data2.std(axis=1) / np.sqrt(temp_data1.shape[1])
    
    # 绘制File1的数据 - 实线
    ax.plot(time_points, temp_mean1, label=f'{label1} Temporal', color=f1['color'], linewidth=4)
    ax.plot(time_points, temp_mean1, label=f'{label1} Temporal', color='white', linewidth=1.5)
    ax.fill_between(time_points, temp_mean1 - temp_std1, temp_mean1 + temp_std1, 
                   alpha=0.2, color=f1['color'])
    
    # 绘制File1的shuffle数据 - 虚线
    ax.plot(time_points, shu_mean1, label=f'{label1} Shuffle', color=f1['color'], 
           linewidth=1.5, linestyle='--', alpha=0.8)
    # ax.fill_between(time_points, shu_mean1 - shu_std1, shu_mean1 + shu_std1, 
    #                alpha=0.1, color=f1['color'])
    
    # 绘制File2的数据 - 实线
    ax.plot(time_points, temp_mean2, label=f'{label2} Temporal', color=f2['color'], linewidth=2)
    # ax.plot(time_points, temp_mean2, label=f'{label2} Temporal', color='white', linewidth=1.5)
    ax.fill_between(time_points, temp_mean2 - temp_std2, temp_mean2 + temp_std2, 
                   alpha=0.2, color=f2['color'])
    
    # 绘制File2的shuffle数据 - 虚线
    ax.plot(time_points, shu_mean2, label=f'{label2} Shuffle', color=f2['color'], 
           linewidth=1.5, linestyle='--', alpha=0.8)
    # ax.fill_between(time_points, shu_mean2 - shu_std2, shu_mean2 + shu_std2, 
    #                alpha=0.1, color='red')
    
    # 标记显著性时间点（File1的temporal vs shuffle）
    sig_points1 = []
    for t_idx in range(len(time_points)):
        temp_vals1 = temp_data1.iloc[t_idx].dropna().values
        shu_vals1 = shu_data1.iloc[t_idx].dropna().values
        
        if len(temp_vals1) > 1 and len(shu_vals1) > 1:
            t_stat, p_value = stats.ttest_ind(temp_vals1, shu_vals1, equal_var=False)
            if p_value < 0.05:
                sig_points1.append(time_points[t_idx])
    
    # 标记显著性时间点（File2的temporal vs shuffle）
    sig_points2 = []
    for t_idx in range(len(time_points)):
        temp_vals2 = temp_data2.iloc[t_idx].dropna().values
        shu_vals2 = shu_data2.iloc[t_idx].dropna().values
        
        if len(temp_vals2) > 1 and len(shu_vals2) > 1:
            t_stat, p_value = stats.ttest_ind(temp_vals2, shu_vals2, equal_var=False)
            if p_value < 0.05:
                sig_points2.append(time_points[t_idx])
    
    # 在显著性时间点添加标记
    if sig_points1:
        y_min = min(temp_mean1.min(), shu_mean1.min(), temp_mean2.min(), shu_mean2.min())
        y_range = max(temp_mean1.max(), shu_mean1.max(), temp_mean2.max(), shu_mean2.max()) - y_min
        sig_y = y_min - 0.05 * y_range
        
        ax.scatter(sig_points1, [sig_y] * len(sig_points1), 
                  color=f1['color'], marker='^', s=30, label=f'{label1} sig (p<0.05)', alpha=0.7)
    
    if sig_points2:
        y_min = min(temp_mean1.min(), shu_mean1.min(), temp_mean2.min(), shu_mean2.min())
        y_range = max(temp_mean1.max(), shu_mean1.max(), temp_mean2.max(), shu_mean2.max()) - y_min
        sig_y = y_min - 0.08 * y_range
        
        ax.scatter(sig_points2, [sig_y] * len(sig_points2), 
                  color=f1['color'], marker='*', s=30, label=f'{label2} sig (p<0.05)', alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('time')
    ax.set_ylabel('firing rate')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    


# 使用简洁版本
# 使用示例
file1 = "./sactemporal_batch2.xlsx"  # 第一个Excel文件
file2 = "./sactemporal_batch6.xlsx"  # 第二个Excel文件，替换为实际路径
dt= 20 #ms
stim1_on = 200 #ms
stim_times = 1000 #ms
stim_off = stim1_on+stim_times
times_relate = {'stim_on':stim1_on,'stim_off':stim_off,'dt':dt,'stim_dur':stim_times}
simple_temporal_plot_combined(file1, file2,times_relate)