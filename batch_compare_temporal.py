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
def simple_temporal_plot_combined(file1, file2):
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
    
    # 创建两个图形：L1和L2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # L1_sactemporal - 两个文件在同一坐标轴
    plot_combined_temporal(ax1, time_points, 
                          df1_L1_temp, df1_L1_shu, 'File1',
                          df2_L1_temp, df2_L1_shu, 'File2',
                          'L1_sactemporal 时间序列对比')
    
    # L2_sactemporal - 两个文件在同一坐标轴
    plot_combined_temporal(ax2, time_points,
                          df1_L2_temp, df1_L2_shu, 'File1',
                          df2_L2_temp, df2_L2_shu, 'File2',
                          'L2_sactemporal 时间序列对比')
    
    plt.tight_layout()
    plt.show()

def plot_combined_temporal(ax, time_points, temp_data1, shu_data1, label1, temp_data2, shu_data2, label2, title):
    """在同一坐标轴上绘制两个文件的时间序列"""
    
    # 计算均值
    temp_mean1 = temp_data1.mean(axis=1)
    shu_mean1 = shu_data1.mean(axis=1)
    temp_mean2 = temp_data2.mean(axis=1)
    shu_mean2 = shu_data2.mean(axis=1)
    
    # 计算标准差（用于阴影区域）
    temp_std1 = temp_data1.std(axis=1)
    shu_std1 = shu_data1.std(axis=1)
    temp_std2 = temp_data2.std(axis=1)
    shu_std2 = shu_data2.std(axis=1)
    
    # 绘制File1的数据 - 实线
    ax.plot(time_points, temp_mean1, label=f'{label1} Temporal', color='blue', linewidth=2)
    ax.fill_between(time_points, temp_mean1 - temp_std1, temp_mean1 + temp_std1, 
                   alpha=0.2, color='blue')
    
    # 绘制File1的shuffle数据 - 虚线
    ax.plot(time_points, shu_mean1, label=f'{label1} Shuffle', color='blue', 
           linewidth=1.5, linestyle='--', alpha=0.8)
    ax.fill_between(time_points, shu_mean1 - shu_std1, shu_mean1 + shu_std1, 
                   alpha=0.1, color='blue')
    
    # 绘制File2的数据 - 实线
    ax.plot(time_points, temp_mean2, label=f'{label2} Temporal', color='red', linewidth=2)
    ax.fill_between(time_points, temp_mean2 - temp_std2, temp_mean2 + temp_std2, 
                   alpha=0.2, color='red')
    
    # 绘制File2的shuffle数据 - 虚线
    ax.plot(time_points, shu_mean2, label=f'{label2} Shuffle', color='red', 
           linewidth=1.5, linestyle='--', alpha=0.8)
    ax.fill_between(time_points, shu_mean2 - shu_std2, shu_mean2 + shu_std2, 
                   alpha=0.1, color='red')
    
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
                  color='blue', marker='^', s=30, label=f'{label1} sig (p<0.05)', alpha=0.7)
    
    if sig_points2:
        y_min = min(temp_mean1.min(), shu_mean1.min(), temp_mean2.min(), shu_mean2.min())
        y_range = max(temp_mean1.max(), shu_mean1.max(), temp_mean2.max(), shu_mean2.max()) - y_min
        sig_y = y_min - 0.08 * y_range
        
        ax.scatter(sig_points2, [sig_y] * len(sig_points2), 
                  color='red', marker='v', s=30, label=f'{label2} sig (p<0.05)', alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('时间点')
    ax.set_ylabel('值')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    sig_info = f'{label1}: {len(sig_points1)}/{len(time_points)} 显著\n{label2}: {len(sig_points2)}/{len(time_points)} 显著'
    ax.text(0.02, 0.98, sig_info, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 使用简洁版本
# 使用示例
file1 = "./sactemporal_batch2.xlsx"  # 第一个Excel文件
file2 = "./sactemporal_batch3.xlsx"  # 第二个Excel文件，替换为实际路径

simple_temporal_plot_combined(file1, file2)