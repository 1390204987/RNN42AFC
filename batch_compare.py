# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 16:23:04 2025
plot and comapare ROC trained by differnt batch
@author: NaN
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def load_and_plot_separately(file1, file2):
    """加载数据并分别绘制分布图和方差图"""
    
    # 加载数据
    df1_L1 = pd.read_excel(file1, sheet_name='L1_sacROC', index_col=0)
    df1_L2 = pd.read_excel(file1, sheet_name='L2_sacROC', index_col=0)
    df2_L1 = pd.read_excel(file2, sheet_name='L1_sacROC', index_col=0)
    df2_L2 = pd.read_excel(file2, sheet_name='L2_sacROC', index_col=0)
    
    print("=== 数据基本信息 ===")
    print(f"文件1 - L1: {df1_L1.shape[1]}个net, {df1_L1.shape[0]}个神经元")
    print(f"文件1 - L2: {df1_L2.shape[1]}个net, {df1_L2.shape[0]}个神经元")
    print(f"文件2 - L1: {df2_L1.shape[1]}个net, {df2_L1.shape[0]}个神经元")
    print(f"文件2 - L2: {df2_L2.shape[1]}个net, {df2_L2.shape[0]}个神经元")
    
    # 展平所有数据
    all_L1_f1 = df1_L1.values.flatten()
    all_L2_f1 = df1_L2.values.flatten()
    all_L1_f2 = df2_L1.values.flatten()
    all_L2_f2 = df2_L2.values.flatten()
    
    # 移除NaN值
    all_L1_f1 = all_L1_f1[~np.isnan(all_L1_f1)]
    all_L2_f1 = all_L2_f1[~np.isnan(all_L2_f1)]
    all_L1_f2 = all_L1_f2[~np.isnan(all_L1_f2)]
    all_L2_f2 = all_L2_f2[~np.isnan(all_L2_f2)]
    
    # 计算每个net的方差
    var_L1_f1 = df1_L1.var(axis=0)
    var_L2_f1 = df1_L2.var(axis=0)
    var_L1_f2 = df2_L1.var(axis=0)
    var_L2_f2 = df2_L2.var(axis=0)
    
    return (all_L1_f1, all_L2_f1, all_L1_f2, all_L2_f2, 
            var_L1_f1, var_L2_f1, var_L1_f2, var_L2_f2)

def plot_distribution_comparison(all_L1_f1, all_L2_f1, all_L1_f2, all_L2_f2):
    """绘制分布对比图 - 简化版统计检验"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 计算统一的坐标轴范围
    all_data = np.concatenate([all_L1_f1, all_L2_f1, all_L1_f2, all_L2_f2])
    x_min, x_max = np.min(all_data), np.max(all_data)
    
    # 计算统一的最大密度
    def get_max_density(data_list, bins=50):
        max_density = 0
        for data in data_list:
            hist, _ = np.histogram(data, bins=bins, density=True)
            max_density = max(max_density, np.max(hist))
        return max_density
    
    all_data_lists = [all_L1_f1, all_L1_f2, all_L2_f1, all_L2_f2]
    y_max = get_max_density(all_data_lists) * 1.1
    
    # 执行关键统计检验
    def key_statistical_tests(data1, data2):
        results = {}
        
        # KS检验
        ks_stat, ks_p = stats.ks_2samp(data1, data2)
        results['KS_p'] = ks_p
        results['KS_sig'] = '***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'ns'
 
        return results
    
    # 执行检验
    l1_tests = key_statistical_tests(all_L1_f1, all_L1_f2)
    l2_tests = key_statistical_tests(all_L2_f1, all_L2_f2)
    
    # 1. L1_sacROC 分布对比
    axes[0].hist(all_L1_f1, bins=50, alpha=0.7, color='blue', 
                label=f'File1 L1 (n={len(all_L1_f1)})', density=True)
    axes[0].hist(all_L1_f2, bins=50, alpha=0.7, color='red', 
                label=f'File2 L1 (n={len(all_L1_f2)})', density=True)
    axes[0].set_title('L1_sacROC 分布对比', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('ROC值')
    axes[0].set_ylabel('密度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(0, y_max)
    
    # 添加统计信息和显著性标注
    axes[0].text(0.05, 0.95, f'File1: μ={np.mean(all_L1_f1):.3f}±{np.std(all_L1_f1):.3f}', 
                transform=axes[0].transAxes, color='blue', fontsize=9)
    axes[0].text(0.05, 0.88, f'File2: μ={np.mean(all_L1_f2):.3f}±{np.std(all_L1_f2):.3f}', 
                transform=axes[0].transAxes, color='red', fontsize=9)
    
    # 显著性标注
    sig_text = f"KS: p={l1_tests['KS_p']:.2e} ({l1_tests['KS_sig']})\n" 
    
    axes[0].text(0.95, 0.95, sig_text, transform=axes[0].transAxes, 
                fontsize=20, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
    
    # 2. L2_sacROC 分布对比
    axes[1].hist(all_L2_f1, bins=50, alpha=0.7, color='blue', 
                label=f'File1 L2 (n={len(all_L2_f1)})', density=True)
    axes[1].hist(all_L2_f2, bins=50, alpha=0.7, color='red', 
                label=f'File2 L2 (n={len(all_L2_f2)})', density=True)
    axes[1].set_title('L2_sacROC 分布对比', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('ROC值')
    axes[1].set_ylabel('密度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(0, y_max)
    

    
    # 显著性标注
    sig_text = f"KS: p={l2_tests['KS_p']:.2e} ({l2_tests['KS_sig']})\n" 
    
    axes[1].text(0.95, 0.95, sig_text, transform=axes[1].transAxes, 
                fontsize=20, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# 使用示例
# plot_distribution_comparison(all_L1_f1, all_L2_f1, all_L1_f2, all_L2_f2)

def plot_variance_comparison(var_L1_f1, var_L2_f1, var_L1_f2, var_L2_f2):
    """绘制方差对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 方差箱线图
    variance_data = [var_L1_f1, var_L2_f1, var_L1_f2, var_L2_f2]
    box_plot = axes[0, 0].boxplot(variance_data, labels=['L1_f1', 'L2_f1', 'L1_f2', 'L2_f2'], 
                                 patch_artist=True)
    
    # 设置颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0, 0].set_title('各net方差分布箱线图', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('方差')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加均值标注
    for i, data in enumerate(variance_data):
        axes[0, 0].text(i+1, np.mean(data), f'μ={np.mean(data):.4f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # 2. 方差柱状图
    mean_variances = [np.mean(var_L1_f1), np.mean(var_L2_f1), np.mean(var_L1_f2), np.mean(var_L2_f2)]
    std_variances = [np.std(var_L1_f1), np.std(var_L2_f1), np.std(var_L1_f2), np.std(var_L2_f2)]
    labels = ['L1_f1', 'L2_f1', 'L1_f2', 'L2_f2']
    colors = ['blue', 'green', 'red', 'orange']
    
    bars = axes[0, 1].bar(labels, mean_variances, yerr=std_variances, 
                         capsize=5, alpha=0.7, color=colors)
    axes[0, 1].set_title('平均方差对比（带标准差）', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('方差')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标注
    for i, (bar, mean_val) in enumerate(zip(bars, mean_variances)):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + std_variances[i] + 0.001,
                       f'{mean_val:.4f}', ha='center', va='bottom')
    
    # 3. 方差散点图
    axes[1, 0].scatter(range(len(var_L1_f1)), var_L1_f1, alpha=0.6, label='L1_f1', color='blue')
    axes[1, 0].scatter(range(len(var_L1_f2)), var_L1_f2, alpha=0.6, label='L1_f2', color='red')
    axes[1, 0].set_title('L1_sacROC 各net方差散点图', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('net索引')
    axes[1, 0].set_ylabel('方差')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 方差散点图 - L2
    axes[1, 1].scatter(range(len(var_L2_f1)), var_L2_f1, alpha=0.6, label='L2_f1', color='blue')
    axes[1, 1].scatter(range(len(var_L2_f2)), var_L2_f2, alpha=0.6, label='L2_f2', color='red')
    axes[1, 1].set_title('L2_sacROC 各net方差散点图', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('net索引')
    axes[1, 1].set_ylabel('方差')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印方差统计信息
    print("\n=== 方差统计信息 ===")
    print(f"{'组别':<10} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print("-" * 60)
    groups = [('L1_f1', var_L1_f1), ('L2_f1', var_L2_f1), ('L1_f2', var_L1_f2), ('L2_f2', var_L2_f2)]
    for name, data in groups:
        print(f"{name:<10} {np.mean(data):<12.6f} {np.std(data):<12.6f} {np.min(data):<12.6f} {np.max(data):<12.6f}")

# 使用示例
file1 = "./sacROC_batch3.xlsx"
file2 = "./sacROC_batch2.xlsx"  # 替换为实际路径

# 加载数据
data_results = load_and_plot_separately(file1, file2)
all_L1_f1, all_L2_f1, all_L1_f2, all_L2_f2, var_L1_f1, var_L2_f1, var_L1_f2, var_L2_f2 = data_results

# 分别绘制分布图和方差图
print("\n绘制分布图...")
plot_distribution_comparison(all_L1_f1, all_L2_f1, all_L1_f2, all_L2_f2)

print("\n绘制方差图...")
plot_variance_comparison(var_L1_f1, var_L2_f1, var_L1_f2, var_L2_f2)