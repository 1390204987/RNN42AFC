# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:01:44 2025

@author: NaN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
from SetFigure import SetFigure
import seaborn as sns


# 创建新的DataFrame
df = pd.DataFrame()

df_bias = pd.read_excel("./behavior_batch1.xlsx", sheet_name='behavior_bias')
df_threshold = pd.read_excel("./behavior_batch1.xlsx", sheet_name='behavior_threshold')

# 提取所有列名中的数字作为 netid
df['netid'] = [int(col.split('_')[1]) for col in df_bias.columns if col.startswith('net_')]
# 提取第一行的 bias 和 threshold 值（根据你的数据结构）
df['bias'] = df_bias.iloc[0].values
df['threshold'] = df_threshold.iloc[0].values

# show_para = "bias"
show_para = "threshold"

df['J'] = (df['netid'] // 1000).astype(int)  # Extract thousands digit
df['Z'] = ((df['netid'] % 1000) // 100).astype(int)  # Extract hundreds digit
df['Y'] = ((df['netid'] % 100) // 10).astype(int)  # Extract tens digit
df['X'] = (df['netid'] % 10).astype(int)  # Extract units digit

# 添加共同的颜色条
# 假设你的数据范围是 vmin, vmax
vmin = df[show_para].min()  # 最小值
vmax = df[show_para].max()  # 最大值
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df_clean
vmax = np.max(df_clean[show_para])
norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2,  vmax=vmax)
cmap = 'bwr'
fig, axs = plt.subplots(5, 5, figsize=(5, 5), 
                        gridspec_kw={'wspace': 1, 'hspace': 1})  # 调整子图间距

# 在每个 (Z, J) 位置添加子图, axs[0, 0] → 左上角 的子图（第 0 行，第 0 列）。axs[2, 5] → 第 2 行，第 5 列 的子图。axs[9, 9] → 右下角 的子图（第 9 行，第 9 列）。

# 创建两个5x5的矩阵来存储比例
positive_ratio_matrix = np.zeros((5, 5))
negative_ratio_matrix = np.zeros((5, 5))
mean_matrix = np.full((5, 5),np.nan)
# 创建矩阵用于相关性分析
J_matrix = np.zeros((5,5))
Z_matrix = np.zeros((5,5))
J_over_Z_matrix = np.zeros((5,5))

J_value = np.linspace(0.1,1,num=5)
Z_value = np.linspace(0.1,1,num=5)

positive_num = 0
negative_num = 0

for z in range(5):
    for j in [0, 1, 2, 3, 4]:
        ax_sub = axs[z, j]
        # ax_sub.set_title(f'Sub Plot for (Z={z}, J={j})')

        # 获取对应 (Z, J) 的数据
        subset_df = df[(df['Z'] == z) & (df['J'] == j)]
        
        # 统计大于0和小于0的比例
        total_count = len(subset_df)
        if total_count > 0:
            positive_ratio = len(subset_df[subset_df[show_para] > 0]) / total_count
            negative_ratio = len(subset_df[subset_df[show_para] < 0]) / total_count
            positive_num = positive_num + len(subset_df[subset_df[show_para] > 0]) 
            negative_num = negative_num + len(subset_df[subset_df[show_para] < 0]) 
            masked_data = np.ma.masked_invalid(subset_df[show_para])
            mean_value = masked_data.mean()
        else:
            positive_ratio = 0
            negative_ratio = 0
            mean_value = np.nan
        # 存储到矩阵中
        positive_ratio_matrix[z, j] = positive_ratio
        negative_ratio_matrix[z, j] = negative_ratio
        mean_matrix[z, j] = mean_value 
        J_matrix[z, j] = J_value[j]
        Z_matrix[z, j] = Z_value[z]
        J_over_Z_matrix[z, j] = J_value[j]/Z_value[z]
        # 绘制子图
        scatter_sub = ax_sub.scatter(subset_df['X'], subset_df['Y'], c=subset_df[show_para], cmap=cmap, s=20, norm=norm)
        # scatter_sub = ax_sub.scatter(subset_df['X'], subset_df['Y'], c=subset_df['layer2noisecorr'], cmap=cmap, s=20, norm=norm)
        # 去掉子图的坐标轴
        ax_sub.axis('off')



cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # 调整颜色条的位置
cb = plt.colorbar(scatter_sub, cax=cax, label='time')

# # 调整布局
# plt.tight_layout()

# 定义替换规则
replacement = {
    0: 0.1,
    1: 0.325,
    2: 0.55,
    3: 0.775,
    4: 1
}
# 替换 Z 的值
df['Z_replaced'] = df['Z'].map(replacement)

# 替换 j 的值
df['J_replaced'] = df['J'].map(replacement)
check = df['J_replaced'] / df['Z_replaced']
df['J_over_Z'] = df['J_replaced'] / df['Z_replaced']
# 创建没有 NaN 的子集
valid_data_j = df[[show_para, 'J_replaced']].dropna()
valid_data_z = df[[show_para, 'Z_replaced']].dropna()
valid_data_joverz = df[[show_para, 'J_over_Z']].dropna()

if len(valid_data_j) > 0:
    corr_j, p_j = pearsonr(valid_data_j[show_para], valid_data_j['J_replaced'])
else:
    print("J 数据全为 NaN")

if len(valid_data_z) > 0:
    corr_z, p_z = pearsonr(valid_data_z[show_para], valid_data_z['Z_replaced'])

else:
    print("Z 数据全为 NaN")

if len(valid_data_joverz) > 0:
    corr_joverz, p_joverz = pearsonr(valid_data_joverz[show_para], valid_data_joverz['J_over_Z'])
else:
    print("J_over_Z 数据全为 NaN")
    
def plot_box(plot_data,x_name,show_para,corr,p):
    
    filtered_data = plot_data[plot_data[show_para] <= 1.0]
    # 创建透明箱型图
    sns.boxplot(data=filtered_data, x=x_name, y=show_para,
                color='lightblue', width=0.6, fliersize=0,  # 统一使用淡蓝色
                boxprops=dict(alpha=0.6, facecolor='lightblue', edgecolor='blue'),  # 箱体颜色
                whiskerprops=dict(alpha=0.7, color='blue'),  # 须线颜色
                capprops=dict(alpha=0.7, color='blue'),  # 须线帽颜色
                medianprops=dict(color='blue', alpha=0.8, linewidth=2))  # 中位数线
    # 叠加散点图，使用不同的颜色
    scatter_plot = sns.stripplot(data=filtered_data, x=x_name, y=show_para,
                                palette='dark:black', alpha=0.7, size=2, 
                                jitter=0.2, edgecolor='gray', linewidth=0.5)
    # 格式化 p 值显示
    if p < 0.001:
        p_value_text = 'p < 0.001'
    else:
        p_value_text = f'p = {p:.3f}'
    # 在图上添加 Pearson 相关系数和 p 值
    text_str = f'Pearson r = {corr:.3f}\n{p_value_text}'
    plt.text(0.98, 0.98, text_str, 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                      edgecolor='gold', linewidth=2))    
    # 获取唯一的J_replaced值并设置x轴刻度
    unique_j_values = sorted(filtered_data[x_name].unique())
    plt.xticks(ticks=range(len(unique_j_values)), labels=[f'{x:.2f}' for x in unique_j_values])
    SetFigure(15)
    
plt.figure()
# plt.scatter( valid_data_j['J_replaced'],valid_data_j[show_para])
plot_box(valid_data_j,'J_replaced',show_para,corr_j,p_j)
SetFigure(15)
plt.show()
plt.savefig("./lunwenfigure/J_behavior.svg")

plt.figure()
# plt.scatter( valid_data_z['Z_replaced'],valid_data_z[show_para])
plot_box(valid_data_z,'Z_replaced',show_para,corr_z,p_z)
plt.show()
plt.savefig("./lunwenfigure/Z_behavior.svg")

plt.figure()
# plt.scatter( valid_data_joverz['J_over_Z'],valid_data_joverz[show_para])
plot_box(valid_data_joverz,'J_over_Z',show_para,corr_joverz,p_joverz)
plt.show()
plt.savefig("./lunwenfigure/JoverZ_behavior.svg")

def format_p_value(p):
    if p < 0.01:
        return f"{p:.2e}"
    else:
        return f"{p:.2f}"
    


# print(f"Pearson 相关性 (J): {corr_j:.2f}", f"P 值: {format_p_value(p_j)}")
# print(f"Pearson 相关性 (Z): {corr_z:.2f}", f"P 值: {format_p_value(p_z)}")
# print(f"Pearson 相关性 (joverZ): {corr_joverz:.2f}", f"P 值: {format_p_value(p_joverz)}")





    


