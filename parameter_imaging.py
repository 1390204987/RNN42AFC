# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:17:44 2024
for find the proper parameter visually
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
# 读取 Excel 数据
df = pd.read_excel("./choice_divergence1_unsmooth.xlsx",engine='openpyxl')
# df = pd.read_excel("./noise_corr4.xlsx",engine='openpyxl')

# 填充 'layer1cor' 列下的 NaN 值为 0
# show_para = "fack_l2choicetime"
# show_para = "fack_l1choicetime"
# show_para = "fack_l2sactime"
# show_para = "fack_l1sactime"
show_para = "delta_sac"
# show_para = "delta_abs"
# show_para = "deltaabs_sac"
imagingdata = df[show_para]
# imagingdata = df['layer2noisecorr']
# imagingdata.fillna(0,inplace=True)
# df['layer2noisecorr'].fillna(0, inplace=True)

# 'netid' column contains values from 0 to 9999
df['J'] = (df['netid'] // 1000).astype(int)  # Extract thousands digit
df['Z'] = ((df['netid'] % 1000) // 100).astype(int)  # Extract hundreds digit
df['Y'] = ((df['netid'] % 100) // 10).astype(int)  # Extract tens digit
df['X'] = (df['netid'] % 10).astype(int)  # Extract units digit
# df['X'] = (df['netid'] // 1000).astype(int)  # Extract thousands digit
# df['Y'] = ((df['netid'] % 1000) // 100).astype(int)  # Extract hundreds digit
# df['Z'] = ((df['netid'] % 100) // 10).astype(int)  # Extract tens digit
# df['J'] = (df['netid'] % 10).astype(int)  # Extract units digi
# df['Z'] = (df['netid'] // 1000).astype(int)  # Extract thousands digit
# df['Y'] = ((df['netid'] % 1000) // 100).astype(int)  # Extract hundreds digit
# df['J'] = ((df['netid'] % 100) // 10).astype(int)  # Extract tens digit
# df['X'] = (df['netid'] % 10).astype(int)  # Extract units digit
# df['Y'] = (df['netid'] // 1000).astype(int)  # Extract thousands digit
# df['Z'] = ((df['netid'] % 1000) // 100).astype(int)  # Extract hundreds digit
# df['X'] = ((df['netid'] % 100) // 10).astype(int)  # Extract tens digit
# df['J'] = (df['netid'] % 10).astype(int)  # Extract units digit
# 获取唯一的 (Z, J) 组合
unique_zj = df[['Z', 'J']].drop_duplicates()

# 创建主图和子图
# fig, axs = plt.subplots(10, 10, figsize=(13, 13))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)


# 获取颜色映射和归一化器
# cmap = cm.viridis
# norm = Normalize(vmin=imagingdata.min(), vmax=imagingdata.max())
# norm = Normalize(vmin=df['layer2noisecorr'].min(), vmax=df['layer2noisecorr'].max())

# 添加共同的颜色条
# 假设你的数据范围是 vmin, vmax
vmin = df[show_para].min()  # 最小值
vmax = df[show_para].max()  # 最大值
# vmin = -1500  # 最小值
# vmax = 1500  # 最大值
# 创建 TwoSlopeNorm，设置 0 为分界点
# norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2,  vmax=vmax)
# 选择一个 diverging colormap（如 'RdBu'，红色表示正，蓝色表示负）
# cmap = 'RdBu'
# cmap = 'coolwarm'
cmap = 'bwr'
# 计算数据的均值和标准差

# mean_value = imagingdata.mean()
# std_value = imagingdata.std()
# 设置颜色映射的范围为均值减3倍标准差到均值加3倍标准差
# vmin = mean_value - 3 * std_value
# vmax = mean_value + 3 * std_value
# norm = Normalize(vmin=vmin, vmax=vmax)
fig, axs = plt.subplots(5, 5, figsize=(5, 5), 
                        gridspec_kw={'wspace': 1, 'hspace': 1})  # 调整子图间距

# 在每个 (Z, J) 位置添加子图, axs[0, 0] → 左上角 的子图（第 0 行，第 0 列）。axs[2, 5] → 第 2 行，第 5 列 的子图。axs[9, 9] → 右下角 的子图（第 9 行，第 9 列）。

# 创建两个5x5的矩阵来存储比例
positive_ratio_matrix = np.zeros((5, 5))
negative_ratio_matrix = np.zeros((5, 5))

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
        else:
            positive_ratio = 0
            negative_ratio = 0
        
        # 存储到矩阵中
        positive_ratio_matrix[z, j] = positive_ratio
        negative_ratio_matrix[z, j] = negative_ratio
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
    1: 0.3,
    2: 0.5,
    3: 0.7,
    4: 0.9
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
    
#将矩阵展为一维数组进行相关性分析
positive_ratios_flat = positive_ratio_matrix.flatten()
negative_ratios_flat = negative_ratio_matrix.flatten()
J_over_Z_flat = J_over_Z_matrix.flatten()
J_flat = J_matrix.flatten()
Z_flat = Z_matrix.flatten() 

# 1. positive_ratio 与 J,Z,J_over_Z 的相关性
pos_J_corr, pos_J_p = pearsonr(positive_ratios_flat, J_flat)
pos_Z_corr, pos_Z_p = pearsonr(positive_ratios_flat, Z_flat)
pos_JoZ_corr, pos_JoZ_p = pearsonr(positive_ratios_flat, J_over_Z_flat)

# 1. negtive_ratio 与 J,Z,J_over_Z 的相关性
neg_J_corr, neg_J_p = pearsonr(negative_ratios_flat, J_flat)
neg_Z_corr, neg_Z_p = pearsonr(negative_ratios_flat, Z_flat)
neg_JoZ_corr, neg_JoZ_p = pearsonr(negative_ratios_flat, J_over_Z_flat)



def format_p_value(p):
    if p < 0.01:
        return f"{p:.2e}"
    else:
        return f"{p:.2f}"

print(f"Pearson 相关性 (J): {corr_j:.2f}", f"P 值: {format_p_value(p_j)}")
print(f"Pearson 相关性 (Z): {corr_z:.2f}", f"P 值: {format_p_value(p_z)}")
print(f"Pearson 相关性 (joverZ): {corr_joverz:.2f}", f"P 值: {format_p_value(p_joverz)}")

print(f"Pearson 相关性 (posJ): {pos_J_corr:.2f}", f"P 值: {format_p_value(pos_J_p)}")
print(f"Pearson 相关性 (posZ): {pos_Z_corr:.2f}", f"P 值: {format_p_value(pos_Z_p)}")
print(f"Pearson 相关性 (posJoZ): {pos_JoZ_corr:.2f}", f"P 值: {format_p_value(pos_JoZ_p)}")

print(f"Pearson 相关性 (negJ): {neg_J_corr:.2f}", f"P 值: {format_p_value(neg_J_p)}")
print(f"Pearson 相关性 (negZ): {neg_Z_corr:.2f}", f"P 值: {format_p_value(neg_Z_p)}")
print(f"Pearson 相关性 (negJoZ): {neg_JoZ_corr:.2f}", f"P 值: {format_p_value(neg_JoZ_p)}")

print(f"总数 (pos_num): {positive_num:.2f}")
print(f"总数 (neg_num): {negative_num:.2f}")

# 显示图形
plt.show()
plt.savefig("./lunwenfigure/l2_l1abs.svg")

# 计算每个J值的平均positive ratio
j_values = [0, 1, 2, 3, 4]
j_positive_ratios = []
j_negative_ratios = []

for j in j_values:
    # 获取该J值对应的所有positive ratio（跨所有Z值）
    j_positive_data = positive_ratio_matrix[:, j]
    j_negative_data = negative_ratio_matrix[:, j]
    
    # 计算平均值（去掉NaN值）
    j_positive_mean = np.nanmean(j_positive_data)
    j_negative_mean = np.nanmean(j_negative_data)
    
    j_positive_ratios.append(j_positive_mean)
    j_negative_ratios.append(j_negative_mean)
    
# 绘制柱状图
fig, (ax) = plt.subplots(1, 1, figsize=(7, 3))
# 1. 每个J值的positive ratio柱状图
x_positions = np.arange(5)  # 0,1,2,3,4 对应5列
bars = ax.bar(x_positions, j_positive_ratios, color='skyblue', alpha=0.7, edgecolor='navy', width=0.5)
ax.set_ylabel('Positive Ratio')
ax.set_xlim(-0.5, 4.5)
# 调整布局，添加紧凑布局
plt.tight_layout(pad=2.0)  # 增加内边距
SetFigure(size=12)
plt.show()
# plt.savefig("./lunwenfigure/deltaabs_sac.svg")

# 绘制柱状图
fig, (ax) = plt.subplots(1, 1, figsize=(7, 3))
# 1. 每个J值的positive ratio柱状图
x_positions = np.arange(5)  # 0,1,2,3,4 对应5列
bars = ax.bar(x_positions, j_negative_ratios, color='skyblue', alpha=0.7, edgecolor='navy', width=0.5)
ax.set_ylabel('Positive Ratio')
ax.set_xlim(-0.5, 4.5)
# 调整布局，添加紧凑布局
plt.tight_layout(pad=2.0)  # 增加内边距
SetFigure(size=12)
plt.show()

# 使用箱形图展示完整的数据分布
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
# 准备数据
data_to_plot = [positive_ratio_matrix[:, j] for j in j_values]
# 绘制箱形图
boxplot = ax.boxplot(data_to_plot, positions=j_values, widths=0.5,
                     patch_artist=True, showmeans=True,
                     meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"navy"})

# 设置箱形图颜色
for box in boxplot['boxes']:
    box.set_facecolor('skyblue')
    box.set_alpha(0.7)

ax.set_ylabel('Positive Ratio')
ax.set_xlabel('J Value')
ax.set_xticks(j_values)
ax.set_xticklabels([f'J={i}' for i in j_values])
ax.set_xlim(-0.5, 4.5)

plt.tight_layout(pad=2.0)
SetFigure(size=12)
plt.show()