# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:17:44 2024
for find the proper parameter visually
@author: NaN
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
# 读取 Excel 数据
df = pd.read_excel("./checkpoint_batchnew1.xlsx",engine='openpyxl')
# df = pd.read_excel("./noise_corr4.xlsx",engine='openpyxl')

# 填充 'layer1cor' 列下的 NaN 值为 0
# show_para = "fack_l2choicetime"
# show_para = "fack_l1choicetime"
# show_para = "fack_l2sactime"
# show_para = "fack_l1sactime"
# show_para = "l1_l2sac"
show_para = "l2sac_choice"
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
# vmin = df[show_para].min()  # 最小值
# vmax = df[show_para].max()  # 最大值
vmin = -1500  # 最小值
vmax = 1500  # 最大值
# 创建 TwoSlopeNorm，设置 0 为分界点
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
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

for z in [0, 1, 2, 3, 4]:
    for j in [0, 1, 2, 3, 4]:
        ax_sub = axs[z, j]
        # ax_sub.set_title(f'Sub Plot for (Z={z}, J={j})')

        # 获取对应 (Z, J) 的数据
        subset_df = df[(df['Z'] == z) & (df['J'] == j)]

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
corr_j,p_j = pearsonr(df[show_para], df['J_replaced'])
corr_z,p_z = pearsonr(df[show_para], df['Z_replaced'])
corr_joverz,p_joverz = pearsonr(df[show_para], df['J_over_Z'])
print(f"Pearson 相关性 (J): {corr_j:.3f}",f"P 值: {p_j:.4f}")
print(f"Pearson 相关性 (Z): {corr_z:.3f}",f"P 值: {p_z:.4f}")
print(f"Pearson 相关性 (joverZ): {corr_joverz:.3f}",f"P 值: {p_joverz:.4f}")

# 显示图形
plt.show()
plt.savefig("./lunwenfigure/l1_l2sac.svg")