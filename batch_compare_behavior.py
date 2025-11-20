# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:24:08 2025
compare batch net behavior output difference
@author: NaN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from SetFigure import SetFigure

# From sns.dark_palette("light blue", 3, input="xkcd")
BLUES = [np.array([0.13333333, 0.13333333, 0.13333333, 1.        ]),
         np.array([0.3597078 , 0.47584775, 0.56246059, 1.        ]),
         np.array([0.58431373, 0.81568627, 0.98823529, 1.        ])]


# 加载数据
df_bias1 = pd.read_excel("./behavior_batch2.xlsx", sheet_name='behavior_bias')
df_bias2 = pd.read_excel("./behavior_batch2cut.xlsx", sheet_name='behavior_bias')

df_threshold1 = pd.read_excel("./behavior_batch2.xlsx", sheet_name='behavior_threshold')
df_threshold2 = pd.read_excel("./behavior_batch2cut.xlsx", sheet_name='behavior_threshold')

# 处理数据
bias_values1 = df_bias1.values.flatten()
bias_values2 = df_bias2.values.flatten()
bias_values1 = bias_values1[~np.isnan(bias_values1)]
bias_values2 = bias_values2[~np.isnan(bias_values2)]

threshold_values1 = df_threshold1.values.flatten()
threshold_values2 = df_threshold2.values.flatten()
threshold_values1 = threshold_values1[~np.isnan(threshold_values1)]
threshold_values2 = threshold_values2[~np.isnan(threshold_values2)]

# 进行t检验
t_bias, p_bias = stats.ttest_ind(bias_values1, bias_values2)

t_threshold, p_threshold = stats.ttest_ind(threshold_values1, threshold_values2)

# # 极简箱线图+散点图 for bias
# fig1 = plt.figure(figsize=(8, 6))

# # 箱线图
# plt.boxplot([bias_values1, bias_values2], 
#             labels=['Batch2_2', 'Batch2cut'],widths=0.8)

# # 散点
# x1 = np.random.normal(1, 0.08, len(bias_values1))
# x2 = np.random.normal(2, 0.08, len(bias_values2))
# plt.scatter(x1, bias_values1, alpha=0.5, color='b', s=15, marker='o')
# plt.scatter(x2, bias_values2, alpha=0.5, color='m', s=15, marker='o')

# plt.ylabel('Behavior Bias')
# plt.title(f'p = {p_bias:.4f}')
# SetFigure(15)
# plt.grid(False)
# plt.tight_layout()
# plt.show()
# plt.savefig('./lunwenfigure/behavsacbias_cut_uncut.svg', transparent=True)

# # 极简箱线图+散点图 for threshold
# fig2 = plt.figure(figsize=(8, 6))

# # 箱线图
# plt.boxplot([threshold_values1, threshold_values2], 
#             labels=['Batch2', 'Batch2cut'],widths=0.8)

# # 散点
# x1 = np.random.normal(1, 0.08, len(threshold_values1))
# x2 = np.random.normal(2, 0.08, len(threshold_values2))
# plt.scatter(x1, threshold_values1, alpha=0.5, color='b', s=15, marker='o')
# plt.scatter(x2, threshold_values2, alpha=0.5, color='m', s=15, marker='o')

# plt.ylabel('Behavior Threshold')
# plt.title(f'p = {p_threshold:.4f}')
# SetFigure(15)
# plt.grid(False)
# plt.tight_layout()
# plt.show()
# plt.savefig('./lunwenfigure/behavsacthreshold_cut_uncut.svg', transparent=True)


from scipy import stats
from scipy.optimize import curve_fit

def psychophysical_fit(ax,x,y,Color):

    # 定义高斯CDF函数
    cdf_gaussian = lambda x, mu, sigma: stats.norm.cdf(x, mu, sigma)


    # --- 曲线拟合 ---
    # 设置初始参数猜测值 [mu, sigma]
    initial_guess = [np.mean(x), np.std(x)]

    
    # 使用curve_fit进行拟合
    try:
        popt, pcov = curve_fit(cdf_gaussian, x, y, 
                              p0=initial_guess, 
                              bounds=([-np.inf, 0.001], [np.inf, np.inf]))
    
        mu_fit, sigma_fit = popt
        # Compute threshold (difference from 50% to 75%)
        z75 = stats.norm.ppf(0.75)  # ≈ 0.674
        threshold = z75 * sigma_fit
        print(f"拟合参数: mu = {mu_fit:.3f}, threshold = {threshold:.3f}")

        # 生成平滑曲线用于绘图
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = cdf_gaussian(x_fit, mu_fit, sigma_fit)

        ax.plot(x_fit, y_fit, 'r-', color=Color, linewidth=2, label='')
        


    except Exception as e:
        print(f"拟合失败: {e}")


df_correct_rate =  pd.read_excel("./behavior_batch2.xlsx", sheet_name='correct_rate')
df_xdatas = pd.read_excel("./behavior_batch2.xlsx", sheet_name='behavior_xdatas')
df_ydatas = pd.read_excel("./behavior_batch2.xlsx", sheet_name='behavior_ydatas')
df_sacydatas = pd.read_excel("./behavior_batch2.xlsx", sheet_name='behavior_sacydatas')

df_correct_ratecut =  pd.read_excel("./behavior_batch2cut.xlsx", sheet_name='correct_rate')
df_xdatascut = pd.read_excel("./behavior_batch2cut.xlsx", sheet_name='behavior_xdatas')
df_ydatascut = pd.read_excel("./behavior_batch2cut.xlsx", sheet_name='behavior_ydatas')
df_sacydatascut = pd.read_excel("./behavior_batch2cut.xlsx", sheet_name='behavior_sacydatas')


# --- xdata 作为 index（取任意一列即可，因为所有网络一致） ---
x_index = df_xdatas.iloc[:, 0]   # 第一列就是 xdata

correct_rate = df_correct_rate.values.flatten()
correct_rate_cut = df_correct_ratecut.values.flatten()

mean_cr = correct_rate.mean(axis=0)
mean_cr_cut = correct_rate_cut.mean(axis=0)

# --- ydata 矩阵：全部列 ---
ydatas_matrix = df_ydatas.copy()
ydatascut_matrix = df_ydatascut.copy()

# --- sacy 矩阵 ---
# sacy_matrix = df_sacydatas.copy()
# sacy_matrix.index = x_index

# --- 计算每个 x 的 std（基于 ydatas_matrix） ---
std_ydatas = ydatas_matrix.std(axis=1)
std_ydatascut = ydatascut_matrix.std(axis=1)

mean_ydatas = ydatas_matrix.mean(axis=1)
mean_ydatascut = ydatascut_matrix.mean(axis=1)
# mean_sacydatas = sacy_matrix.mean(axis=1)

x_index = x_index.to_numpy() - np.pi
mean_ydatas = mean_ydatas.to_numpy()
std_y_array = std_ydatas.to_numpy()

mean_ydatascut = mean_ydatascut.to_numpy()
std_y_arraycut = std_ydatascut.to_numpy()

# --- 绘图 ---
fig3 = plt.figure(figsize=(8,6))
ax = fig3.add_subplot(111)  # 创建axes对象

ax.errorbar(
    x_index,
    mean_ydatas,
    yerr=std_y_array,
    fmt='o',
    capsize=3,
    linewidth=1.5,
    markersize=5,
    color = 'b'
)

ax.errorbar(
    x_index,
    mean_ydatascut,
    yerr=std_y_arraycut,
    fmt='o',
    capsize=3,
    linewidth=1.5,
    markersize=5,
    color= 'm'
)

psychophysical_fit(ax,x_index,mean_ydatas,'b')
psychophysical_fit(ax,x_index,mean_ydatascut,'m')

ax.text(0.98, 0.02, f'mean_cr = {mean_cr:.3f}\nmean_cr_cut = {mean_cr_cut:.3f}', 
        transform=ax.transAxes,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=9)

xmin = np.min(x_index)
xmax = np.max(x_index)
ax.set_ylim([-0.05,1.05])
ax.set_xlim([xmin*1.1,xmax*1.1])
ax.set_yticks([0,0.5,1])
ax.set_xticks([xmin, 0, xmax])
SetFigure(15)

plt.show()

plt.savefig('./lunwenfigure/psycho_cut_uncut.pdf', transparent=True)

