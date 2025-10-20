# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:31:45 2025

@author: NaN
"""
import matplotlib.pyplot as plt

def SetFigure(size=20):
    """
    Beautify the current matplotlib figure. Inspired by MATLAB SetFigure.
    """
    fig = plt.gcf()
    fig.patch.set_facecolor('white')  # Set background color to white

    # Set font sizes for all text elements
    for ax in fig.get_axes():
        # ax.tick_params(direction='out', length=6, width=1.5)
        ax.set_facecolor('none')  # Set axes background to transparent
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.title.set_fontsize(size)
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        ax.tick_params(labelsize=size)
        ax.tick_params(axis='both', which='both', 
                       bottom=True, top=False, 
                       left=True, right=False,
                       direction='out',         # 刻度线朝外
                       length=5,               # 刻度线长度
                       width=1.5)              # 刻度线宽度

    # Set error bar cap size (for errorbar plots)
    for line in fig.findobj(match=plt.Line2D):
        if hasattr(line, '_capstyle'):
            line.set_markersize(10)

    # Turn off box around legends
    for legend in fig.legends:
        legend.set_frame_on(False)
