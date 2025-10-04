# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 19:03:13 2022
this for plot the net's connection map
@author: NZ
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_input2neuron_connectivity(effective_weight,heading_selectivity,rule_name,**kwargs):
    
    Winput2neuron = effective_weight['I2H_weight']
    Winput2neuron = Winput2neuron.detach().cpu().numpy()
    if len(heading_selectivity)== 0:
        sort_ind = np.arange(0,Winput2neuron.shape[0],1,dtype=int)
    else:
        sort_ind = np.argsort(heading_selectivity[:,0])
    # Wlim = np.max(np.abs(Winput2neuron))
    Wlim = np.percentile(np.abs(Winput2neuron), 95)
    show_Winput2neuron = Winput2neuron[sort_ind,:].T
    plt.figure()
    plt.imshow(show_Winput2neuron, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim,origin='lower')
    plt.grid(False)
    plt.colorbar()
    plt.ylabel('from inputs')
    plt.xlabel('to neurons(sort by sensory selectivity)')
    plt.title('input2neurons connectivity')
    if 'figname_append' in kwargs:
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/1hiddenHD_S2H'
    else:  
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/1hiddenHD_S2H'
    plt.savefig(figname+'.png', transparent=True)

def plot_h2h_connectivity(effective_weight,heading_selectivity,saccade_selectivity,rule_name,**kwargs):
    
    Wh2h = effective_weight['H2H_weight']
    Wh2h = Wh2h.detach().cpu().numpy()
    # Wlim = np.max(np.abs(Wh2h))
    Wlim = np.percentile(np.abs(Wh2h), 95)
    if len(heading_selectivity) == 0:
        sort_ind_heading = np.arange(0,Wh2h.shape[1],1,dtype=int)
    else:
        sort_ind_heading = np.argsort(heading_selectivity[:,0])
    if len(saccade_selectivity) == 0:
        sort_ind_saccade = np.arange(0,Wh2h.shape[0])
    else:
        sort_ind_saccade = np.argsort(saccade_selectivity[:,0])
    Wh2h = Wh2h[:,sort_ind_heading]
    show_Wh2h = Wh2h[sort_ind_saccade,:]
    plt.figure()
    # plt.imshow(Wh2h, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim)
    plt.imshow(show_Wh2h, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim,origin='lower')
    plt.grid(False)
    plt.colorbar()
    plt.xlabel('from hidden(sort by selectivity)')
    plt.ylabel('to hidden(sort by sac selectivity)')
    plt.title('hidden2hidden connectivity')
    if 'figname_append' in kwargs:
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/1hiddenHD_H2H'
    else:  
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/1hiddenHD_H2H'
    plt.savefig(figname+'.png', transparent=True)
    
def plot_h2output_connectivity(effective_weight,saccade_selectivity,rule_name,**kwargs):
    
    Wh2output = effective_weight['H2O_weight']
    Wh2output = Wh2output.detach().cpu().numpy()
    # Wlim = np.max(np.abs(Wh2output))
    Wlim = np.percentile(np.abs(Wh2output), 95)
    if len(saccade_selectivity) == 0:
        sort_ind = np.arange(0,Wh2output.shape[1],1,dtype=int)
    else:
        sort_ind = np.argsort(saccade_selectivity[:,0])
    show_Wh2output = Wh2output[:,sort_ind]
    plt.figure()
    plt.imshow(show_Wh2output, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim,origin='lower')
    plt.grid(False)
    # plt.xticks( np.arange(e_size), (sort_ind) )
    plt.colorbar()
    plt.tick_params(axis='x', labelsize=8)   
    plt.xticks(rotation=-90)  

    plt.xlabel('from hidden(sort by sac selectivity)')
    plt.ylabel('to output')
    plt.title('hidden2output connectivity')
    if 'figname_append' in kwargs:
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/H2O'
    else:  
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/H2O'
    plt.savefig(figname+'.png', transparent=True)
    
def plot_connectivity(state_dict,**kwargs):
    
    Winput2neuron = state_dict['rnn.input2h.weight']
    Winput2neuron = Winput2neuron.detach().cpu().numpy()
    
    Wh2h = state_dict['rnn.h2h.weight']
    Wh2h = Wh2h.detach().numpy()
    
    Wh2output = state_dict['fc.weight']
    Wh2output = Wh2output.detach().numpy()
    
    plt.figure()
    plt.imshow(Winput2neuron, cmap = 'bwr_r',origin='lower')
    plt.grid(False)
    plt.colorbar()
    plt.xticks(rotation=-90)
    
def plot_forward_connectivity(effective_weight,hidden1_size,heading_selectivity,saccade_selectivity,rule_name,**kwargs):
    
    Wh2h = effective_weight['H2H_weight']
    Wh2h = Wh2h.detach().numpy()
    Wh12h2 = Wh2h[hidden1_size:,:hidden1_size]
    Wlim = np.max(np.abs(Wh12h2))
    
    if len(heading_selectivity) == 0:
        sort_ind_heading = np.arange(0,Wh12h2.shape[1],1,dtype=int)
    else:
        sort_ind_heading = np.argsort(heading_selectivity[:,0])
    if len(saccade_selectivity) == 0:
        sort_ind_saccade = np.arange(0,Wh12h2.shape[0])
    else:
        sort_ind_saccade = np.argsort(saccade_selectivity[:,0])
    Wh12h2 = Wh12h2[:,sort_ind_heading]
    show_Wh12h2 = Wh12h2[sort_ind_saccade,:]
    plt.figure()
    # plt.imshow(Wh2h, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim)
    plt.imshow(show_Wh12h2, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim,origin='lower')
    plt.grid(False)
    plt.colorbar()
    plt.xlabel('from hidden(sort by selectivity)')
    plt.ylabel('to hidden(sort by sac selectivity)')
    plt.title('forward connectivity')
    if 'figname_append' in kwargs:
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/1hiddenHD_H12H2'
    else:  
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/1hiddenHD_H12H2'
    plt.savefig(figname+'.png', transparent=True)
    
def plot_feedback_connectivity(effective_weight,hidden1_size,heading_selectivity,choice_selectivity,rule_name,**kwargs):
    
    Wh2h = effective_weight['H2H_weight']
    Wh2h = Wh2h.detach().cpu().numpy()

    Wh22h1 = Wh2h[:hidden1_size,hidden1_size:]
    Wlim = np.max(np.abs(Wh22h1))
    
    if len(heading_selectivity) == 0:
        sort_ind_heading = np.arange(0,Wh22h1.shape[1],1,dtype=int)
        # 如果没有heading selectivity数据，不绘制分界线
        plot_heading_divider = False
    else:
        sort_ind_heading = np.argsort(heading_selectivity[:,0])
        sorted_heading = heading_selectivity[sort_ind_heading, 0]
        positive_indices = np.where(sorted_heading > 0)[0]
        first_positive = positive_indices[0]
        heading_sign_changes = [first_positive]
        plot_heading_divider = len(heading_sign_changes) > 0
    if len(choice_selectivity) == 0:
        sort_ind_choice = np.arange(0,Wh22h1.shape[0])
        plot_choice_divider = False
    else:
        sort_ind_choice = np.argsort(choice_selectivity[:,0])
        sorted_choice = choice_selectivity[sort_ind_choice, 0]
        positive_indices = np.where(sorted_choice > 0)[0]
        first_positive = positive_indices[0]
        choice_sign_changes = [first_positive]
        plot_choice_divider = len(choice_sign_changes) > 0
    Wh22h1 = Wh22h1[:,sort_ind_choice]
    show_Wh22h1 = Wh22h1[sort_ind_heading,:]
    plt.figure()
    # plt.imshow(Wh2h, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim)
    plt.imshow(show_Wh22h1, cmap = 'bwr_r',vmin=-Wlim, vmax=Wlim,origin='lower')
    
    # 标记heading selectivity符号变化的分界线
    if plot_heading_divider:
        for change_pos in heading_sign_changes:
            plt.axhline(y=change_pos - 0.5, color='black', linewidth=2, linestyle='--', alpha=0.8)
            # 添加文本标签说明分界线含义
            plt.text(show_Wh22h1.shape[1] * 1.02, change_pos - 0.5, 'Heading\nZero', 
                    verticalalignment='center', fontsize=8, color='black')
    
    # 标记choice selectivity符号变化的分界线
    if plot_choice_divider:
        for change_pos in choice_sign_changes:
            plt.axvline(x=change_pos - 0.5, color='black', linewidth=2, linestyle='--', alpha=0.8)
            # 添加文本标签说明分界线含义
            plt.text(change_pos - 0.5, show_Wh22h1.shape[0] * 1.02, 'choice\nZero', 
                    horizontalalignment='center', verticalalignment='bottom', fontsize=8, color='black')
            
 
    
    plt.grid(False)
    plt.colorbar()
    plt.xlabel('from hidden(sort by choice)')
    plt.ylabel('to hidden(sort by heading)')
    plt.title('forward connectivity')
    if 'figname_append' in kwargs:
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/'+kwargs['figname_append']+'/H22H1'
    else:  
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','')+'/HD_H22H1'
    plt.savefig(figname+'.png', transparent=True)    
    
    # ----------------------------
    # 创建符号 mask
    # ----------------------------
    if len(heading_selectivity) > 0 and len(choice_selectivity) > 0:
        heading_sign = np.sign(sorted_heading)
        choice_sign = np.sign(sorted_choice)
        sign_match_mask = np.outer(heading_sign, choice_sign) > 0
        sign_opposite_mask = ~sign_match_mask
    else:
        sign_match_mask = np.ones_like(Wh22h1, dtype=bool)
        sign_opposite_mask = np.zeros_like(Wh22h1, dtype=bool)
    
    # ----------------------------
    # 提取两组权重
    # ----------------------------
    weights_same = Wh22h1[sign_match_mask]
    weights_opposite = Wh22h1[sign_opposite_mask]
    
    # ----------------------------
    # 绘图
    # ----------------------------
    
    plt.figure(figsize=(6,4))
    plt.hist(weights_same, bins=100, alpha=0.6, label='same sign', color='blue', log=True)
    plt.hist(weights_opposite, bins=100, alpha=0.6, label='opposite sign', color='orange', log=True)
    plt.xlabel('Feedback weights (H2->H1)')
    plt.ylabel('Count (log scale)')
    plt.title('Distribution of feedback weights (log scale)')
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize=(4,5))
    plt.violinplot([weights_same, weights_opposite], showmedians=True)
    plt.xticks([1,2], ['same', 'opposite'])
    plt.ylabel('Feedback weights (H2->H1)')
    plt.title('Weight distribution by sign match')
    plt.show()
    
    epsilon = 1e-6  # 根据你的数据规模调整
    weights_same_log = weights_same[weights_same>0.01]
    weights_opposite_log = weights_opposite[weights_opposite>0.01]
    
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(weights_same_log, weights_opposite_log, alternative='two-sided')
    print(f"Mann-Whitney U: stat={stat}, p={p}")
    
