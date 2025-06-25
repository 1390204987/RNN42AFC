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
    Winput2neuron = Winput2neuron.detach().numpy()
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
    Wh2h = Wh2h.detach().numpy()
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
    Wh2output = Wh2output.detach().numpy()
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
    Winput2neuron = Winput2neuron.detach().numpy()
    
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