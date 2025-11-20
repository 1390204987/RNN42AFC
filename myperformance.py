# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:13:16 2022

@author: NaN
"""
"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import scipy.stats as stats
import roc_tool
from scipy.optimize import curve_fit
from SetFigure import SetFigure

from mytask import generate_trials, rule_name, get_dist

# import mynetwork1hidden
# from mynetwork1hidden import Net

# import mynetwork_new1
# from mynetwork_new1 import Net

import mynetwork_new3
from mynetwork_new3 import Net

# import mynetwork6
# from mynetwork6 import Net

# import mynetwork_new2
# from mynetwork_new2 import Net

# import mynetwork1hidden
# from mynetwork1hidden import Net

# import mynetworkhebb
# from mynetworkhebb import Net

plt.close('all')

save = False
THETA = 0.3 * np.pi

# From sns.dark_palette("light blue", 3, input="xkcd")
BLUES = [np.array([0.13333333, 0.13333333, 0.13333333, 1.        ]),
         np.array([0.3597078 , 0.47584775, 0.56246059, 1.        ]),
         np.array([0.58431373, 0.81568627, 0.98823529, 1.        ])]



def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)



################ Psychometric - Varying Coherence #############################
def _psychometric_dm(model_dir, rule, stim_mod, params_list, batch_shape):
    """Base function for computing psychometric performance in 2AFC tasks

    Args:
        model_dir : model name
        rule : task to analyze
        params_list : a list of parameter dictionaries used for the psychometric mode
        batch_shape : shape of each batch. Each batch should have shape (n_rep, ...)
        n_rep is the number of repetitions that will be averaged over

    Return:
        ydatas: list of performances
    """
    print('Starting psychometric analysis of the {:s} task...'.format(rule_name[rule]))
    
    modelparams = torch.load(model_dir)
    state_dict = modelparams["state_dict"]
    hp = modelparams["hp"]
    hp["sigma_x"] = 0 # in the testing set don't add noise to the input
    net = Net(hp,dt = hp['dt'])
    #remove prefixe "module"
    state_dict = {k.replace("module.",""): v for k, v in state_dict.items()}
    msg = net.load_state_dict(state_dict, strict=False)
    print("Load pretrained model with msg: {}".format(msg))
 

    
    ydatas = list()
    for params in params_list:
        test_trial = generate_trials(rule,hp,'psychometric',stim_mod, params = params)
        x,y,y_loc,c_mask = test_trial.x,test_trial.y,test_trial.y_loc,test_trial.c_mask
        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.float)    
        c_mask = torch.from_numpy(c_mask).type(torch.float)
        inputs = x
        y_hat,activity = net(inputs)
        
        y_hat = y_hat.detach().numpy()
        y_hat = popvec(y_hat)
        
        y_hat = np.reshape(y_hat[-1],batch_shape)
        
        
        if rule == 'dm':
            T2_loc = np.zeros(batch_shape)
            T1_loc = T2_loc + np.pi             
        if rule == 'coltargdm':
            T1_loc = np.reshape(params['stim2_locs'],batch_shape)
            T2_loc = np.reshape(params['stim3_locs'],batch_shape)
        # Average over the first dimension of each batch
        choose_T1 = (get_dist(y_hat-T1_loc)<THETA).sum(axis=0)
        choose_T2 = (get_dist(y_hat-T2_loc)<THETA).sum(axis=0)
        ydatas.append(choose_T1/(choose_T1+choose_T2))
    
    return ydatas
        
def psychometric_dm(model_dir,**kwargs):
    rule = 'dm'
    stim_mod = 1   # 1 is fine task 2 is coarse task
    if stim_mod == 1:
        stim_coh = np.ones(8)*1.8
        stim_loc = np.array([-12,-6,-3,0,3,6,12])*3/360*np.pi+np.pi
        # stim_loc = np.array([-6,-4.5,-4,-3,-2,-1.5,0])*3/360*np.pi+np.pi
        n_rep = 20
        unique_n_stim = len(stim_loc)
    elif stim_mod == 2:
        stim_coh = np.array([0.5, 0.15, 0.05, 0, 0.01, 0.05, 0.15, 0.5])*0.05
        stim_loc = np.array([0, 0, 0, 0, np.pi, np.pi, np.pi, np.pi])
        n_rep = 8
        unique_n_stim = (len(stim_coh)-1)*len(stim_loc)+1
    batch_size = n_rep*unique_n_stim
    batch_shape =  (n_rep,unique_n_stim)
    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)   
    
    stim_locs = stim_loc[ind_stim]
    stim_strengths = stim_coh[ind_stim]
    
    params_list = list()
    stim_times = [1000]
    
    for stim_time in stim_times:
        params = {'stim_locs': stim_locs,
                  'stim_strengths': stim_strengths,
                  'stim_time': stim_time}
        
        params_list.append(params)
    
    if stim_mod == 1:
        xdatas = [stim_loc]
    elif stim_mod == 2:
        xdatas = [stim_coh]
    ydatas = _psychometric_dm(model_dir, rule,stim_mod, params_list, batch_shape)
    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in stim_times],
                              colors=BLUES,
                              legtitle='Stim. time (ms)', rule=rule,
                              figname_append = kwargs['figname_append'])
    
def psychometric_color_dm(model_dir,**kwargs):
    rule = 'coltargdm'
    stim_mod = 1 # 1 is fine task 2 is coarse task
    if stim_mod == 1:
        stim1_coh = np.ones(7)*0.1
        stim1_loc = np.array([-12,-6,-3,0,3,6,12])*6/360*np.pi+np.pi
        # stim1_loc = np.array([-25,-24,-23,23,24,25])*6/360*np.pi+np.pi
        n_rep = 100
        unique_n_stim = len(stim1_loc)
    elif stim_mod ==2:
        stim1_coh = np.array([0.5,0.15,0.05,0,0.01,0.05,0.15,0.5])*0.05
        stim1_loc = np.array([0,0,0,0,np.pi,np.pi,np.pi,np.pi])
        n_rep = 8
        unique_n_stim = (len(stim1_coh)-1)*len(stim1_loc)+1
    batch_size = n_rep*unique_n_stim
    batch_shape = (n_rep,unique_n_stim)
    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)
    
    stim1_locs = stim1_loc[ind_stim]
    stim1_strengths = stim1_coh[ind_stim]
    seed = 3
    rng = np.random.RandomState(seed)
    stim2_locs = rng.choice([np.pi,2*np.pi],(batch_size,))
    stim3_locs = (stim2_locs+np.pi)%(2*np.pi)
    
    params_list = list()
    stim1_times = [1000]
    
    for stim1_time in stim1_times:
        params = {'stim1_locs': stim1_locs,
                  'stim1_strengths': stim1_strengths,
                  'stim_time': stim1_time,
                  'stim2_locs': stim2_locs,
                  'stim3_locs': stim3_locs}
        
        params_list.append(params)
        
    if stim_mod == 1:
        xdatas = [stim1_loc]
    elif stim_mod == 2:
        xdatas = [stim1_coh]
    
    ydatas = _psychometric_dm(model_dir, rule, stim_mod, params_list, batch_shape)
      
    plot_psychometric_choice(xdatas,ydatas,
                             labels=[str(t) for t in stim1_times],
                             colors=BLUES,
                             legtitle='Stim. time(ms)', rule=rule,**kwargs)
    
   
def plot_psychometric_choice(xdatas,ydatas,labels,colors,**kwargs):
    """
    Standard function for plotting the psychometric curves

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    """
    fs = 15
    fig,ax = plt.subplots(figsize=(5.8,3.3),constrained_layout=True)
    # ax = fig.add_axes([0.25,0.25,0.65,0.65])
    fits = list()
    for i in range(len(xdatas)):
        # Analyze performance of the choice tasks
        cdf_gaussian = lambda x, mu, sigma : stats.norm.cdf(x, mu, sigma)
        
        xdata = xdatas[i]-np.pi
        ydata = ydatas[i]
        ax.plot(xdata,ydata,'o',markersize=3.5,color = colors[i])
        
        try:
            xmin = np.min(xdata)
            xmax = np.max(xdata)
            x_plot = np.linspace(xmin,xmax,100)
            (mu,sigma),_ = curve_fit(cdf_gaussian,xdata,ydata,bounds=([-0.5,0.001],[0.5,10]))
            fits.append((mu,sigma))
            ax.plot(x_plot, cdf_gaussian(x_plot,mu,sigma), label = labels[i],
                    linewidth=1, color = colors[i])
        except:
            pass
    bias = mu
    # Compute threshold (difference from 50% to 75%)
    z75 = stats.norm.ppf(0.75)  # ≈ 0.674
    threshold = z75 * sigma
    plt.xlabel('heading',fontsize=fs)
    plt.ylim([-0.05,1.05])
    plt.xlim([xmin*1.1,xmax*1.1])
    plt.yticks([0,0.5,1])
    plt.xticks([xmin, 0, xmax])
    if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
        plt.yticks([0,0.5,1],['','',''])
    else:
        plt.ylabel('P(choice 1)',fontsize=fs)
    plt.title(rule_name[kwargs['rule']], fontsize=fs, y=0.95)
    plt.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=fs)
        
    if len(xdatas)>1:
        if len(kwargs['legtitle'])>10:
            loc = (0.0, 0.5)
        else:
            loc = (0.0, 0.5)
        leg = plt.legend(title=kwargs['legtitle'],fontsize=fs,frameon=False,
                         loc=loc,labelspacing=0.3)
        plt.setp(leg.get_title(),fontsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    text_x = xmax * 0.05  # 文字放在左侧
    text_y = 0.95              # 接近顶部
    if 'correct_rate' in kwargs:
        correct_rate =  kwargs['correct_rate'] 
        # 将文字放在坐标轴右上角之外一点
        ax.text(1.05, 1.0,  # x, y 坐标 >1 表示在外面
                f"Bias = {bias:.3f}\nThreshold = {threshold:.3f}\ncorrect rate = {correct_rate:.3f}",
                transform=ax.transAxes,  # 用相对坐标系 (0~1)
                fontsize=fs*0.9,
                ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                clip_on=False) 
    else:
        # 将文字放在坐标轴右上角之外一点
        ax.text(1.05, 1.0,  # x, y 坐标 >1 表示在外面
                f"Bias = {bias:.3f}\nThreshold = {threshold:.3f}",
                transform=ax.transAxes,  # 用相对坐标系 (0~1)
                fontsize=fs*0.9,
                ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                clip_on=False) 
    # SetFigure(15)        
    # plt.show()


    # if 'figname_append' in kwargs:
    #     figname += kwargs['figname_append']

    if save:
        figname = './batchfigure/'+rule_name[kwargs['rule']].replace(' ','') + '/'+kwargs['figname_append']+'/'
        os.makedirs(figname,exist_ok=True)
        plt.savefig(figname+'psycho_curve.png', transparent=True)

    return bias, threshold

# model_dir = './checkpoint/1hiddenHDsignrestrict.t7'   
# model_dir = './checkpoint/1hiddenHDtask.t7'          
# psychometric_dm(model_dir,figname_append = 'readoutsign_abs')  

# model_dir = './checkpoint/coltargdm1hidden.t7'           
# # psychometric_dm(model_dir,figname_append='all1hidden')   
# psychometric_color_dm(model_dir,figname_append='coltargdm1hidden')   

# model_dir = './checkpoint/dm2hiddennet2.t7'           
# psychometric_dm(model_dir,figname_append='dm2hiddenrnn2') 

# model_dir = './checkpoint/continue2hiddennet2.t7'           
# psychometric_dm(model_dir,figname_append = 'continue2hiddenrnn2')  
# psychometric_color_dm(model_dir,figname_append = 'continue2hiddenrnn1')  

# model_dir = './checkpoint/continuealltask1hiddenhebb.t7'

# psychometric_dm(model_dir,figname_append = 'continuealltask1hiddenhebb')
# psychometric_color_dm(model_dir,figname_append = 'continuealltask1hiddenhebb')

# model_dir = './checkpoint/coltargdm802hiddennet2keep.t7'      
# model_dir = './checkpoint_batch14/0000colorhdnet5.t7'           
# psychometric_dm(model_dir,figname_append = '2AFC802hiddennet2')  
# psychometric_color_dm(model_dir,figname_append = 'colorhdnet3')  


# # figname_suffix = f'checkgpu/{i}'
# model_dir = './checkpoint/checkgpu.t7'         

# # neuralactivity_color_dm(model_dir,figname_append=figname_suffix) 
# # psychometric_color_dm(model_dir,figname_append=figname_suffix)
# psychometric_color_dm(model_dir,figname_append='continue2AFC2hidden2') 











