# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 20:58:30 2022
for dynamic analysis
unfinished
@author: NZ
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import scipy.stats as stats
from scipy.optimize import curve_fit

from mytask import generate_trials, rule_name, get_dist

import mynetwork1hidden
from mynetwork1hidden import Net

from mytools import popvec,get_y_direction

from scipy.stats import pearsonr

plt.figure()
plt.close('all')


# get neural data from test trials
def _neuralactivity_dm(model_dir, rule, stim_mod, params_list, batch_shape):
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
    print('Starting neural activity analysis of the {:s} task...'.format(rule_name[rule]))
    
    modelparams = torch.load(model_dir)
    state_dict = modelparams["state_dict"]
    hp = modelparams["hp"]
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
    e_size = net.rnn.e_size
    return activity, test_trial, state_dict ,y_hat,net,hp

def neuralactivity_delaysac(model_dir,**kwargs):
    rule = 'delaysaccade'
    stim_mod = 1
    stim_loc = np.linspace(0,315,num=8)/180*np.pi
    n_rep = 10
    unique_n_stim = len(stim_loc)
    batch_size = n_rep*unique_n_stim
    batch_shape = (n_rep,unique_n_stim)
    ind_stim_loc,ind_stim = np.unravel_index(range(batch_size),batch_shape)
    
    stim_locs = stim_loc[ind_stim]
    
    params_list = list()
    stim_times = [300]
    
    for stim_time in stim_times:
        params = {'stim_locs':stim_locs,
                  'stim_time':stim_time}
        
        params_list.append(params)
        
    neural_activity,test_trial,state_dict,y_hat,net,hp = _neuralactivity_dm(
        model_dir, rule,stim_mod, params_list, batch_shape)
    neural_activity = neural_activity.detach().numpy()
    #get test_trail related information
    inputs = test_trial.x

    #pack time related information
    stim_ons = test_trial.ons
    dt = test_trial.dt
    times_relate = {'stim_ons':stim_ons,'dt':dt,'stim_dur':stim_times}
    
    pro_neural_activity = np.mean(neural_activity,0)
    pca = plot_trajectory(pro_neural_activity,neural_activity,times_relate,ind_stim)
    # give the constant inputs in a  time period to calculate fixed points
    #input_shape = [batch_size,n_neurons]   
    fixation_timepoint = int(np.unique(stim_ons-100/dt)) #pick the time point at 100ms before stim_ons
    input_fixation = inputs[fixation_timepoint,:,:]
    output_fixation = neural_activity[fixation_timepoint-1,:,:]
    # fixedhidden_fixation = get_fixed_points(batch_size,hp,input_fixation,output_fixation,net)
    stimulus_timepoint = int(np.unique(stim_ons+100/dt))
    input_stimulus = inputs[stimulus_timepoint,:,:]
    output_stimulus = neural_activity[stimulus_timepoint-1,:,:]
    fixedhidden_stimulus = get_fixed_points(batch_size,hp,input_stimulus,output_stimulus,net)
    visualize_fixed_points(fixedhidden_stimulus,neural_activity,pca,times_relate,ind_stim)
    
    jac,selected_fps = get_Jacobian(fixedhidden_stimulus,hp,input_stimulus,output_stimulus,net,ind_stim)
    visualize_Jacobian(neural_activity,times_relate,pca,jac,selected_fps,ind_stim)
    
def plot_trajectory(pro_neural_activity,neural_activity,times_relate,ind_stim):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(pro_neural_activity)
    pc = pca.components_
    
    dt = times_relate['dt']
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    return pca    
    # neural_activity = neural_activity-np.mean(neural_activity,0)
    
    activity_pc = np.dot(neural_activity,pc.T)
    plt.figure()
    for itrial in range(activity_pc.shape[1]):
        color = np.multiply((ind_stim[itrial]+1),[(1,0,1)])/(np.max(ind_stim)+1)
        plt.plot(activity_pc[:,itrial,0],activity_pc[:,itrial,1],color =color[0,:])
    plt.scatter(activity_pc[0,:,0],activity_pc[0,:,1],marker='o')
    plt.scatter(activity_pc[time_end_ind,:,0],activity_pc[time_end_ind,:,1],marker='^')
    plt.show()
    neural_point = np.dot(pro_neural_activity,pc.T) 
    # check could the trials form a same condition clustered together
    plt.figure()
    plt.scatter(neural_point[:,0],neural_point[:,1])    
    plt.show()
    
def get_fixed_points(batch_size,hp,inputs,outputs,net):
    
    # Here hidden activity is the variable to be optimized
    # Initialized randomly for search in parallel (activity all positive)
    inputs = torch.from_numpy(inputs).type(torch.float)
    hidden_size = hp['hidden_size']
    output = torch.tensor(np.random.rand(batch_size, hidden_size),
                            requires_grad=True, dtype=torch.float32)
    # hidden = torch.tensor(np.random.rand(batch_size, hidden_size))
    # hidden = torch.from_numpy(outputs).type(torch.float)
    # output = torch.from_numpy(outputs).type(torch.float)
    hidden = output
    # output.ruquires_grad = True
    output.requires_grad = True

    # Use Adam optimizer          
    optimizer = optim.Adam([output], lr=0.001)
    criterion = nn.MSELoss()
    
    running_loss = 0
    for i in range(10000):
        optimizer.zero_grad()   # zero the gradient buffers        
        # Take the one-step recurrent function from the trained network    
        hiddens = (hidden,output)
        new_h,n_output = net.rnn.recurrence(inputs, hiddens)
        loss = criterion(n_output,output)
        # loss = criterion(new_output,output)
        loss.backward()
        optimizer.step() #Does the update
        
        running_loss += loss.item()
        if i%200 == 199:
            #running_loss /=1000
            print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
            running_loss = 0   
    return output
            
def visualize_fixed_points(hidden,neural_activity,pca,times_relate,ind_stim):
    fixedpoints = hidden.detach().numpy()
    print(fixedpoints.shape)
    pc = pca.components_
    # get time related infor
    dt = times_relate['dt']
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    
    # plot in the same space as activity
    plt.figure()
    activity_pc = np.dot(neural_activity,pc.T)
    fixedpoints_pc = np.dot(fixedpoints,pc.T)
    for itrial in range(activity_pc.shape[1]):
        color = np.multiply((ind_stim[itrial]+1),[(1,0,1)])/(np.max(ind_stim)+1)
        plt.plot(activity_pc[:time_end_ind,itrial,0],activity_pc[:time_end_ind,itrial,1],color = color[0,:])
        #plt.plot(activity_pc[:,itrial,0],activity_pc[:,itrial,1],color = color[0,:])
        #Fixed points are shown in cross
        plt.plot(fixedpoints_pc[itrial, 0], fixedpoints_pc[itrial, 1], 'x',color = color[0,:])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    
def get_Jacobian(fixedhidden,hp,inputs,outputs,net,ind_stim):
    # index of fixed point to focus on    
    # choose the trial from different condition
    all_jac = []
    check_select_trialind = []
    selected_fps = []
    check_deltah = []
    conditions = np.unique(ind_stim)
    # for icondition in range(len(conditions)):
    for itrial in range(len(ind_stim)):
        # select_trialind = np.where(ind_stim==conditions[icondition])[0][0]
        select_trialind = itrial
        check_select_trialind.append(select_trialind)
        fixedpoints = fixedhidden.detach().numpy()
        fp = torch.from_numpy(fixedpoints[select_trialind])
        selected_fps.append(fp)
        fp.requires_grad = True
        i_input = inputs[select_trialind,:]
        i_input = torch.from_numpy(i_input)
        hidden_size = hp['hidden_size']
        # i_outputs = outputs[select_trialind,:]
        # i_outputs  = torch.from_numpy(i_outputs)
        i_outputs = fp
        i_outputs.requires_grad = True
        hiddens = (fp,i_outputs)
        new_hidden,new_outputs = net.rnn.recurrence(i_input,hiddens)
        # deltah = new_hidden-fp
        deltaO = new_outputs-i_outputs
        # check_deltah.append(deltah.detach().numpy())
        # check_deltah = deltah.detach().numpy()
        jacT = torch.zeros(hidden_size,hidden_size)
        for i in range(hidden_size):
            output = torch.zeros(hidden_size)
            output[i] = 1.
            jacT[:,i] = torch.autograd.grad(deltaO,fp,grad_outputs=output,retain_graph=True)[0]
        
            jac = jacT.detach().numpy().T
        all_jac.append(jac)
    return all_jac,selected_fps
    
def visualize_Jacobian(neural_activity,times_relate,pca,all_jac,selected_fps,ind_stim):
    # get time related infor
    dt = times_relate['dt']
    stim_ons = times_relate['stim_ons']
    stim_on = np.unique(stim_ons)*dt
    stim_dur = times_relate['stim_dur']
    stim_end = stim_on + stim_dur[0]
    time_steps = np.arange(neural_activity.shape[0])*dt
    time_begin_ind = np.where(time_steps>=stim_on)[0][0]
    time_end_ind = np.where(time_steps<=stim_end)[0][-1]
    
    pc = pca.components_ # axis for visualization projection
    all_fp = []
    all_end_pts = []
    all_eigval = []
    for ijac in range(len(all_jac)):
        jac = all_jac[ijac]
        eigval,eigvec = np.linalg.eig(jac)
        all_eigval.append(np.real(np.max(eigval)))
        vec = np.real(eigvec[:,np.argmax(eigval)])
        end_pts = np.array([+vec,-vec])*2
        """fp + end_pts after projection"""
        # end_pts = np.dot(end_pts,pc.T)
        # fp = selected_fps[ijac].detach().numpy()
        # fp  = np.dot(fp,pc.T)
        # end_pts = fp + end_pts
        
        """fp + end_pts before projection"""
        fp = selected_fps[ijac].detach().numpy()
        end_pts = fp + end_pts
        end_pts = np.dot(end_pts,pc.T)
        fp  = np.dot(fp,pc.T)
        
        
        
        all_end_pts.append(end_pts)
        all_fp.append(fp)
    activity_pc = np.dot(neural_activity,pc.T)
    # plt in the same space as activity
    all_fp = np.asarray(all_fp)
    all_end_pts = np.asarray(all_end_pts)
    plt.figure()
    conditions = np.unique(ind_stim)
    for icondition in range(len(conditions)):
        color = np.multiply((icondition+1),[(1,0,1)])/(len(conditions)+1)
        select_trials = ind_stim==conditions[icondition]
        mean_fp = np.mean(all_fp[select_trials,:],0)
        mean_end_pts = np.mean(all_end_pts[select_trials,:,:],0)
        # plt.plot(activity_pc[:time_end_ind,select_trials,0],activity_pc[:time_end_ind,select_trials,1],color = color[0,:])
        # plt.plot(activity_pc[:,itrial,0],activity_pc[:,itrial,1],color = color[0,:])
        plt.plot(mean_fp[0],mean_fp[1], 'x',color = color[0,:])
        plt.plot(mean_end_pts[:,0],mean_end_pts[:,1],color = color[0,:])
    plt.show()
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
model_dir = './checkpoint/delaysacsignrestrict.t7'           
neuralactivity_delaysac(model_dir,figname_append='delaysacsignrestrict')     