# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:00:48 2022
here I train all the tasks sequentially and with continue learning
@author: NaN
"""
import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import  mytask

# import mynetworkhebb
# from mynetworkhebb import Net

# import mynetwork_new8
# from mynetwork_new8 import Net

# import mynetwork_new3
# from mynetwork_new3 import Net

import mynetwork8
from  mynetwork8 import Net

# import mynetwork1hidden
# from mynetwork1hidden import Net

import mytools
from mytools import get_perf

def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    seed = 3
    # num_ring = database1.get_num_ring(ruleset)
    # n_rule = database1.get_num_rule(ruleset)
    # num_ring = 3
    num_ring = 3

    n_rule = 3
    n_eachring = 32
    n_outputdir = 16
    # n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_outputdir+1
    n_input_heading = 1+n_eachring
    n_input_targcolor = (num_ring-1)*n_eachring
    n_input_rules = n_rule
    # n_output = n_outputdir+1
    hp = {
            # batch size for training
            'batch_size_train': 256,
            # batch_size for testing
            'batch_size_test': 64,        
            'ruleset': ruleset,
            # 'rule_start': 1+num_ring*n_eachring,
            'rule_start': 1+num_ring*n_eachring,
            # Stopping performance
            'target_perf': {'dm':0.95,'coltargdm':0.95,'delaysaccade':0.90},
            'n_input': n_input,
            'n_input_heading': n_input_heading,
            'n_input_targcolor':n_input_targcolor,
            'n_input_rules':n_rule,
            # number of output units
            'n_output': n_output,
            'alpha': 0.1,
            # input noise
            'sigma_x': 0.1,
            'sigma_rec1':0.1,
             'sigma_rec2':0.1,
            'sigma_feedforward':0.1,
            'sigma_feedback':0.1,
            # recurrent connectivity
            'recur1':1,
            'recur2':1,
            'fforwardstren':1,
            'fbackstren':1,                
            # 'sigma_x': 0.0001,
            # number of units each ring
            'n_rule': n_rule,
            'num_ring': num_ring,
            'n_eachring': n_eachring,
            'loss_type': 'lsq',
            'L1_h':0,
            'L2_h':0,
            'L1_w':0,
            'L2_w':0,
            'L1_tau':20,
            'L2_tau':200,
            'dt': 20,
            'seed':seed,
            'rng': np.random.RandomState(seed),
            'easy_task': 1,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0.1,
            'ksi_intsyn': 0.01,            
            'hidden_size1': 256,
            'hidden_size2': 128,
            'hidden_size3':64}
            # 'hidden_size3': 32}
            # 'hidden_size1': 64}
    return hp

def get_loss(device,hp, net, trial):
    x,y,y_loc,c_mask = trial.x,trial.y,trial.y_loc,trial.c_mask
    # x = torch.from_numpy(x).type(torch.float).to(device)
    # y = torch.from_numpy(y).type(torch.float).to(device)
    # y_loc_tensor = torch.from_numpy(y_loc).type(torch.float).to(device)
    # c_mask = torch.from_numpy(c_mask).type(torch.float).to(device)
    inputs = x
    
    y_hat,activity = net(inputs)
    n_output = hp['n_output']
    y_shaped = torch.reshape(y, (-1, n_output))
    y_hat_shaped = torch.reshape(y_hat,(-1,n_output))
    mask_shaped = torch.reshape(c_mask,(-1,n_output))

    loss = torch.mean(torch.sum(torch.square(y_shaped-y_hat_shaped)*mask_shaped,1))

    perf = torch.mean(get_perf(y_hat,y_loc))
    
    return loss, perf

def do_eval(device,hp,net,log,rule_train,netname,savepath):
    """Do evaluation.

    Args:
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """  
    
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)
        
    print('Epoch {:7d}'.format(log['epoch'][-1]) +
          '| Time {:0.2f} s'.format(log['times'][-1]) +
          '| Now training '+rule_name_print +   
          '| Loss {:0.4f}'.format(log['running_loss'][-1]) +
          '| perf {:0.4f}'.format(log['perf'][-1]))
   

    # n_rep = 16
    # batch_size_test_rep = int(hp['batch_size_test']/n_rep)
    batch_size_test = hp['batch_size_test']

    perf_tmp = []                             
    # for i_rep in range(n_rep):
    for i_rule_test in rule_train:
        trial = mytask.generate_trials(
                i_rule_test, hp,device, 'random',stim_mod=1, batch_size=batch_size_test)
        loss_test,perf_test = get_loss(device,hp,net, trial) 
        perf_tmp.append(perf_test)
        
        # Stack the list of tensors and compute mean
        perf_stack = torch.stack(perf_tmp)
        mean_perf = torch.mean(perf_stack)
        
        log['perf_'+i_rule_test].append(mean_perf.item())
        print('{:15s}'.format(i_rule_test) +
             '| perf {:0.2f}'.format(mean_perf.item()))
        sys.stdout.flush()
            
        # log['perf_'+i_rule_test].append(torch.mean(perf_tmp, dtype=np.float64))
        # print('{:15s}'.format(i_rule_test) +
        #      '| perf {:0.2f}'.format(torch.mean(perf_tmp)))
        # sys.stdout.flush()
        
        
    # if hasattr(rule_train,'__iter__'):
    #     rule_tmp = rule_train
    # else:
    #     rule_tmp = [rule_train]        

    # perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    # log['perf_avg'].append(perf_tests_mean)

    # perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    # log['perf_min'].append(perf_tests_min)
    
    checkpoint(net, log, hp, netname, savepath )        
    
    return log

def train(netname,          
          savepath,
          device,
          hp=None,
          max_steps=1e7,
          display_epoch=20,
          rule_trains=None,
          rule_prob_map=None,
          ruleset='2AFC',
          basedon = None,
          ): 
    '''Train the network sequentially.

    Args:
        model_dir: str, training directory
        rule_trains: a list of list of tasks to train sequentially
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps for each list of tasks
        display_step: int, display steps
        ruleset: the set of rules to train
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    '''    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # device = torch.device("cpu")
    print(device)
    
    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)

    if not basedon is None:
        # trained on allready trained model
        modelparams = torch.load(basedon)
        state_dict = modelparams["state_dict"]
        hp = default_hp
        net = Net(hp,device,dt = hp['dt'])
        state_dict = {k.replace("module.",""): v for k, v in state_dict.items()}
        msg = net.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
    else:
        hp = default_hp        
        net = Net(hp,device,dt = hp['dt'])
    
    net.to(device)
     
    if hp['L2_w']==1:
        # optimizer = optim.AdamW(net.parameters(), lr=0.01,weight_decay=0.001)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = mytask.rules_dict[ruleset]
        rule_trains = mytask.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
        
    # Get all rules by flattening the list of lists
    hp['rules'] = [rs for rs in rule_trains]  
    
    # Number of training iterations for each rule
    rule_train_iters = [len(r)*max_steps for r in rule_trains]    
    
    # Using continual learning or not
    c, ksi = hp['c_intsyn'], hp['ksi_intsyn']
    
    #store result
    log = defaultdict(list)
    log['netname'] = netname
    
    #Rcord time
    t_start = time.time()   
   
    # Looping
    step_total = 0
    iteration_num = 0 #count the iteration num
    for i_rule_train,rule_train_now in enumerate(hp['rule_trains']):
        epoch = 0
        
        # At the beginning of new tasks
        # Only if using intelligent synapses
        # get the weight for the last task
        if hasattr(net,'hebb'):
            weight_input2hidden = net.hebb.w1.to(device)
            weight_h2O = net.hebb.w2.to(device)
            w_current = [weight_input2hidden,weight_h2O]
            net_weight = [net.hebb.w1.detach(),net.hebb.w2.detach()]
        else:
            weight_input2hidden = net.rnn.input2h.weight.data  # input2hidden weight
            weight_h2heffective = net.rnn.h2h.effective_weight().data # h2h weight
            # weight_h2heffective = net.rnn.h2h.weight().data.to(device) # h2h weight
            weight_h2O = net.fc.weight.data#h2Output weight                    
            w_current = [weight_input2hidden,weight_h2heffective,weight_h2O]         
            net_weight = [net.rnn.input2h.weight.detach(),net.rnn.h2h.effective_weight().detach(),net.fc.weight.detach()]
            # w_current = [weight_h2heffective]         
            # net_weight = [net.rnn.h2h.effective_weight().detach()]            
            
            # weight_input2hidden = net.rnn.input2h.effective_weight().data.to(device)  # input2hidden weight
            # weight_h2heffective = net.rnn.h2h.effective_weight().data.to(device) # h2h weight
            # weight_h2O = net.fc.effective_weight().data.to(device) #h2Output weight                    
            # w_current = [weight_input2hidden,weight_h2heffective,weight_h2O]         
            # net_weight = [net.rnn.input2h.effective_weight().detach(),net.rnn.h2h.effective_weight().detach(),net.fc.effective_weight().detach()]
                
        if i_rule_train == 0:
            w_anc0 = w_current
            Omega0 = [torch.zeros(w.shape) for w in w_anc0]#ttransform weight importance to parameter
            omega0 = [torch.zeros(w.shape) for w in w_anc0] # evaluate important weight
            w_delta = [torch.zeros(w.shape) for w in w_anc0]
            penalty = 0
        elif c > 0: 
            w_anc0_prev = w_anc0
            w_anc0 = w_current
            w_delta = [w-w_prev for w,w_prev in zip(w_anc0, w_anc0_prev)]
            
            # Make sure all elements in omega0 are non-negative
            # normalizing o and sum o for all task
            # Penalty
            
            Omega0 = [torch.relu(O+o/(w_d**2 + ksi))
                      for O, o, w_d in zip(Omega0, omega0, w_delta)]
            
            #Update cost
            # loss = torch.tensor([0],dtype=torch.float32)
            C = torch.tensor(c,dtype=torch. float32).to(device)
            for w,O,w_cur in zip(net_weight, Omega0, w_current):
                # penalty = torch.mean(w-w_cur)
                  penalty = C*torch.sum(O*(torch.square(w-w_cur)))
               
        # Reset
        omega0 = [torch.zeros(w.shape) for w in w_anc0]
        task_trainagain = train_para(device,hp,log,net,rule_train_now,netname,savepath,
                                     rule_trains,optimizer,
                                     max_steps,i_rule_train,t_start,
                                     penalty,omega0,display_epoch=20)
         
            #when finished the first iteration check whether other allready 
            #trained tasks performance under target CR. if so, start a new 
            #iteration
        while task_trainagain:
            #reset i_rule_train for another iteration        
            # i_rule_train = 0
            iteration_num += 1 # for another iteration
            for i_rule_trainagain in task_trainagain:
                rule_train_now = i_rule_trainagain
                if i_rule_trainagain == task_trainagain[-1]: # in the last task get the tasks need to train again
                    task_trainagain = train_para(device,hp,log,net,rule_train_now,netname,savepath,
                                             rule_trains,optimizer,
                                             max_steps,i_rule_trainagain,t_start,
                                             penalty,omega0,display_epoch=20)
                else:
                    train_para(device,hp,log,net,rule_train_now,netname,savepath,
                                             rule_trains,optimizer,
                                             max_steps,i_rule_trainagain,t_start,
                                             penalty,omega0,display_epoch=20)
  
                            
        print("Optimization finished!")            
            
def checkpoint(model, log, hp, outModelName, savepath):
    print('saving...')
    state = {
        'state_dict': model.state_dict(),
        # 'acc': acc,
        # 'epoch': epoch,
        'log':log,
        'rng_state':torch.get_rng_state(),
        'hp':hp
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    # torch.save(state, f'./checkpoint_batch13/{outModelName}.t7')
    torch.save(state, f'{savepath}{outModelName}.t7')

def train_para(device,hp,log,net,rule_train_now,netname,savepath,
               rule_trains,optimizer,
               max_steps,i_rule_train,t_start,
               penalty,omega0,display_epoch=20):
    epoch = 0
    running_loss = 0 
    perf = 0
    task_trainagain = []
    # Keep training until reach max iterations
    # max_iter = rule_train_iters[i_rule_train]
    while (epoch*hp['batch_size_train']<=max_steps): 
        try:
            # Validation
            if epoch%display_epoch == 1:
                log['trials'].append(epoch * hp['batch_size_train'])
                log['times'].append(time.time()-t_start)
                log['epoch'].append(epoch)
                log['running_loss'].append(running_loss)
                log['perf'].append(perf)
                log = do_eval(device,hp,net,log,[rule_train_now],netname,savepath)
                if log['perf_'+rule_train_now][-1] >= hp['target_perf'][rule_train_now]:                    
                    print('Perf reached the target: {:0.2f}'.format(
                        hp['target_perf'][rule_train_now]))
                                                            
                    # if rule_train_now == rule_trains[-1]: #check if it's the last rule trained
                        # get the task need train in next iteration
                    rule_train_before = list(set(rule_trains)-set([rule_train_now]))
                        
                    check_log = do_eval(device,hp,net,log,rule_train_before,netname,savepath)
                    for i_rule in rule_train_before:
                        trained_task_name = i_rule
                        rule_train_before_perf = check_log['perf_'+trained_task_name]
                        if rule_train_before_perf[-1] < hp['target_perf'][trained_task_name]:
                            task_trainagain.append(trained_task_name)
                                        
                    break
            # Training 
            # Generate a random batch of trials.
            # Each batch has the same trial length
            trial = mytask.generate_trials(
                    rule_train_now, hp,device, 'random',stim_mod=1,noise_on=True,
                    batch_size=hp['batch_size_train'])
            
            x,y,y_loc,c_mask = trial.x,trial.y,trial.y_loc,trial.c_mask
            # x = torch.from_numpy(x).type(torch.float).to(device)
            # y = torch.from_numpy(y).type(torch.float).to(device)
            # y_loc_tensor = torch.from_numpy(y_loc).type(torch.float).to(device) 
            # c_mask = torch.from_numpy(c_mask).type(torch.float).to(device)
            inputs = x
            optimizer.zero_grad()   # zero the gradient buffers
            y_hat,activity = net(inputs)
            n_output = hp['n_output']
            y_shaped = torch.reshape(y, (-1, n_output))
            y_hat_shaped = torch.reshape(y_hat,(-1,n_output))
            mask_shaped = torch.reshape(c_mask,(-1,n_output))
            if i_rule_train == 0:
                loss = torch.mean(torch.sum(torch.square(y_shaped-y_hat_shaped)*mask_shaped,1))
            else:
                loss = torch.mean(torch.sum(torch.square(y_shaped-y_hat_shaped)*mask_shaped,1))+penalty 
            L1_h_norm = torch.mean(torch.abs(y_hat_shaped))
            L2_h_norm = torch.mean(torch.pow(y_hat_shaped,2))
            for param in net.parameters():
                L1_w_norm = torch.mean(abs(param))
                L2_w_norm = torch.mean(torch.pow(param,2))
                
            if hp['L1_h']>0:
                loss+=L1_h_norm
            if hp['L2_h']>0:
                loss+=loss+L2_h_norm
            if hp['L1_w']>0:
                loss+=loss+L1_w_norm
            if hp['L2_w']>0:
                loss+=loss+L2_w_norm            
            
            perf = torch.mean(get_perf(y_hat,y_loc))
            
            # Continual learning with intelligent synapses
            if hasattr(net,'hebb'):
                weight_input2hidden = net.hebb.w1
                weight_h2O = net.hebb.w2
                w_prev = [weight_input2hidden,weight_h2O]         
            else:
                weight_input2hidden = net.rnn.input2h.weight.data  # input2hidden weight
                weight_h2heffective = net.rnn.h2h.effective_weight().data # h2h weight
                weight_h2O = net.fc.weight.data #h2Output weight 
                w_prev = [weight_input2hidden,weight_h2heffective,weight_h2O]
                # w_prev = [weight_h2heffective]
            
            # This will compute the gradient BEFORE train step
            # retain_graph=True
            # net.rnn.input2h.weight.requires_grad_(True)
            # weight_input2hidden.retain_grad()
            
            
            
            loss.backward()
            
            # # 冻结不需要训练的参数
            # for param in net.parameters():
            #     param.requires_grad = False  # 先冻结所有参数
            
            # # 只允许 net.rnn.h2h 的参数训练
            # for param in net.rnn.h2h.parameters():
            #     param.requires_grad = True
            
            # # 计算梯度
            # retain_graph = True  # 如果需要多次反向传播，设置为 True
            # loss.backward(retain_graph=retain_graph)           
                
            
            

            
            w_grad = []
            if hasattr(net,'hebb'):
                w_grad.append(net.hebb.w1.grad)
                w_grad.append(net.hebb.w2.grad)
            else:
                w_grad.append(net.rnn.input2h.weight.grad)
                w_grad.append(net.rnn.h2h.weight.grad)
                w_grad.append(net.fc.weight.grad)
            # Get the weight after train step
            optimizer.step()
            w_now = []
            if hasattr(net,'hebb'):
                w_now.append(net.hebb.w1.data)
                w_now.append(net.hebb.w2.data)
            else:
                w_now.append(net.rnn.input2h.weight.data)
                w_now.append(net.rnn.h2h.effective_weight().data)
                w_now.append(net.fc.weight.data)

            # Update synaptic importance
            
            #print for debug
            
            omega0 = [
                o.to(device)-(w_n.to(device)-w_p.to(device))*w_g.to(device) for o,w_n,w_p,w_g in
                zip(omega0,w_now,w_prev,w_grad)
                ]
            # omega0 = [
            #     o-(w_n-w_p)*w_g for o,w_n,w_p,w_g in
            #     zip(omega0,w_now,w_prev,w_grad)
            #     ]
            running_loss = loss.item()
            epoch += 1
        
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break
        
    return task_trainagain
    
# # netname = 'color2h1'
# netname = 'checkrelu'
# savepath = './checkpoint/'
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# train(netname, savepath, device, ruleset = 'coltargdm')

# train(netname, savepath, ruleset = '2AFC')
# basednet = './checkpoint/coltargdm802hiddennet2keep.t7'
# netname = '2AFC802hiddennet2'
# netname = 'continue2hiddennet2'
# train(netname,ruleset = '2AFC',basedon = basednet)
# train(netname,ruleset = '2AFC')

