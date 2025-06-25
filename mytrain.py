# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:47:53 2021
here I trained all the task simultaneously
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


# import  mytask
import mytask_new as mytask

# import mynetwork
# from mynetwork import Net

# import mynetwork1hidden
# from mynetwork1hidden import Net

# import mynetwork_new2
# from mynetwork_new2 import Net

import mynetwork6
from mynetwork6 import Net

# import mynetwork_new
# from mynetwork_new import Net

# import mynetworkhebb
# from mynetworkhebb import Net

# import mynetworkforward
# from mynetworkforward import Net

import mytools
from mytools import get_perf
# import mynetworklstm1hidden
# from mynetworklstm1hidden import Net

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
    # num_ring = 1
    num_ring = 2

    n_rule = 3
    n_eachring = 36
    n_outputdir = 32
    # n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_outputdir+1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 1024,        
            'ruleset': ruleset,
            'rule_start': 1+num_ring*n_eachring,
            # Stopping performance
            'target_perf': 0.98,
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            'alpha': 0.2,
            # input noise
            'sigma_x': 0.01,
            # number of units each ring
            'n_rule': n_rule,
            'num_ring': num_ring,
            'n_eachring': n_eachring,
            'loss_type': 'lsq',
            'L1_h':1,
            'L2_h':1,
            'L1_w':1,
            'L2_w':0,
            'dt': 20,
            'rng': np.random.RandomState(seed),
            'easy_task': 1,
            # 'hidden_size1': 64,
            # 'hidden_size2': 64}
            # 'hidden_size3': 32}
            'hidden_size1': 64}
    return hp

def get_loss(hp, net, trial):
    x,y,y_loc,c_mask = trial.x,trial.y,trial.y_loc,trial.c_mask
    x = torch.from_numpy(x).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)
    y_loc_tensor = torch.from_numpy(y_loc).type(torch.float)
    c_mask = torch.from_numpy(c_mask).type(torch.float)
    inputs = x
    
    y_hat,activity = net(inputs)
    n_output = hp['n_output']
    y_shaped = torch.reshape(y, (-1, n_output))
    y_hat_shaped = torch.reshape(y_hat,(-1,n_output))
    mask_shaped = torch.reshape(c_mask,(-1,n_output))

    loss = torch.mean(torch.sum(torch.square(y_shaped-y_hat_shaped)*mask_shaped,1))

    perf = np.mean(get_perf(y_hat,y_loc))
    
    return loss, perf

def do_eval(hp,net,log,rule_train):
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

    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)

        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = mytask.generate_trials(
                rule_test, hp, 'random',stim_mod=1, batch_size=batch_size_test_rep)
            loss_test,perf_test = get_loss(hp,net, trial) 
            perf_tmp.append(perf_test)
            
        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()
        
    if hasattr(rule_train,'__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)
    
    checkpoint(net, log, hp, netname )        
    
    return log
    
def train(netname,
          device='cpu',
          hp=None,
          max_steps=1e7,
          display_epoch=20,
          rule_trains=None,
          rule_prob_map=None,
          ruleset = 'all',
          ):
    
    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    
    net = Net(hp,dt = hp['dt'])    
    if hp['L2_w']==1:
        # optimizer = optim.AdamW(net.parameters(), lr=0.01,weight_decay=0.001)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = mytask.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']#
    
    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()    
        
    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))
    
    #store result
    log = defaultdict(list)
    log['netname'] = netname
    
    #Rcord time
    t_start = time.time()
    
    # penalty on deviation from initial weight(wait to be complete)    
    epoch = 0
    running_loss = 0 
    perf = 0
    while epoch*hp['batch_size_train']<=max_steps:           
        try:
            # validation
            if epoch%display_epoch==1:
                log['trials'].append(epoch * hp['batch_size_train'])
                log['times'].append(time.time()-t_start)
                log['epoch'].append(epoch)
                log['running_loss'].append(running_loss)
                log['perf'].append(perf)
                log = do_eval(hp,net,log,hp['rule_trains'])
                if log['perf_min'][-1] >= hp['target_perf']:
                    print('Perf reached the target: {:0.2f}'.format(
                        hp['target_perf']))
                    break
        
            #Training
            rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                          p=hp['rule_probs'])
    
            # Generate a random batch of trials.
            # Each batch has the same trial length
            trial = mytask.generate_trials(
                    rule_train_now, hp, mode='random',stim_mod=1,noise_on=True,
                    batch_size=hp['batch_size_train'])

            x,y,y_loc,c_mask = trial.x,trial.y,trial.y_loc,trial.c_mask
            x = torch.from_numpy(x).type(torch.float)
            y = torch.from_numpy(y).type(torch.float)
            y_loc_tensor = torch.from_numpy(y_loc).type(torch.float)
            c_mask = torch.from_numpy(c_mask).type(torch.float)
            inputs = x
            optimizer.zero_grad()   # zero the gradient buffers 
            y_hat,activity = net(inputs)
            n_output = hp['n_output']
            y_shaped = torch.reshape(y, (-1, n_output))
            y_hat_shaped = torch.reshape(y_hat,(-1,n_output))
            mask_shaped = torch.reshape(c_mask,(-1,n_output))
    
            loss = torch.mean(torch.sum(torch.square(y_shaped-y_hat_shaped)*mask_shaped,1))
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
                

            perf = np.mean(get_perf(y_hat,y_loc))
            # loss.requires_grad_(True)
            
            loss.backward()
            optimizer.step()
    
 
            running_loss = loss.item()
            epoch += 1
    
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break

    print("Optimization finished!")


    
def checkpoint(model, log, hp, outModelName):
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
    torch.save(state, f'./checkpoint/{outModelName}.t7')
    

# netname = '2hiddenHD'
# ruleset = 'delaysac'
# hp = get_default_hp(ruleset)
# netname = 'allsignrestrict'
# netname = 'delaysac2hiddentanh'
# netname = 'hd2hiddenlayer'
# netname = 'colorhd2hiddenlayer'
# netname = 'delaysac2hidden'
# netname = 'all2hiddennet2'
# netname = 'coltargdmpeceptron'
# train(netname,ruleset = 'coltargdm')
netname = 'colorhd1hiddencheck'
train(netname, ruleset = 'coltargdm')

  