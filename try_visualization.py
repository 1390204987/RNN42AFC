# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:24:39 2021

@author: NaN
"""

# for network visualization

env.reset(no_step=True)
env.timing.update({'fixation': ('constant',500),
                   'stimulus': ('constant',500)})
perf = 0 
num_trial = 500
activity_dict = {}
trial_infos = {}
stim_activity = [[],[]]
condition = [[],[],[],[]]    
checkcorrect = []
checkgt = []
checkgroundtruth = []
for i in range(num_trial):
    env.new_trial()
    ob, gt = env.ob, env.gt
    inputs = torch.from_numpy(ob[:,np.newaxis,:]).type(torch.float)
    action_pred, rnn_activity = net(inputs)
    
    # Compute performance
    action_pred = action_pred.detach().numpy()
    choice = np.argmax(action_pred[-1,0,:])
    correct = choice == gt[-1]
    
    #Log trial info
    trial_info = env.trial
    trial_info.update({'correct': correct, 'choice': choice})
    trial_infos[i] = trial_info
    
    #Log stimulus period activity
    rnn_activity = rnn_activity[:,0,:].detach().numpy()
    activity_dict[i] = rnn_activity
    
    #Compute stimulus selectivity for all units
    #Compute each neuron's response in trials where ground_truth=0 and 1 respectively
    rnn_activity = rnn_activity[env.start_ind['stimulus']:env.end_ind['stimulus']]
    stim_activity[env.trial['ground_truth']].append(rnn_activity)
    checkcorrect.append(correct)
    checkgt.append(gt[-1])
    checkgroundtruth.append(env.trial['ground_truth'])
    if env.trial['ground_truth']== 0:
        if correct==True:
            condition[0].append(rnn_activity)
        else:
            condition[1].append(rnn_activity)
    else:
        if correct==True:
            condition[2].append(rnn_activity)
        else:
            condition[3].append(rnn_activity)

print('average performance', np.mean([val['correct'] for val in trial_infos.values()]))
                      
            
import matplotlib.pyplot as plt

e_size = net.rnn.e_size
trial = 2

plt.figure(1)
_ = plt.plot(activity_dict[trial][:,:e_size],color='blue', label='Excitatory')
_ = plt.plot(activity_dict[trial][:,e_size:],color='red', label='Inhibitory')
plt.xlabel('Time step')
plt.ylabel('Activity')            

neuron = 6
plt.figure(2)
_ = [plt.plot(stim_activity[0][trial][:,neuron], color='blue', label='0') for trial in range(len(stim_activity[0]))]
_ = [plt.plot(stim_activity[1][trial][:,neuron], color='red', label='1') for trial in range(len(stim_activity[1]))]
plt.xlabel('Time step')
plt.ylabel('Activity')          

mean_activity = []        
std_activity = []
for ground_truth in [0,1]:
    activity = np.concatenate(stim_activity[ground_truth], axis=0)
    mean_activity.append(np.mean(activity, axis=0))
    std_activity.append(np.std(activity, axis=0))

#Compute d'
selectivity= (mean_activity[0]-mean_activity[1])
selectivity /= np.sqrt((std_activity[0]**2+std_activity[1]**2+1e-7)/2)

#Sort index for selectivity, seperately for E and I
ind_sort = np.concatenate((np.argsort(selectivity[:e_size]),
                          np.argsort(selectivity[e_size:])+e_size))


# plot distribution of stimulus selectivity
plt.figure()
plt.hist(selectivity)
plt.xlabel('selectiviy')
plt.ylabel('Number of neurons')

W = net.rnn.h2h.effective_weight().detach().numpy()
# Sort by selectivity
W = W[:, ind_sort][ind_sort, :]
wlim = np.max(np.abs(W))
plt.figure()
plt.imshow(W, cmap='bwr_r', vmin=-wlim, vmax=wlim)
plt.colorbar()
plt.xlabel('From neurons')
plt.ylabel('To neurons')
plt.title('Network connectivity')
            


stim0correct_response = np.array(condition[0])
stim0error_response = np.array(condition[1])
stim1corect_response = np.array(condition[2])
stim1error_response = np.array(condition[3])

mean_stim0correct_response = np.mean(stim0correct_response)
mean_stim0error_response = np.mean(stim0error_response)
mean_stim1corect_response = np.mean(stim1corect_response)
mean_stim1error_response = np.mean(stim1error_response)