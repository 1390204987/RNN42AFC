# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 11:48:33 2021

the definition of task variable sign
heading: range from 0~2pi, heading angle > pi belong to Categary1(C1),
heading angle < pi belong to Categary2(C2)
Target loc:  T1 is the target of choose C1, T2 is the target of choose C2,
in HD task, T1 always locate in 0 direction (right side of the screen)
in colortarget HD task T1 loc if assigned by stim2 loc 

@author: NaN
"""

# delay saccade task


from __future__ import division
import six
import numpy as np
import torch

rules_dict = \
    {'all':['delaysaccade','coltargdm','dm'],
     'reall':['delaysaccade','dm','coltargdm'],
     'delaysaccade':['delaysaccade'],
     'dm':['dm'],
     '2AFC':['dm','coltargdm'],
     'coltargdm':['coltargdm'],
     'dsaccoldm':['delaysaccade','coltargdm'],
     'dsacdm':['delaysaccade','dm']}

# Store indices of rules
rule_index_map = dict()
# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind
        
        
def get_num_ring(ruleset):       
    '''get number of stimulus rings'''
    return 1

def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])

def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule]+config['rule_start']

# def get_dist(original_dist):
#     '''Get the distance in periodic boundary conditions'''
#     return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

def get_dist(original_dist, device='cpu'):
    '''Get the distance in periodic boundary conditions (PyTorch version)'''
    if isinstance(original_dist, np.ndarray):
        original_dist = torch.from_numpy(original_dist).to(device)  # 兼容 NumPy 输入
    return torch.minimum(
        torch.abs(original_dist),
        2 * torch.pi - torch.abs(original_dist)
    )

class Trial(object):
    """Class representing a batch of trials with GPU support."""

    def __init__(self, config, tdim, batch_size, on, off, device='cpu'):
        """Initialize a batch of trials on specified device.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
            device: str, 'cpu' or 'cuda'
        """
        self.device = device
        self.float_type = torch.float32  # Using torch dtype instead of numpy
        self.config = config
        self.dt = self.config['dt']
        self.on = on
        self.off = off
        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        
        # Generate preferences directly on target device
        self.pref = torch.linspace(0, 2*np.pi, self.n_eachring+1, device=self.device)[:-1]  # preferences
        self.ypref = torch.linspace(0, 2*np.pi, 16+1, device=self.device)[:-1]  # preferences
        
        self.batch_size = batch_size
        self.tdim = tdim
        
        # Initialize tensors directly on target device
        self.x = torch.zeros((tdim, batch_size, self.n_input), 
                          dtype=self.float_type, device=self.device)
        self.y = torch.zeros((tdim, batch_size, self.n_output), 
                          dtype=self.float_type, device=self.device)
        
        if self.config['loss_type'] == 'lsq':
            self.y[:,:,:] = 0.05
            
        # y_loc is the stimulus location of the output
        self.y_loc = -torch.ones((tdim, batch_size), 
                               dtype=self.float_type, device=self.device)

        # Convert numpy calculation to torch
        self._sigma_x = config['sigma_x'] * torch.sqrt(torch.tensor(2/config['alpha'], 
                                                                  device=self.device))

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var 
    
    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or stimulus output with full GPU support.
    
        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
        """
        # Convert all inputs to tensors on the correct device
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = [torch.tensor(s, dtype=self.float_type, device=self.device) 
                    if not isinstance(s, torch.Tensor) else s.to(self.device)
                    for s in self.expand(strengths)]
        mods = self.expand(mods)
    
        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]:offs[i], i, 0] = 1
            elif loc_type == 'stim':
                if mods[0] == 1:
                    # GPU-optimized Gaussian window
                    t_start, t_end = ons[i], offs[i]
                    t_center = (t_start + t_end) / 2
                    sigma = (t_end - t_start) /7
                    
                    # Generate time array directly on GPU
                    t = torch.arange(t_start, t_end, device=self.device)
                    
                    # Compute Gaussian weights
                    gaussian_weights = strengths[i] * torch.exp(-0.5 * ((t - t_center) / sigma)**2)
                    
                    # Apply to stimulus signal
                    stim_signal = self.add_x_loc(locs[i]) * gaussian_weights.unsqueeze(1)
                    stim_slice = slice(1+(mods[i]-1)*self.n_eachring, 1+mods[i]*self.n_eachring)
                    self.x[ons[i]:offs[i], i, stim_slice] += stim_signal
                else:
                    stim_slice = slice(1+(mods[i]-1)*self.n_eachring, 1+mods[i]*self.n_eachring)
                    self.x[ons[i]:offs[i], i, stim_slice] += self.add_x_loc(locs[i]) * strengths[i]
                    
            elif loc_type == 'fix_out':
                output_val = torch.tensor(0.8 if self.config['loss_type'] == 'lsq' else 1.0,
                                        device=self.device)
                self.y[ons[i]:offs[i], i, 0] = output_val
                
            elif loc_type == 'out':
                y_loc_signal = self.add_y_loc(locs[i])
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i], i, 1:] += y_loc_signal * strengths[i]
                else:
                    y_tmp = y_loc_signal / torch.sum(y_loc_signal)
                    self.y[ons[i]:offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]:offs[i], i] = locs[i]
                
            else:
                raise ValueError('Unknown loc_type')    
                
    def add_x_noise(self, config):
        """Add input noise with GPU support and temporal smoothing.
        
        Args:
            config: configuration dictionary containing noise parameters
        """
        n_heading = config['n_input_heading']
        
        # Generate noise directly on GPU
        noise_shape = ((self.off - self.on)//1, self.batch_size, n_heading - 1)
        seed=config["seed"]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        x_noise = torch.randn(*noise_shape, device=self.device, generator=generator)* self._sigma_x
        # 添加dropout使噪声稀疏化
        dropout_rate = 0  # 调整这个值控制稀疏程度（0-1）
        # x_noise = torch.nn.functional.dropout(x_noise, p=dropout_rate)
        # Smooth noise along time dimension (axis=0)
        if x_noise.shape[0] > 1:  # Only smooth if multiple time steps
            # Method 1: Gaussian smoothing (PyTorch implementation)
            kernel_size = 7# Should be odd
            sigma = 1.0
            channels = x_noise.shape[2]
            
            # Create 1D Gaussian kernel
            x = torch.arange(kernel_size, device=self.device) - kernel_size//2
            gauss_kernel = torch.exp(-x**2/(2*sigma**2))
            gauss_kernel = gauss_kernel / gauss_kernel.sum()
            
            # Reshape for conv1d (output_channels, input_channels, kernel_size)
            gauss_kernel = gauss_kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
            
            # Pad and convolve each channel separately
            padding = kernel_size // 2
            x_noise_smoothed = torch.zeros_like(x_noise)
            for b in range(self.batch_size):
                for c in range(channels):
                    # Reshape to (1, 1, time) for conv1d
                    input_slice = x_noise[:, b, c].view(1, 1, -1)
                    smoothed = torch.nn.functional.conv1d(
                        input_slice,
                        gauss_kernel[c:c+1],
                        padding=padding
                    )
                    x_noise_smoothed[:, b, c] = smoothed.squeeze()
            
            x_noise = x_noise_smoothed
    
        # Add noise to the appropriate slice
        self.x[self.on:self.off, :, 1:n_heading] += x_noise
        # self.x[self.on:self.on+(self.off - self.on)//2, :, 1:n_heading] += x_noise
        # self.x[self.on+(self.off - self.on)//2:self.off, :, 1:n_heading] += x_noise      
    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask with GPU support.
        
        Args:
            pre_offs: list of int, pre-response offset times
            post_ons: list of int, post-response onset times
        """
        pre_on = int(100/self.dt)  # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)
        
        if self.config['loss_type'] == 'lsq':
            # Initialize mask on correct device
            c_mask = torch.zeros((self.tdim, self.batch_size, self.n_output), 
                               dtype=self.float_type, device=self.device)
            
            for i in range(self.batch_size):
                # Post-response period
                c_mask[pre_offs[i]:, i, :] = 10.
                
                # Pre-response period
                c_mask[pre_on:pre_offs[i], i, :] = 2.
                
            # Fixation channel weighting
            c_mask[:, :, 0] *= 1.
            self.c_mask = c_mask
        else:
            # Cross-entropy loss version
            c_mask = torch.zeros((self.tdim, self.batch_size), 
                               dtype=self.float_type, device=self.device)
            
            for i in range(self.batch_size):
                # Post-response period
                c_mask[post_ons[i]:, i] = 5.
                
                # Pre-response period
                c_mask[pre_on:pre_offs[i], i] = 1.
            
            # Normalize and flatten
            self.c_mask = c_mask.view(-1)
            if self.c_mask.mean() > 0:  # Avoid division by zero
                self.c_mask /= self.c_mask.mean()
            
    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input with GPU support."""
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start']+rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength
    
    def add_x_loc(self, x_loc):
        """Input activity given location (GPU version)."""
        # Convert x_loc to tensor if it isn't already
        if not isinstance(x_loc, torch.Tensor):
            x_loc = torch.tensor(x_loc, device=self.device)
        
        # Compute distance with periodic boundary
        dist = get_dist(x_loc - self.pref)
        dist = dist / (torch.pi/8)  # Normalize
        return 0.8 * torch.exp(-dist**2/2)
    
    def add_y_loc(self, y_loc):
        """Target response given location (GPU version)."""
        # Convert y_loc to tensor if it isn't already
        if not isinstance(y_loc, torch.Tensor):
            y_loc = torch.tensor(y_loc, device=self.device)
        
        # Compute distance with periodic boundary
        dist = get_dist(y_loc - self.ypref)
        
        if self.config['loss_type'] == 'lsq':
            dist = dist / (torch.pi/8)  # Normalize
            y = 0.8 * torch.exp(-dist**2/2)
        else:
            # One-hot output
            y = torch.zeros_like(dist)
            ind = torch.argmin(dist)
            y[ind] = 1.
        return y
        
        
def test_init(config, mode, device='cuda', **kwargs):
    '''
    Test initialization of model with GPU support.
    Fixation is on then off.
    
    Args:
        config: configuration dictionary
        mode: unused (kept for compatibility)
        device: str, target device ('cuda' or 'cpu')
        **kwargs: additional arguments
        
    Returns:
        Trial object initialized on specified device
    '''
    dt = config['dt']
    tdim = int(10000/dt)
    fix_offs = [int(800/dt)]
    batch_size = 1

    # Initialize Trial on target device
    trial = Trial(config, tdim, batch_size, on=0, off=tdim, device=device)
    
    # Add fixation input
    trial.add('fix_in', offs=fix_offs)

    return trial    
        
def delaysaccade_(config, mode, stim_mod, device='cuda', **kwargs):
    '''
    GPU-optimized version of delaysaccade task generator.
    
    Args:
        config: configuration dictionary
        mode: generation mode ('random' or 'psychometric')
        stim_mod: stimulus modality
        device: target device ('cuda' or 'cpu')
        **kwargs: additional parameters depending on mode
        
    Returns:
        Trial object initialized on specified device
    '''
    dt = config['dt']
    
    # Convert rng to torch if needed (assuming config['rng'] is numpy RandomState)
    if 'rng' in config and isinstance(config['rng'], np.random.RandomState):
        torch.manual_seed(config['rng'].randint(2**32))
    
    if mode == 'random':
        batch_size = kwargs['batch_size']
        
        # Generate parameters directly on target device
        stim_locs = torch.rand(batch_size, device=device) * 2 * torch.pi
        stim_ons = int(random.choice([300, 500, 700]) / dt)
        
        # Delay periods - converted to tensor for potential GPU ops
        delay_options = torch.tensor([300, 500, 700, 900, 1100, 1300], device=device)
        fix_offs = stim_ons + int(delay_options[torch.randint(0, len(delay_options), (1,))] / dt)
        
        response_options = torch.tensor([200, 400, 600], device=device)
        stim_offs = fix_offs + int(response_options[torch.randint(0, len(response_options), (1,))] / dt)
        
        tdim = stim_offs + int(500/dt)
        
    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = torch.tensor(p['stim_locs'], device=device)
        batch_size = len(stim_locs)
        
        params = kwargs['params']
        stim_dur = int(params['stim_time']/dt)
        stim_on = int(random.uniform(300,400)/dt)
        stim_ons = torch.full((batch_size,), stim_on, dtype=torch.int32, device=device)
        fix_offs = stim_ons[0] + int(700/dt)  # Fixed delay for psychometric
        stim_offs = stim_on + stim_dur
        tdim = stim_offs + int(500/dt)
    
    check_ons = stim_offs + int(100/dt)
    response_locs = stim_locs  # For this task, response matches stimulus location
    
    # Initialize trial on target device
    trial = Trial(config, tdim, batch_size, stim_ons.item(), stim_offs.item(), device=device)
    
    # Add trial components
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons.item(), offs=stim_offs.item(), mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=stim_offs.item(), post_ons=check_ons)
    
    # Set trial epochs
    trial.epochs = {
        'fix1': (None, stim_ons.item()),
        'delay1': (stim_ons.item(), fix_offs),
        'go1': (fix_offs, None)
    }
    
    return trial
def delaysaccade(config, mode, stim_mod, device='cuda', **kwargs):
    """
    GPU-optimized wrapper for delaysaccade task generation.
    
    Args:
        config: configuration dictionary
        mode: generation mode ('random' or 'psychometric')
        stim_mod: stimulus modality (passed but not used in this wrapper)
        device: target device ('cuda' or 'cpu')
        **kwargs: additional parameters depending on mode
        
    Returns:
        Trial object initialized on specified device
    """
    # Explicitly pass device parameter to the underlying function
    return delaysaccade_(config, mode, stim_mod=False, device=device, **kwargs)

def dm_(config, mode, stim_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    the stimuluss and two target are shown
    stimulus off
    fixation off
    saccade two one of the target acorrding to the stimulus direction and target location
    
    :param mode: the mode of generating. Options: 'random', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']

    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        if stim_mod == 1: # fine task
            # stim_locs_part1 = np.random.uniform(2/3*np.pi,4/3*np.pi,(int(batch_size/4),))
            # stim_locs_part2 = np.random.uniform(19/20*np.pi,21/20*np.pi,(batch_size-int(batch_size/4),))
            # stim_locs_part3 = np.random.uniform(29/30*np.pi,31/30*np.pi,(batch_size-int(batch_size/4),))
            # stim_locs_part4 = np.random.uniform(39/40*np.pi,41/40*np.pi,(batch_size-int(batch_size/4),))
            # stim_locs = np.concatenate((stim_locs_part1,stim_locs_part2,stim_locs_part3,stim_locs_part4))
            stim_locs_part1 = np.random.uniform(2/3*np.pi,4/3*np.pi,(int(batch_size/4),))
            stim_locs_part2 = np.random.uniform(19/20*np.pi,21/20*np.pi,(int(batch_size/4),))
            stim_locs_part3 = np.random.uniform(29/30*np.pi,31/30*np.pi,(batch_size-int(batch_size*2/4),))
            stim_locs = np.concatenate((stim_locs_part1,stim_locs_part2,stim_locs_part3))
            # stim_locs_range = np.array([-24,-12,-6, 6, 12,24])/360*np.pi+np.pi
            # stim_locs = rng.choice(stim_locs_range,(batch_size,))
            # stim_coh_range = np.array([0.08])
            stim_coh_range = np.random.uniform(0.02,0.05,batch_size)
            if ('easy_task' in config) and config['easy_task']:
                stim_coh_range *= 30
        if stim_mod == 2: #coarse task
            stim_locs = rng.choice(0, np.pi, (batch_size,))
            stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
            if ('easy_task' in config) and config['easy_task']:
                stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])

            
        #stims_mean = rng.uniform(0.8,1.2,(batch_size,))            

        stims_coh  = rng.choice(stim_coh_range, (batch_size,))
        stim_strengths = stims_coh
        
        # Time of stimuluss on/off
        stim_on = int(rng.uniform(100,400)/dt)        
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        stim_off = (stim_on+stim_dur)
        fix_offs = (stim_ons+stim_dur+int(50/dt)).astype(int)        
        stim_offs = (stim_ons+stim_dur).astype(int)  
        tdim = stim_on+stim_dur+int(500/dt)     
    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        stim_strengths = p['stim_strengths']
        # Time of stimuluss on/off
        batch_size = len(stim_locs)
        
        stim_dur = p['stim_time']
        stim_dur = int(stim_dur/dt)
        stim_on = int(rng.uniform(100,400)/dt)
        stim_off = (stim_on+stim_dur)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        fix_offs = (stim_ons+stim_dur+int(50/dt)).astype(int)                
        tdim = stim_on+stim_dur+int(500/dt)     
        
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    trial = Trial(config, tdim, batch_size,stim_on,stim_off)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=fix_offs, strengths=stim_strengths, mods=1)
    trial.add('fix_out', offs=fix_offs)


    stim_cats = stim_locs<=3.15 # Category of stimulus 1
    # Target location
    out_locs = list()
    for i in range(batch_size):
        if stim_cats[i] == 0:
            out_locs.append(np.pi)
        else:
            out_locs.append(0)
            
    trial.add('out', out_locs, ons=fix_offs)    
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)    

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}    
    
    return trial

def dm(config, mode, stim_mod, device='cuda', **kwargs):
    """
    GPU-optimized wrapper for direction discrimination task generation.
    
    Args:
        config: configuration dictionary
        mode: generation mode ('random' or 'psychometric')
        stim_mod: stimulus modality (1: fine, 2: coarse)
        device: target device ('cuda' or 'cpu')
        **kwargs: additional parameters depending on mode
        
    Returns:
        Trial object initialized on specified device
    """
    # Pass all parameters including device to the underlying function
    return dm_(config, mode, stim_mod, device=device, **kwargs)




def coltargdm(config, mode, stim_mod, device='cuda', **kwargs):
    '''
    GPU-optimized color target discrimination task generator.
    
    Args:
        config: configuration dictionary
        mode: generation mode ('random' or 'psychometric')
        stim_mod: stimulus modality (unused in this task but kept for compatibility)
        device: target device ('cuda' or 'cpu')
        **kwargs: additional parameters depending on mode
        
    Returns:
        Trial object initialized on specified device
    '''
    dt = config['dt']
    
    # Convert rng to torch if needed
    if 'rng' in config and isinstance(config['rng'], np.random.RandomState):
        torch.manual_seed(config['rng'].randint(0, 2**31-1))
    
    if mode == 'random':
        batch_size = kwargs['batch_size']
        
        # Generate stimulus locations directly on GPU
          # 生成 stim1_locs（方向刺激）
        stim1_locs = torch.rand(batch_size, device=device) * 2 * torch.pi  # 初始化在 [0, 2π)
        
        # 强制其中一半 < π，另一半 ≥ π
        perm1 = torch.randperm(batch_size, device=device)
        half = batch_size // 2
        stim1_locs[perm1[:half]] = torch.rand(half, device=device) * torch.pi
        stim1_locs[perm1[half:]] = torch.rand(batch_size - half, device=device) * torch.pi + torch.pi
      
        # Generate coherence levels
        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        
        # # 从离散值中随机选取，每个 trial 一次
        # rand_idx = torch.randint(len(stim_coh_range), (batch_size,), device=device)
        # stims1_coh = torch.tensor(stim_coh_range, device=device)[rand_idx]
        
        # 生成 stim1 coherence
        stim1_coh_range = torch.rand(batch_size, device=device) * 2
        stims1_coh = stim1_coh_range
        
        # 生成 stim2_locs（颜色刺激）为 0 或 π，50% 为 π
        stim2_locs = torch.zeros(batch_size, device=device)
        perm2 = torch.randperm(batch_size, device=device)
        stim2_locs[perm2[:half]] = torch.tensor(torch.pi, device=device)  # 随机 50% 设置为 π
        
        stim3_locs = (stim2_locs + torch.pi) % (2*torch.pi)
        stims2_coh = torch.ones(batch_size, device=device)
        stims3_coh = torch.ones(batch_size, device=device)
        
        # Timing parameters - 使用 torch 的随机函数
        stim1_on = int(torch.randint(100, 600, (1,), device=device).item() / dt)  # 在CPU上生成然后转换
        stim_dur = int(torch.tensor([400, 800, 1600], device=device)[torch.randint(0, 3, (1,))].item() / dt)
        stim1_off = stim1_on + stim_dur
        fix_off = stim1_off + int(50/dt)
        stim2_on = stim1_on
        stim2_off = fix_off + int(500/dt)
        stim3_on = stim1_on
        stim3_off = fix_off + int(500/dt)
        tdim = fix_off + int(500/dt)
        
    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = torch.tensor(p['stim1_locs'], device=device)
        stims1_coh = torch.tensor(p['stim1_strengths'], device=device)
        stim2_locs = torch.tensor(p['stim2_locs'], device=device)
        stim3_locs = torch.tensor(p['stim3_locs'], device=device)
        batch_size = len(stim1_locs)
        
        stim_dur = int(p['stim_time']/dt)
        # stim1_on = int(torch.randint(100, 600, (1,), device=device).item() / dt)  # 在CPU上生成然后转换
        stim1_on = int(200/ dt)  # 在CPU上生成然后转换
        stim1_off = stim1_on + stim_dur
        fix_off = stim1_off + int(50/dt)
        stim2_on = stim1_on
        stim2_off = fix_off + int(500/dt)
        stim3_on = stim1_on
        stim3_off = fix_off + int(500/dt)
        tdim = fix_off + int(500/dt)
        
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # 其余代码保持不变...
    # Determine response locations
    stim1_cats = stim1_locs < torch.pi
    heading0_loc = stim1_locs == torch.pi
    stim1_cats[heading0_loc] = torch.rand(1, device=device) > 0.5
    
    # Initialize trial on target device
    trial = Trial(config, tdim, batch_size, stim1_on, stim1_off, device=device)
    
    # Add trial components
    trial.add('fix_in', offs=fix_off)
    trial.add('stim', stim1_locs, ons=stim1_on, offs=stim1_off, strengths=stims1_coh, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_on, offs=stim2_off, mods=2)
    trial.add('stim', stim3_locs, ons=stim3_on, offs=stim3_off, mods=3)
    
    # Determine target locations, true S3, false S2
    stim_locs = torch.where(stim1_cats, stim3_locs, stim2_locs)
    
    trial.add('fix_out', offs=fix_off)
    trial.add('out', stim_locs, ons=fix_off)
    trial.add_c_mask(pre_offs=fix_off, post_ons=fix_off + int(100/dt))
    
    trial.epochs = {
        'fix1': (None, stim1_on),
        'stim1': (stim1_on, stim1_off),
        'go1': (fix_off, None)
    }
    
    return trial


rule_mapping ={'delaysaccade':delaysaccade,
               'dm':dm,
               'coltargdm':coltargdm}

rule_name ={'delaysaccade':'delaysaccade',
               'dm':'dm',
               'coltargdm':'coltargdm'}

def generate_trials(rule, hp, device='cuda', mode='random', stim_mod=1, noise_on=True, **kwargs):
    """Generate one batch of data with GPU support.

    Args:
        rule: str or list, the rule(s) for this batch
        hp: dictionary of hyperparameters
        device: str, target device ('cuda' or 'cpu')
        mode: str, generation mode ('random', 'test', 'psychometric')
        stim_mod: int, 1 for fine task, 2 for coarse task
        noise_on: bool, whether to add input noise (~0.03)
        **kwargs: additional parameters depending on mode

    Returns:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    
    # Get the appropriate rule function and pass device parameter
    rule_func = rule_mapping[rule]
    trial = rule_func(config, mode, stim_mod, device=device, **kwargs)
    
    # Handle rule timing parameters
    rule_on = kwargs.get('rule_on', None)
    rule_off = kwargs.get('rule_off', None)
    
    # Handle rule replacement if specified
    rule = kwargs.get('replace_rule', rule)
    
    if rule == 'testinit':
        return trial
    
    # Convert rule to list if needed
    if isinstance(rule, str):
        rule_strength = [kwargs.get('rule_strength', 1.0)]
        rule = [rule]
    else:
        rule_strength = kwargs.get('rule_strength', [1.0] * len(rule))
    
    # Add rules to trial
    for r, s in zip(rule, rule_strength):
        trial.add_rule(r, on=rule_on, off=rule_off, strength=s)
        
    # Add input noise if enabled
    if noise_on:
        trial.add_x_noise(config)
    
    return trial


