# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:58:59 2023
the definition of task variable sign
heading: range from 0~2pi, heading angle > pi belong to Categary1(C1),
heading angle < pi belong to Categary 2(C2)
Target loc:  T1 is the target of choose C1, T2 is the target of choose C2,
in HD task, T1 always locate in 0 direction (right side of the screen)
in colortarget HD task T1 loc if assigned by stim2 loc 

different from file mytask.py, the input here has 3 dimension time*direction*velocity 
@author: NaN
"""
from __future__ import division
import six
import numpy as np

rules_dict = \
    {'all':['delaysaccade','coltargdm','dm'],
     'reall':['delaysaccade','dm','coltargdm'],
     'delaysaccade':['delaysaccade'],
     'dm':['dm'],
     'inference':['inference'],
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

def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size,ons):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32' # This should be the default
        self.config = config
        self.dt = self.config['dt']
        self.ons = ons

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.dir_pref = np.arange(0,2*np.pi,2*np.pi/np.sqrt(self.n_eachring)) # preferences
        self.ypref = np.arange(0,2*np.pi,2*np.pi/32) # preferences   
        self.vel_pref = np.arange(0,3,3/np.sqrt(self.n_eachring))#velocity  

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)        
        if self.config['loss_type'] == 'lsq':
            self.y[:,:,:] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response        
        self.y_loc = -np.ones((tdim, batch_size)      , dtype=self.float_type)

        self._sigma_x = config['sigma_x']*np.sqrt(2/config['alpha'])        

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var 

    def add(self, loc_type, loc_dir=None, loc_vel=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            loc_dir:
            loc_vel:
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            velocity: float or list, velocity of input
            mods: int or list, modalities of input or target output(decide which ring to add infor)
        """
        ons = self.expand(ons) 
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)
        
        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1    
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[ons[i]:offs[i],i,1+(mods[i]-1)*self.n_eachring:1+mods[i]*self.n_eachring]\
                    += self.add_x_loc(loc_dir[i],loc_vel[i])*strengths[i]
            elif loc_type == 'fix_out':
                #Notice this shouldn't be set as 1, because the output is logistic and saturate at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i],i,0]=0.8
                else:
                    self.y[ons[i]:offs[i],i,0] = 1
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i],i,1:] += self.add_y_loc(loc_dir[i])
                else:
                    y_tmp = self.add_y_loc(loc_dir[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]:offs[i],i,1:] += y_tmp
                self.y_loc[ons[i]:offs[i],i] = loc_dir[i]
            else:
                raise ValueError('Unknown loc_type')
                
    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape)*self._sigma_x
    
    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask.           

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """
        
        pre_on = int(100/self.dt) # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)
        
        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output),dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[pre_offs[i]:,i,:] = 10.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i],i,:] = 2.
            
            c_mask[:,:,0] *= 1. # Fixation is important
            self.c_mask = c_mask
        else:
            c_mask = np.zeros((self.tdim, self.batch_size),dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[post_ons[i]:, i] = 5.
                c_mask[pre_on:pre_offs[i], i] = 1.
        
            self.c_mask = c_mask.reshape((self.tdim*self.batch_size,))
            self.c_mask /=self.c_mask.mean()
            
    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input."""
        if isinstance(rule,int):
            self.x[on:off,:,self.config['rule_start']+rule]=strength
        else:
            ind_rule = get_rule_index(rule,self.config)
            self.x[on:off,:,ind_rule] = strength
                    
    def add_x_loc(self, x_loc_dir, x_loc_vel):
        dist_dir = get_dist(x_loc_dir-self.dir_pref) #periodic boundary
        dist_dir /= np.pi/8
        dist_dir = 0.8*np.exp(-dist_dir**2/2)
        dist_vel = get_dist(x_loc_vel-self.vel_pref)
        dist = dist_dir.T@dist_vel
        dist = dist.reshape((-1))
        return dist
        
    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc-self.ypref) # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi/8
            y = 0.8*np.exp(-dist**2/2)
        else:
            #One-hot output
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y
        
        
def test_init(config, mode, **kwargs):
    '''
    Test initialization of model. mode is not actually used
    Fixation is on then off.

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    mode : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    dt = config['dt']
    tdim = int(10000/dt)
    fix_offs = [int(800/dt)]
    batch_size = 1
    
    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
        
    return trial

def inference_(config, mode, stim_mod,**kwargs):
    '''
    fixate whenever fixation point is shown.
    the stimulus and two targets are shown(stimulus is consisted of 2 modalities, 
        the task is to discrim it's homology or not)
    stimulus off
    fixation off
    saccade to one of the target according to the homology of 2 stimuli

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    mode : TYPE
        the mode of generating. Options: 'random','explicit'...
    stim_mod : int
        stim modality.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    dt = config['dt']
    rng = config['rng']
    
    if mode == 'random':#randomly generate parameters
        batch_size = kwargs['batch_size']
        if stim_mod == 1: # different motion direction
            delta_dir = rng.choice([1/36*np.pi,1/18*np.pi,1/9*np.pi,1/3*np.pi,
                                          -1/36*np.pi,-1/18*np.pi,-1/9*np.pi,-1/3*np.pi],size=(batch_size,))
            stim1_loc_dir = rng.choice([1/3*np.pi, 1/2*np.pi, 2/3*np.pi, np.pi],size=(batch_size,))
            stim2_loc_dir = stim1_loc_dir+delta_dir
            
            stim1_loc_vel = np.ones(batch_size)*1
            stim2_loc_vel = np.ones(batch_size)*1
            stim_coh_range = np.random.uniform(0.01,0.08,batch_size)
            if ('easy_task' in config) and config['easy_task']:
                stim_coh_range *= 30
    
        stims_coh = rng.choice(stim_coh_range,(batch_size,))
        stim1_strengths = stims_coh
        stim2_strengths = stims_coh
 
        # Time of stimulus on/off 
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)  
        stim_dur = int(rng.choice([400,800,1600])/dt)
        fix_offs = (stim_ons+stim_dur+int(50/dt)).astype(int)         
        
        tdim = stim_on+stim_dur+int(500/dt)
        
    elif mode == 'psychometric' :
        p = kwargs['params']
        stim_locs = p['stim_locs']
        stim_strenghs = p['stim_strengths']
        # Time of stimulus on/off
        batch_size = len(stim_locs)
        
        stim_dur = p['stim_time']
        stim_dur = int(stim_dur/dt)
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        fix_offs = (stim_ons+stim_dur+int(50/dt)).astype(int)
        
        tdim = stim_on+stim_dur+int(500/dt)
        
    else:
        raise ValueError('Unknown mode:' + str(mode))
        
    # time to check the sacccade location
    check_ons = fix_offs + int(100/dt)
                                  
    trial = Trial(config, tdim, batch_size, stim_ons)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_loc_dir, stim1_loc_vel, ons=stim_ons, offs=fix_offs, strengths=stim1_strengths, mods=1)  
    trial.add('stim', stim2_loc_dir, stim2_loc_vel, ons=stim_ons, offs=fix_offs, strengths=stim2_strengths, mods=1)       
    trial.add('fix_out', offs=fix_offs)
    
    stim_discrepancy = stim1_loc_dir-stim2_loc_dir
    stim_cats = stim_discrepancy<=np.pi/10 # if stim_cats=1 the stimulus are homology
    
    # Target location
    out_locs = list()
    for i in range(batch_size):
        if stim_cats[i] == 0:
            out_locs.append(0)
        else:
            out_locs.append(np.pi)
            
    trial.add('out', out_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)
    
    trial.epochs = {'fix1'   : (None, stim_ons),
                    'stim1'  : (stim_ons, fix_offs),
                    'go1'    : (fix_offs,None)}
    
    return trial
            
    
def inference(config, mode, stim_mod,**kwargs):
    return inference_(config,mode,stim_mod,**kwargs)    


rule_mapping = {'inference':inference}

rule_name = {'inference':'inference'}

def generate_trials(rule, hp, mode, stim_mod, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        stim_mod: 1 is fine task ,2 is coarse task
        noise_on: bool, whether input noise is given  (~0.03)

    Return:
        trial: Trial class instance, containing input and target output
    """

    config = hp
    trial = rule_mapping[rule](config, mode, stim_mod, **kwargs)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else: # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else: # default behavior
        rule_off = None
        
        
    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']   
        
    if rule is 'testinit':
        # Add no rule
        return trial       

    if isinstance(rule, six.string_types):
        # rule is not iterable
        # Expand to list
        if 'rule_strength' in kwargs:
            rule_strength = [kwargs['rule_strength']]
        else:
            rule_strength = [1.]
        rule = [rule]
    else:
        if 'rule_strength' in kwargs:
            rule_strength = kwargs['rule_strength']
        else:
            rule_strength = [1.] * len(rule)    
            
    for r, s in zip(rule, rule_strength):
        trial.add_rule(r, on=rule_on, off=rule_off, strength=s)
        
    if noise_on:
        trial.add_x_noise()

    return trial
        
        
        
        
        
        
        
        
        

        

