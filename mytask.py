# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 11:48:33 2021

the definition of task variable sign
heading: range from 0~2pi, heading angle > pi belong to Categary1(C1),
heading angle < pi belong to Categary 2(C2)
Target loc:  T1 is the target of choose C1, T2 is the target of choose C2,
in HD task, T1 always locate in 0 direction (right side of the screen)
in colortarget HD task T1 loc if assigned by stim2 loc 

@author: NaN
"""

# delay saccade task


from __future__ import division
import six
import numpy as np

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
        self.pref = np.arange(0,2*np.pi,2*np.pi/self.n_eachring) # preferences    
        self.ypref = np.arange(0,2*np.pi,2*np.pi/16) # preferences   
        
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
    
    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
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
                self.x[ons[i]: offs[i], i, 1+(mods[i]-1)*self.n_eachring:1+mods[i]*self.n_eachring] \
                    += self.add_x_loc(locs[i])*strengths[i]        
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0        
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i])*strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i]        
            else:
                raise ValueError('Unknown loc_type')        
                
    def add_x_noise(self,config):
        """Add input noise."""
        n_heading = config['n_input_heading']
        x_heading = self.x[:,:,:n_heading]
        x_noise = np.random.randn(*x_heading.shape)
        
        # self.x += self.config['rng'].randn(*self.x.shape)*self._sigma_x    
        self.x[:,:,:n_heading] += x_noise*self._sigma_x
        self.x[:,:,:n_heading] += x_noise*self._sigma_x*np.mean(self.x[:,:,1:n_heading],axis=2,keepdims=True)
        # self.x += np.random.randn(*self.x.shape)*self._sigma_x*np.sqrt(self.x)
    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """     
        
        pre_on   = int(100/self.dt) # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)        
        
        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # c_mask[post_ons[i]:, i, :] = 5.
                c_mask[pre_offs[i]:, i, :] = 10.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 2.       
                # try not limit prepare saccade response % add 2025/5/5
                # c_mask[pre_on:pre_offs[i], i, 1:] = 0.
            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 1. # Fixation is important  
            self.c_mask = c_mask             
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Having it larger than 1 encourages the network to achieve higher performance
                c_mask[post_ons[i]:, i] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i] = 1.

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size,))
            self.c_mask /= self.c_mask.mean()
        
    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input."""
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start']+rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength        
        
    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc-self.pref)  # periodic boundary
        dist /= np.pi/8
        return 0.8*np.exp(-dist**2/2)        
        
    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc-self.ypref)  # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi/8
            y = 0.8*np.exp(-dist**2/2)
        else:
            # One-hot output
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y        
        
        
def test_init(config, mode, **kwargs):
    '''
    Test initialization of model. mode is not actually used
    Fixation is on then off.
    '''
    dt = config['dt']
    tdim = int(10000/dt)
    fix_offs  = [int(800/dt)]
    batch_size = 1

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)

    return trial        
        
def delaysaccade_(config, mode,stim_mod,**kwargs):        
    '''
    Fixate whenever fixation point is shown,
    saccade to the location of the previously shown stimulus
    whenever the fixation point is off
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (stim_on, stim_off)

    The output should be fixation location for (0, fix_off)
    and the stimulus location for (fix_off, T)

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
        
        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size)*2*np.pi      
        # stim_locs = np.linspace(0,315,num=2)/180*np.pi
        stim_ons  = int(rng.choice([300, 500, 700])/dt)     
        # fix_offs = int(1000/dt)  
        fix_offs = stim_ons + int(rng.choice([300, 500, 700, 900, 1100,1300])/dt)  
        stim_offs = fix_offs + int(rng.choice([200, 400, 600])/dt)   
        # stim_offs = stim_ons + int(rng.choice([200, 400, 600])/dt)         
        tdim     = stim_offs + int(500/dt)          
        
    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        batch_size = len(stim_locs)
        
        params = kwargs['params']
        stim_dur = params['stim_time']
        stim_dur = int(stim_dur/dt)
        stim_on = int(rng.uniform(300,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        fix_offs = stim_ons + int(rng.choice([700])/dt)  
        stim_offs = stim_on+stim_dur
        tdim     = stim_offs + int(500/dt)
        
    check_ons= stim_offs + int(100/dt)        
    # Response locations
    stim_locs = np.array(stim_locs)      
    response_locs = stim_locs
        
        
    trial = Trial(config, tdim, batch_size, stim_ons)        
    trial.add('fix_in', offs=fix_offs)        
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=1)        
    trial.add('fix_out', offs=fix_offs)        
    trial.add('out', response_locs, ons=fix_offs)        
    trial.add_c_mask(pre_offs=stim_offs, post_ons=check_ons)        
        
    trial.epochs = {'fix1'     : (None, stim_ons),
                   'delay1'   : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}        
        
    return trial        
        
def delaysaccade(config, mode, stim_mod,**kwargs):
    return delaysaccade_(config, mode, False, **kwargs) 

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
        fix_offs = (stim_ons+stim_dur+int(50/dt)).astype(int)        
        
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
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        fix_offs = (stim_ons+stim_dur+int(50/dt)).astype(int)        
        
        tdim = stim_on+stim_dur+int(500/dt)     
        
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    trial = Trial(config, tdim, batch_size,stim_ons)
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

def dm(config,mode,stim_mod,**kwargs):
    return dm_(config,mode,stim_mod,**kwargs)




def coltargdm(config, mode,stim_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    One stimuli is shown in ring 1 for 1000ms,
    at the same time two different color dot are show in ring 2 and ring 3.
    If the stimulus is category 1, then go to the location of ring 2, otherwise ring 3    

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
        # stim1_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))        
        # stim1_locs_part1 = np.random.uniform(2/3*np.pi,4/3*np.pi,(int(batch_size),))
        # stim1_locs = stim1_locs_part1
        stim1_locs_part1 = np.random.uniform(2/3*np.pi,4/3*np.pi,(int(batch_size/3),))
        stim1_locs_part2 = np.random.uniform(19/20*np.pi,21/20*np.pi,(int(batch_size/3),))
        stim1_locs_part3 = np.random.uniform(29/30*np.pi,31/30*np.pi,(batch_size-int(batch_size*2/3),))
        stim1_locs = np.concatenate((stim1_locs_part1,stim1_locs_part2,stim1_locs_part3))
        # stim1_locs_part1 = np.random.uniform(2/3*np.pi,4/3*np.pi,(int(batch_size*7/10),))
        # stim1_locs_part2 = np.random.uniform(19/20*np.pi,21/20*np.pi,(batch_size-int(batch_size*3/10),))
        # stim1_locs = np.concatenate((stim1_locs_part1,stim1_locs_part2))
        # stim1_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        # stim1_coh_range = np.array([0.08])
        # stim1_locs = np.random.uniform(2/3*np.pi,4/3*np.pi,int(batch_size))
        # stim1_coh_range = np.random.uniform(0.01,0.08,batch_size)
        stim1_coh_range = np.random.uniform(0.05,0.08,batch_size)
        stim1_coh_range *= 30
        stims1_coh  = rng.choice(stim1_coh_range, (batch_size,))
        
        # Color stimulus
        # stim2_locs = rng.choice([np.pi,2*np.pi], (batch_size,))
        stim2_locs = rng.choice([np.pi,0], (batch_size,))
        stim3_locs = (stim2_locs+np.pi)%(2*np.pi) 
        stims2_coh  = np.ones((batch_size,))
        stims3_coh  = np.ones((batch_size,))
        # Time of stimuluss on/off

        stim1_ons  = int(rng.uniform(100,600)/dt)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        stim1_offs = stim1_ons + stim_dur
        fix_offs  = stim1_offs + int(50/dt)
        stim2_ons  = stim1_ons
        stim2_offs = fix_offs + int(500/dt)
        stim3_ons  = stim1_ons
        stim3_offs = fix_offs + int(500/dt)

        
        tdim = fix_offs + int(500/dt)
    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stims1_coh = p['stim1_strengths']
        
        #color stimulus
        stim2_locs = p['stim2_locs']
        stim3_locs = p['stim3_locs']
        
        # Time of stim on/off
        batch_size = len(stim1_locs)
        
        stim_dur = p['stim_time']
        stim_dur = int(stim_dur/dt)
        stim1_ons = int(rng.uniform(100,600)/dt)
        stim1_offs = stim1_ons + stim_dur
        fix_offs = stim1_offs+int(50/dt)
        stim2_ons  = stim1_ons
        stim2_offs = fix_offs + int(500/dt)
        stim3_ons  = stim1_ons
        stim3_offs = fix_offs + int(500/dt)
        
        tdim = fix_offs + int(500/dt)
        
        
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = fix_offs + int(100/dt)       
    
    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    heading0_loc = stim1_locs==0
    stim1_cats[heading0_loc] = rng.choice([np.pi,0]) 
    trial = Trial(config, tdim, batch_size,stim1_ons)

    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stims1_coh, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons,offs=stim2_offs, mods=2)
    trial.add('stim', stim3_locs, ons=stim3_ons, offs=stim3_offs, mods=3)   
    
    # Target location
    stim_locs = list()
    for i in range(batch_size):
        if stim1_cats[i] == 0:
            stim_locs.append(stim2_locs[i])
        else:
            stim_locs.append(stim3_locs[i])
            
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', stim_locs, ons=fix_offs)
    
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'go1'      : (fix_offs, None)}

   
    return trial


rule_mapping ={'delaysaccade':delaysaccade,
               'dm':dm,
               'coltargdm':coltargdm}

rule_name ={'delaysaccade':'delaysaccade',
               'dm':'dm',
               'coltargdm':'coltargdm'}

def generate_trials(rule, hp, mode,stim_mod, noise_on=True, **kwargs):
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
    trial = rule_mapping[rule](config, mode,stim_mod, **kwargs)
    
    # Add rule input to every task  bnvvvv
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
        trial.add_x_noise(config)

    return trial


