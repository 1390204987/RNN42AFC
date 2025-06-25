# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:49:15 2022

for some function used frequently in different script

@author: NaN
"""
import numpy as np
import torch
import json

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

def popvec_tensor(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    y_output = y[...,1:-1]
    pref = torch.arange(0, 2*np.pi, 2*np.pi/y_output.shape[-1])
    temp_sum = y_output.sum(axis=-1)
    temp_cos = torch.sum(y_output*torch.cos(pref), axis=-1)/temp_sum
    temp_sin = torch.sum(y_output*torch.sin(pref), axis=-1)/temp_sum    
    loc = torch.atan2(temp_sin, temp_cos)
    return loc

def popvec_2d_tensor(y,n_eachring):
    """Population vector read out along time.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (time,Batch, Units)

    Returns:
        Readout locations: Numpy array (time,Batch,)
    """
    y_max_value,y_max_indices = torch.max(y,dim = 2)
    fixdim = (y_max_indices==0)*1
    fixdim = fixdim.float()
    # fixdim = torch.zeros(np.shape(y_max_indices))
    saccadedim = y_max_indices/n_eachring*2*np.pi
    y_2d = (fixdim,saccadedim)
    return y_2d
    
    
        
def get_y_direction(y_hat,y_loc):
    """ Get the output(saccade) direction.
    
    Args:
        y_hat: Actual output.Numpy array(Times, Batch, Unit)
       
    Returns:
        y_direction: Numpy array(Batch,)        
    """
    y_hat_np = y_hat.detach().numpy()
    if len(y_hat_np.shape) != 3:
        raise ValueError('y_hat_np must have shape (Time, Batch, Unit)')
    # Only look at last time points    
    y_hat_np = y_hat_np[-1]        
   
    # get the location of y_hat 
    y_hat_loc = popvec(y_hat_np[..., 1:])
    
    # check whether the outputs belong to the y_locs+- 0.2*pi
    y_loc = y_loc[-1]
    unique_y_loc = np.unique(y_loc)
    
    y_direction = np.full_like(y_hat_loc,np.nan)
    for i_loc in range(len(unique_y_loc)):
        y_direction[np.where(np.abs(y_hat_loc-unique_y_loc[i_loc])<=0.2*np.pi)] = unique_y_loc[i_loc]
        y_direction[np.where(np.abs(np.abs(y_hat_loc-unique_y_loc[i_loc])-2*np.pi)<=0.2*np.pi)] = unique_y_loc[i_loc]
    return y_direction

def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    y_hat_np = y_hat.detach().cpu().numpy()
    if len(y_hat_np.shape) != 3:
        raise ValueError('y_hat_np must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat_np = y_hat_np[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat_np[..., 0]
    y_hat_loc = popvec(y_hat_np[..., 1:])

    # Fixating? Correctly saccading? (at the last time point)
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.1*np.pi

    # Should fixate? in HD and color target HD this value is False
    should_fix = y_loc < 0 

    # performance
    # perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    perf = np.sum((1-should_fix) * corr_loc * (1-fixating))/len(corr_loc)
    
    return perf
