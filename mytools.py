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
    """Population vector readout with GPU support.

    Args:
        y: population output on a ring network. Tensor (Batch, Units)

    Returns:
        Readout locations: Tensor (Batch,)
    """
    # Calculate sums
    y = torch.as_tensor(y, dtype=torch.float32)
    device = y.device
    n_units = y.shape[-1]
    
    # Generate preferences directly on the correct device
    pref = torch.linspace(0, 2*torch.pi, n_units+1, device=device)[:-1]
    

    temp_sum = y.sum(dim=-1, keepdim=True)
    
    temp_sum = torch.where(temp_sum == 0, torch.ones_like(temp_sum), temp_sum)  # Avoid division by zero
    
    # Calculate cosine and sine components
    temp_cos = torch.sum(y * torch.cos(pref), dim=-1) / temp_sum.squeeze(-1)
    temp_sin = torch.sum(y * torch.sin(pref), dim=-1) / temp_sum.squeeze(-1)
    
    # Calculate angle and wrap to [0, 2π)
    loc = torch.atan2(temp_sin, temp_cos)
    return loc % (2*torch.pi)

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
    
    
        
def get_y_direction(y_hat, y_loc):
    """Get the output (saccade) direction.

    Args:
        y_hat: Actual output. Tensor (Time, Batch, Unit)
        y_loc: Ground truth direction. Tensor or array (Time, Batch)

    Returns:
        y_direction: Tensor (Batch,)  — final decoded direction
    """
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat, dtype=torch.float32)
    
    if y_hat.ndim != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')

    device = y_hat.device

    # Only use the last time point
    y_hat_last = y_hat[-1]  # shape: (Batch, Unit)
    y_hat_loc = popvec(y_hat_last[..., 1:])  # shape: (Batch,), tensor

    y_loc_last = y_loc[-1] if isinstance(y_loc, torch.Tensor) else torch.tensor(y_loc[-1], device=device)
    unique_y_loc = torch.unique(y_loc_last)

    y_direction = torch.full_like(y_hat_loc, fill_value=float('nan'))

    for loc in unique_y_loc:
        # Direct match within 0.2π
        mask1 = torch.abs(y_hat_loc - loc) <= 0.2 * torch.pi
        # Wraparound case near 0 and 2π
        mask2 = torch.abs(torch.abs(y_hat_loc - loc) - 2 * torch.pi) <= 0.2 * torch.pi
        y_direction = torch.where(mask1 | mask2, loc, y_direction)

    return y_direction  # torch tensor

def get_perf(y_hat, y_loc):
    """Get performance with GPU support.

    Args:
        y_hat: Actual output. Tensor (Time, Batch, Unit)
        y_loc: Target output location (-1 for fixation). Tensor (Time, Batch)

    Returns:
        perf: Scalar tensor on CPU
    """
    # Ensure tensors are on same device
    device = y_hat.device
    y_loc = y_loc.to(device)
    
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])  # Assume popvec is GPU-compatible

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    # Calculate angular distance
    original_dist = y_loc - y_hat_loc
    dist = torch.minimum(torch.abs(original_dist), 
                        2*torch.pi - torch.abs(original_dist))
    corr_loc = dist < 0.1*torch.pi

    # Should fixate? (False for HD and color target HD tasks)
    should_fix = y_loc < 0 

    # Convert boolean tensors to float for arithmetic operations
    should_fix_float = should_fix.float()
    fixating_float = fixating.float()
    corr_loc_float = corr_loc.float()

    # Calculate performance
    perf = torch.sum((1-should_fix_float) * corr_loc_float * (1-fixating_float)) / len(corr_loc)
    
    return perf  # Return as CPU scalar for compatibility
