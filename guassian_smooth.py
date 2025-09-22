# -*- coding: utf-8 -*-
"""
Created on Fri May  9 20:53:19 2025
guassian smooth neural psth
@author: NaN
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
def guass_smooth(align_markers,spike_aligned):
    # Parameters
    bin_size_rate = 40  # in ms
    step_size_rate = 40  # in ms
    smooth_factor = 80   # in ms (sigma for Gaussian filter)
    p_critical = 0.01
    spike_time_win = 20 # in ms

    
    
    # Calculate time centers for bins
    start_time = align_markers[0] + bin_size_rate / 2
    end_time = align_markers[1] - bin_size_rate / 24
    t_centers = np.arange(start_time, end_time + step_size_rate, step_size_rate)

    
    # Initialize spike histogram array
    n_trials = spike_aligned.shape[1]
    n_bins = len(t_centers)
    spike_hist = np.zeros((n_bins,n_trials))
    
    for k in range(n_bins):
        # Calculate window boundaries
        win_beg = int(((k) * step_size_rate) / spike_time_win)
        win_end = int(((k) * step_size_rate + bin_size_rate) / spike_time_win)
        
        # Count spikes in window and convert to Hz
        spike_counts = np.sum(spike_aligned[win_beg:win_end+1,:], axis=0)
        spike_hist[k,:] = spike_counts / (bin_size_rate/spike_time_win)  # Convert to Hz
    
    
    # Apply Gaussian smoothing (sigma in bins = smooth_factor/step_size_rate)
    sigma_bins = smooth_factor / step_size_rate
    spike_hist_smoothed = gaussian_filter1d(spike_hist, sigma=sigma_bins, axis=1)

    
    return t_centers, step_size_rate, spike_hist_smoothed