# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 14:37:16 2025
FOR functions used in CP value calculation
@author: 13902
"""
        
import numpy as np
from scipy import stats
from scipy.integrate import trapz

def rocN(x, y, N=100, permuteN=0):
    """
    Compute area under ROC given distributions x and y
    Uses N points to construct the ROC
    
    Parameters:
    x : array-like - positive class samples
    y : array-like - negative class samples
    N : int - number of points to use for ROC construction
    permuteN : int - number of permutations for p-value calculation
    
    Returns:
    auROC : float - area under ROC curve
    bestz : float - best threshold (if requested)
    perm : dict - permutation test results (pValue, std, auROCPerm)
    """
    
    # Convert inputs to numpy arrays and flatten
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Check for NaNs or empty arrays
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or len(x) == 0 or len(y) == 0:
        perm = {
            'pValue': np.nan,
            'std': np.nan,
            'auROCPerm': np.nan
        }
        return np.nan, np.nan, perm
    
    # Sort all unique values from both distributions with -inf and inf
    z = np.sort(np.concatenate([[-np.inf], x, y, [np.inf]]))
    
    # Adjust N if there are fewer unique values than requested points
    if len(z) < N:
        N = len(z)
    else:
        # Redistribute thresholds between min and max (excluding -inf and inf)
        z = np.concatenate([[-np.inf], 
                           np.linspace(z[1], z[-2], N-2), 
                           [np.inf]])
    
    # Initialize false alarm and hit rates
    fa = np.zeros(N)
    hit = np.zeros(N)
    
    # Calculate hit and false alarm rates for each threshold
    for i in range(N):
        idx = N - i - 1  # Python uses 0-based indexing
        fa[idx] = np.sum(y > z[i])
        hit[idx] = np.sum(x > z[i])
    
    # Normalize by number of samples
    fa = fa / len(y)
    hit = hit / len(x)
    
    # Calculate AUC using trapezoidal integration
    auROC = trapz(hit, fa)
    
    # Permutation test
    perm = {
        'pValue': np.nan,
        'std': np.nan,
        'auROCPerm': np.nan
    }
    
    if permuteN > 0:
        combined = np.concatenate([x, y])
        total_length = len(combined)
        x_length = len(x)
        
        # Generate random permutations
        randPermXY = np.zeros((total_length, permuteN))
        for n in range(permuteN):
            randPermXY[:, n] = np.random.permutation(combined)
        
        randPermX = randPermXY[:x_length, :]
        randPermY = randPermXY[x_length:, :]
        
        # Calculate ROC for permutations
        faPerm = np.zeros((N, permuteN))
        hitPerm = np.zeros((N, permuteN))
        
        for i in range(N):
            idx = N - i - 1
            faPerm[idx, :] = np.sum(randPermY > z[i], axis=0)
            hitPerm[idx, :] = np.sum(randPermX > z[i], axis=0)
        
        faPerm = faPerm / len(y)
        hitPerm = hitPerm / len(x)
        
        # Calculate AUC for each permutation
        perm_aucs = np.zeros(permuteN)
        for n in range(permuteN):
            perm_aucs[n] = trapz(hitPerm[:, n], faPerm[:, n])
        
        # Calculate p-value (two-tailed)
        p_value = np.sum(perm_aucs > auROC) / permuteN
        if auROC < 0.5:
            p_value = 1 - p_value
        p_value *= 2  # Two-tailed
        
        perm = {
            'pValue': p_value,
            'std': np.std(perm_aucs),
            'auROCPerm': perm_aucs
        }
    
    # Find best threshold (z that maximizes hit - fa)
    bestz = np.nan
    if len(hit) > 0 and len(fa) > 0:
        correct_rate = hit - fa
        max_idx = np.argmax(correct_rate)
        if correct_rate[max_idx] > 0.1:
            bestz = z[max_idx]
    
    return auROC, bestz, perm


# Example usage:
if __name__ == "__main__":
    # Generate some test data
    np.random.seed(42)
    x = np.random.normal(1, 1, 100)  # Positive class
    y = np.random.normal(0, 1, 100)  # Negative class
    
    # Calculate ROC
    auc, best_thresh, perm_results = rocN(x, y, N=100, permuteN=1000)
    
    print(f"AUC: {auc:.3f}")
    print(f"Best threshold: {best_thresh:.3f}")
    print(f"p-value: {perm_results['pValue']:.4f}")