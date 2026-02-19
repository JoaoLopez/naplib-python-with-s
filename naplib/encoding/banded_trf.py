import numpy as np
import pandas as pd
from scipy import signal as sig
from sklearn.linear_model import Ridge

def prepare_feature_matrix(trial_data, feature_list, basis_dict, feature_alphas):
    """
    Concatenates multiple feature tracks into a single matrix, applying 
    non-linear bases and alpha-scaling.
    
    Parameters
    ----------
    trial_data : dict
        A single trial dictionary containing feature arrays.
    feature_list : list of str
        The features to include in the matrix.
    basis_dict : dict
        Mapping of feature names to basis functions (e.g., splines).
    feature_alphas : dict
        Mapping of feature names to their optimized ridge regularization values.
        Features are scaled by 1/alpha for banded ridge regression.

    Returns
    -------
    X : ndarray, shape (time, features)
        The design matrix for the current trial.
    """
    mats = []
    for ft in feature_list:
        # Ensure 2D (time x feature_dims)
        x = np.atleast_2d(trial_data[ft].T).T
        
        # Apply Spline/Basis expansion if defined in analyze_features
        if ft in basis_dict:
            x = apply_bases(x, basis_dict[ft])
        
        # Scale by optimized alpha (default to 1.0 if not yet optimized)
        alpha = feature_alphas.get(ft, 1.0) 
        mats.append(x / alpha)
        
    return np.concatenate(mats, axis=1) if mats else None



def banded_ridge_iteration(data, current_feat, prev_feats, alphas, info, basis_dict, feature_alphas):
    """
    Execute a single iteration of the banded ridge regression pipeline. 
    Determines the optimal alpha for a new feature given a set of 
    previously optimized "background" features.

    Parameters
    ----------
    data : list of dict
        The naplib data object containing 'eeg' and feature tracks.
    current_feat : str
        The name of the feature currently being optimized.
    prev_feats : list of str
        Features that have already been optimized in previous iterations.
    alphas : ndarray
        Array of alpha values to sweep over for cross-validation.
    info : dict
        Metadata containing 'fs', 'tmin', and 'tmax' for time-lagging.
    basis_dict : dict
        Basis functions for spline expansion.
    feature_alphas : dict
        Optimized alphas for `prev_feats`.

    Returns
    -------
    coef_dict : dict
        Nested dictionary of coefficients: `coef_dict[trial_idx][alpha_val]`.
    corrs : ndarray, shape (trials, alphas, channels)
        Correlation coefficients for every trial, alpha, and EEG channel.
    """
    num_trials = len(data)
    num_ch = data[0]['eeg'].shape[1]
    fs, tmin, tmax = info['fs'], info['tmin'], info['tmax']
    
    coef_dict = {trl: {} for trl in range(num_trials)}
    
    # Training: Fit Ridge for every trial and every alpha
    for trl in range(num_trials):
        x_base = prepare_feature_matrix(data[trl], prev_feats, basis_dict, feature_alphas)
        y = data[trl]['eeg']
        
        for alpha in alphas:
            x_new = prepare_feature_matrix(data[trl], [current_feat], basis_dict, {current_feat: alpha})
            x_total = np.concatenate([x_base, x_new], axis=1) if x_base is not None else x_new
            
            # Expand to Toeplitz matrix for TRF estimation
            x_lag = time_lag(x_total, tmin, tmax, fs).reshape(x_total.shape[0], -1)
            x_lag = np.nan_to_num(x_lag)
            
            mdl = Ridge(alpha=1, solver='cholesky')
            mdl.fit(x_lag, y)
            coef_dict[trl][alpha] = mdl.coef_
            
    # Validation: Leave-one-trial-out prediction
    corrs = np.zeros((num_trials, len(alphas), num_ch))
    for trl in range(num_trials):
        y_test = data[trl]['eeg']
        x_base = prepare_feature_matrix(data[trl], prev_feats, basis_dict, feature_alphas)
        
        for a_idx, alpha in enumerate(alphas):
            x_new = prepare_feature_matrix(data[trl], [current_feat], basis_dict, {current_feat: alpha})
            x_test = np.concatenate([x_base, x_new], axis=1) if x_base is not None else x_new
            x_test_lag = np.nan_to_num(time_lag(x_test, tmin, tmax, fs).reshape(x_test.shape[0], -1))
            
            # Average coefficients from training trials (LOO)
            avg_coef = np.mean([coef_dict[t][alpha] for t in range(num_trials) if t != trl], axis=0)
            pred_y = x_test_lag @ avg_coef.T
            
            # Compute correlation per channel
            for ch in range(num_ch):
                # Assumes pairwise_correlation returns (r, p)
                corrs[trl, a_idx, ch] = pairwise_correlation(y_test[:, ch], pred_y[:, ch])[0]
                
    return coef_dict, corrs