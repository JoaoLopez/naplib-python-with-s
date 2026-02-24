"""
=========================================================
TRF Comparison: Iterative RidgeCV vs. Banded Regularization
=========================================================

This example compares two approaches for encoding models with multiple 
stimulus features:
1. **Iterative Standard TRF**: Adds features sequentially, optimizing a 
   single global regularization parameter (alpha) via 5-fold cross-validation 
   using ``sklearn.linear_model.RidgeCV``.
2. **Banded TRF**: Adds features sequentially, but optimizes a unique 
   alpha for each feature band.

The comparison focuses on three key metrics:
- **Total Correlation**: Final predictive accuracy with all features.
- **Delta R**: The marginal improvement in correlation as each feature is 
  added to the model.
- **Noise Robustness**: The ability of the model to ignore a "Null" noise band 
  injected between meaningful features.

The script uses synthetic neural responses driven by a speech envelope and 
onset peak rate, with Gaussian noise injected as a distractor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.stats import zscore
import naplib as nl
from naplib.encoding import TRF, BandedTRF
from sklearn.linear_model import Ridge

###############################################################################
# 1. Prepare Synthetic Data (Keeping your extraction logic)
###############################################################################
data = nl.io.load_speech_task_data()
n_trials = 5
data = data[:n_trials]
feat_fs = 100

data['aud_spec'] = [resample(nl.features.auditory_spectrogram(trl['sound'], 11025), trl['resp'].shape[0], axis=0) for trl in data]
data['env'] = [zscore(np.sum(trl['aud_spec'], axis=1)) for trl in data]
data['peak_rate'] = [nl.features.peak_rate(trl['aud_spec'], feat_fs) for trl in data]

np.random.seed(42)
for i in range(len(data)):
    noise = np.random.randn(data[i]['resp'].shape[0])
    data[i]['noise'] = (noise / np.std(noise)) * np.std(data[i]['env'])

tmin, tmax, sfreq = -0.1, 0.5, 100
feature_list = ['env', 'noise', 'peak_rate']
alphas = np.logspace(-2, 7, 10)

###############################################################################
# 2. Fit Standard TRF (Iterative RidgeCV for direct comparison)
###############################################################################
print("Fitting Iterative Standard TRF (RidgeCV)...")
standard_total_r = []
standard_delta_r = []
prev_r = 0

for i in range(len(feature_list)):
    current_feats = feature_list[:i+1]
    
    # 1. Prepare feature matrices for each trial
    all_X = []
    for trl in data:
        curr_X = [trl[ft][:, np.newaxis] if trl[ft].ndim == 1 else trl[ft] for ft in current_feats]
        all_X.append(np.concatenate(curr_X, axis=1))
    
    y = data['resp']
    
    best_alpha_r = -np.inf
    best_alpha_total_r = 0

    # 2. Sweep over alpha values
    for alpha in alphas:
        # Fit a model for EVERY trial individually
        trial_models = []
        for t_idx in range(n_trials):
            m = TRF(tmin, tmax, sfreq, estimator=Ridge(alpha=1.0))
            # Fitting on a single trial (list of 1 trial)
            m.fit(X=[all_X[t_idx]/alpha], y=[y[t_idx]])
            trial_models.append(m)
        
        # 3. Perform LOTO Prediction: 
        # For each trial, predict using the average of all OTHER trial models
        loto_trial_rs = []
        for t_idx in range(n_trials):
            # Get indices for all trials except current one
            other_indices = [idx for idx in range(n_trials) if idx != t_idx]
            
            # Average the coefficients and intercepts
            avg_coef = np.mean([
                np.stack([mmdl.coef_ for mmdl in trial_models[idx].models_], axis=1)
                 for idx in other_indices], axis=0)

            # 2. Prepare the delayed X matrix for the held-out trial
            # _delay_time_series produces shape (n_samples, n_features * n_delays)
            from mne.decoding.receptive_field import _delay_time_series
            x_delayed = _delay_time_series(all_X[t_idx], tmin, tmax, sfreq, fill_mean=False)
            x_delayed = x_delayed.reshape(x_delayed.shape[0], -1)
            
            # 3. Manually compute the matrix product: Y_hat = XW + b
            # x_delayed: (samples, feats*lags), avg_coef.T: (feats*lags, targets)
            y_hat = x_delayed @ avg_coef
            
            # 4. Compute correlation with ground truth
            # nl.stats.pairwise_correlation computes r for each target channel
            r = nl.stats.pairwise_correlation(y[t_idx], y_hat)
            loto_trial_rs.append(np.mean(r))

        # Average R across all LOTO folds for this alpha
        avg_alpha_r = np.mean(loto_trial_rs)
        
        if avg_alpha_r > best_alpha_r:
            best_alpha_r = avg_alpha_r
            # Store the final model (averaged across all trials) for kernel plotting
            final_best_model = avg_coef 
            
    # 4. Record results for this feature set
    standard_total_r.append(best_alpha_r)
    standard_delta_r.append(best_alpha_r - prev_r)
    prev_r = best_alpha_r

print(f"Final Standard LOTO Total R: {standard_total_r[-1]:.4f}")

###############################################################################
# 3. Fit Banded TRF (Sequential Band Optimization)
###############################################################################
print("Fitting Banded TRF...")
banded_model = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
banded_model.fit(data=data, feature_order=feature_list, target='resp')

# For summary metrics on the test set specifically:
df_summary = banded_model.summary()

###############################################################################
# 4. Comprehensive Comparison Plots
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comparison A: Cumulative R
banded_cumulative_r = [banded_model.scores_[:,:,:i+1].mean() for i in range(len(feature_list))]
axes[0].plot(feature_list, standard_total_r, 'o--', label='Standard (RidgeCV)', color='#7f7f7f', markersize=8)
axes[0].plot(feature_list, banded_cumulative_r, 'D-', label='Banded TRF', color='#1f77b4', markersize=8)
axes[0].set_title(r'Cumulative Predictive Accuracy ($R$)', fontweight='bold')
axes[0].set_ylabel('Mean Pearson Correlation')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Comparison B: Delta R (Unique Variance)
x = np.arange(len(feature_list))
width = 0.35
axes[1].bar(x - width/2, np.abs(standard_delta_r), width, label=r'Standard Delta $R$', color='#aaaaaa')
axes[1].bar(x + width/2, np.abs(df_summary['Delta R']), width, label=r'Banded Delta $R$', color='#d62728')
axes[1].set_xticks(x)
axes[1].set_xticklabels(feature_list)
axes[1].set_title('Marginal Improvement (Delta $R$)', fontweight='bold')
axes[1].set_ylabel(r'$\Delta R$ Improvement')
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
plt.show()

###############################################################################
# 5. Kernel Comparison: Standard vs. Banded
###############################################################################
best_ch = 0
lags = np.linspace(tmin, tmax, banded_model._ndelays)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Plot Standard TRF Kernels (from the final model containing all features)
# Standard TRF.coef_ is usually (n_targets, n_features_total, n_delays)
# Note: we must slice the n_features_total to match our bands
std_coef = final_best_model[:,best_ch].reshape(len(feature_list), len(lags))

# Plot Banded TRF Kernels
# BandedTRF.coef_ is (n_targets, n_bands, n_delays, n_trials)
banded_coef = banded_model.coef_[best_ch].mean(axis=-1)

colors = ['#1f77b4', '#7f7f7f', '#d62728'] # Env (Blue), Noise (Gray), Peak (Red)

for i, feat in enumerate(feature_list):
    # Standard Model Plot
    # Standard TRF has all features concatenated; we need to extract indices
    # This logic assumes simple features; if using basis functions, indices change.
    axes[0].plot(lags, std_coef[i, :], label=f'Std: {feat}', color=colors[i], alpha=0.8)
    
    # Banded Model Plot
    axes[1].plot(lags, banded_coef[i, :], label=f'Banded: {feat}', color=colors[i], lw=2)



axes[0].set_title(f'Standard TRF Kernels (Global $\\alpha$)\nChannel {best_ch}')
axes[1].set_title(f'Banded TRF Kernels (Independent $\\alpha$)\nChannel {best_ch}')

for ax in axes:
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel('Lag (s)')
    ax.legend(fontsize='small', frameon=False)

axes[0].set_ylabel('Weights (a.u.)')
plt.tight_layout()
plt.show()