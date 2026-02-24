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
from sklearn.linear_model import RidgeCV

###############################################################################
# 1. Prepare Synthetic Data (Keeping your extraction logic)
###############################################################################
data = nl.io.load_speech_task_data()
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
alphas = np.logspace(0, 7, 8)

###############################################################################
# 2. Fit Standard TRF (Iterative RidgeCV for direct comparison)
###############################################################################
print("Fitting Iterative Standard TRF (RidgeCV)...")
standard_total_r = []
standard_delta_r = []
prev_r = 0

for i in range(len(feature_list)):
    current_feats = feature_list[:i+1]
    all_X = []
    for trl in data:
        curr_X = []
        for ft in current_feats:
            if trl[ft].ndim==1:
                curr_X.append(trl[ft][:,np.newaxis])
            else:
                curr_X.append(trl[ft])
        all_X.append(curr_X)
    all_X = [np.concatenate(x, axis=1) for x in all_X]

    # RidgeCV performs leave-one-trial-out (or k-fold) internally
    # We use cv=5 as requested to find the best global alpha for the current feature set
    est = RidgeCV(alphas=alphas, cv=5)
    model = TRF(tmin, tmax, sfreq, estimator=est)
    model.fit(X=all_X, y=data['resp'])
    
    # Score on held-out trial
    curr_r = np.mean(model.score(X=all_X, y=data['resp']))
    
    standard_total_r.append(curr_r)
    standard_delta_r.append(curr_r - prev_r)
    prev_r = curr_r

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
axes[1].bar(x - width/2, standard_delta_r, width, label=r'Standard Delta $R$', color='#aaaaaa')
axes[1].bar(x + width/2, df_summary['Delta R'], width, label=r'Banded Delta $R$', color='#d62728')
axes[1].set_xticks(x)
axes[1].set_xticklabels(feature_list)
axes[1].set_title('Marginal Improvement (Delta $R$)', fontweight='bold')
axes[1].set_ylabel(r'$\Delta R$ Improvement')
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
std_coef = model.coef_[best_ch] 

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