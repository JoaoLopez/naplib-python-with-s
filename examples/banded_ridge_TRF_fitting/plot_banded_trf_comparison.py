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
from mne.decoding.receptive_field import _delay_time_series

###############################################################################
# 1. Prepare Synthetic Data
# -------------------------
# Load data and extract acoustic features (Envelope and Peak Rate). 
# We inject a "Null" noise band to test how each model handles irrelevant data.

data = nl.io.load_speech_task_data()
n_trials = 3
data = data[:n_trials]
feat_fs = 100

# Preprocess features: Compute spectrogram, resample to match response, and compute metrics
data['aud_spec'] = [resample(nl.features.auditory_spectrogram(trl['sound'], 11025), trl['resp'].shape[0], axis=0) for trl in data]
data['env'] = [zscore(np.sum(trl['aud_spec'], axis=1)) for trl in data]
data['peak_rate'] = [nl.features.peak_rate(trl['aud_spec'], feat_fs) for trl in data]

# Inject Noise Band: Scaled to match the variance of the envelope
np.random.seed(42)
for i in range(len(data)):
    noise = np.random.randn(data[i]['resp'].shape[0])
    data[i]['noise'] = (noise / np.std(noise)) * np.std(data[i]['env'])

tmin, tmax, sfreq = -0.1, 0.4, 100
feature_list = ['env', 'noise', 'peak_rate']
alphas = np.logspace(-2, 5, 15)

###############################################################################
# 2. Fit Standard TRF with Alpha Path Tracking
# --------------------------------------------
# We simulate a "Standard" TRF approach by finding a single optimal alpha for 
# the combined feature matrix.

print("Fitting Standard TRF & Tracking Alpha Path...")
standard_total_r = []
standard_delta_r = []
standard_alpha_paths = [] 
prev_r = 0

for i in range(len(feature_list)):
    current_feats = feature_list[:i+1]
    all_X = []
    for trl in data:
        curr_X = [trl[ft][:, np.newaxis] if trl[ft].ndim == 1 else trl[ft] for ft in current_feats]
        curr_X = np.concatenate(curr_X, axis=1)
        # Apply time delays to features
        curr_X = _delay_time_series(curr_X, tmin, tmax, sfreq, fill_mean=False)
        curr_X = curr_X.reshape(curr_X.shape[0], -1)
        all_X.append(curr_X)
    
    y = data['resp']
    path_for_this_set = []
    best_alpha_r = -np.inf

    for alpha in alphas:
        # Leave-one-trial-out (LOTO) cross-validation
        trial_betas = [Ridge(alpha=alpha).fit(tx, ty).coef_ for tx, ty in zip(all_X, y)]
        loto_trial_rs = []
        for t_idx in range(n_trials):
            other_indices = [idx for idx in range(n_trials) if idx != t_idx]
            avg_coef = np.mean([trial_betas[idx] for idx in other_indices], axis=0)

            # Predict on the held-out trial
            y_hat = (all_X[t_idx]/alpha) @ avg_coef.T
            r = nl.stats.pairwise_correlation(y[t_idx], y_hat)
            loto_trial_rs.append(np.mean(r))

        avg_alpha_r = np.mean(loto_trial_rs)
        path_for_this_set.append(avg_alpha_r)
        
        if avg_alpha_r > best_alpha_r:
            best_alpha_r = avg_alpha_r
            final_best_model = avg_coef 
            
    standard_alpha_paths.append(path_for_this_set)
    standard_total_r.append(best_alpha_r)
    standard_delta_r.append(best_alpha_r - prev_r)
    prev_r = best_alpha_r

###############################################################################
# 3. Fit Banded TRF
# -----------------
# The BandedTRF optimizes a separate alpha for each feature band sequentially.

print("Fitting Banded TRF...")
banded_model = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
banded_model.fit(data=data, feature_order=feature_list, target='resp')

df_summary = banded_model.summary()

###############################################################################
# 4a. Comprehensive Comparison Plots
# ----------------------------------
# Visualize how cumulative accuracy grows and how much unique variance (Delta R) 
# each feature contributes.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comparison A: Cumulative Predictive Accuracy
banded_cumulative_r = [banded_model.scores_[:,:,i].mean() for i in range(len(feature_list))]
axes[0].plot(feature_list, standard_total_r, 'o--', label='Standard (RidgeCV)', color='#7f7f7f', markersize=8)
axes[0].plot(feature_list, banded_cumulative_r, 'D-', label='Banded TRF', color='#1f77b4', markersize=8)
axes[0].set_title(r'Cumulative Predictive Accuracy ($R$)', fontweight='bold')
axes[0].set_ylabel('Mean Pearson Correlation')
axes[0].set_xlabel('Feature Set (Cumulative)')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Comparison B: Delta R (Unique Variance)
x = np.arange(len(feature_list))
width = 0.35
axes[1].bar(x - width/2, standard_delta_r, width, label=r'Standard $\Delta R$', color='#aaaaaa')
axes[1].bar(x + width/2, df_summary['Delta R'], width, label=r'Banded $\Delta R$', color='#d62728')
axes[1].set_xticks(x)
axes[1].set_xticklabels(feature_list)
axes[1].set_title('Marginal Improvement ($\Delta R$)', fontweight='bold')
axes[1].set_ylabel(r'Improvement in $R$')
axes[1].set_yscale('symlog', linthresh=1e-4) # Symlog to visualize small noise contributions
axes[1].legend()

plt.tight_layout()
plt.show()

###############################################################################
# 4b. Visualization: Alpha Paths
# ------------------------------
# Contrast the global alpha sweep of standard Ridge with the per-feature 
# optimization paths of Banded Ridge.

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Plot Standard Alpha Path
for path in standard_alpha_paths:
    axes[0].semilogx(alphas, path, 'o-', color='black', alpha=0.3)
    axes[0].plot(alphas[best_idx], path[best_idx], '*',
        markersize=14, markeredgecolor='k')
axes[0].set_title('Standard TRF: Global Alpha Sweep\n(Full Feature Set)')
axes[0].set_xlabel(r'Regularization ($\alpha$)')
axes[0].set_ylabel('Mean Correlation ($r$)')

# Plot Banded Alpha Paths
for i, feat in enumerate(feature_list):
    path = banded_model.alpha_paths_[feat]
    best_idx = np.argmax(path)
    axes[1].semilogx(alphas, path, 'o-', label=f'Band: {feat}')
    axes[1].plot(alphas[best_idx], path[best_idx], '*', markersize=14, markeredgecolor='k')

axes[1].set_title('Banded TRF: Sequential Alpha Sweeps\n(Per-Feature Regularization)')
axes[1].set_xlabel(r'Regularization ($\alpha$)')
axes[1].legend()

plt.tight_layout()
plt.show()

###############################################################################
# 5. Kernel Comparison: Standard vs. Banded
# -----------------------------------------
# Inspect the resulting TRF weights. Banded models typically suppress noise 
# more effectively by assigning it a separate, higher regularization value.

best_ch = 0
lags = np.linspace(tmin, tmax, banded_model._ndelays)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Extract Standard TRF Kernels
std_coef = final_best_model[best_ch, :].reshape(len(feature_list), len(lags))

# Extract Banded TRF Kernels (average across trials)
banded_coef = banded_model.coef_[best_ch].mean(axis=-1)

colors = ['#1f77b4', '#7f7f7f', '#d62728'] # Env (Blue), Noise (Gray), Peak (Red)

for i, feat in enumerate(feature_list):
    # Standard Model Plot
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