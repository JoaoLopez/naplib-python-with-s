"""
===========================================================
TRF Comparison: Iterative RidgeCV vs. Banded Regularization
===========================================================

This example compares two approaches for encoding models with multiple 
stimulus features:

1. **Iterative Standard TRF**: Adds features sequentially, optimizing a 
   single global regularization parameter (alpha) via cross-validation.
2. **Banded TRF**: Adds features sequentially, but optimizes a unique 
   alpha for each feature band.

The comparison focuses on predictive accuracy ($R$), marginal improvement ($Delta R$),
and the model's ability to ignore irrelevant noise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.stats import zscore, ttest_1samp
import naplib as nl
from naplib.encoding import TRF, BandedTRF
from sklearn.linear_model import Ridge
from mne.decoding.receptive_field import _delay_time_series

###############################################################################
# 1. Prepare Synthetic Data
# -------------------------
# We load speech task data and compute the auditory envelope and peak rate.
# A "noise" feature is added to test regularization robustness.

data = nl.io.load_speech_task_data()
n_trials = 2
data = data[:n_trials]
feat_fs = 100

# Preprocess features
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
# the combined feature matrix using leave-one-trial-out cross-validation.

print("Fitting Standard TRF & Tracking Alpha Path...")
standard_p = []
standard_total_r = []
standard_delta_r = []
standard_alpha_paths = [] 
prev_r = 0
prev_r_all = 0

for i in range(len(feature_list)):
    current_feats = feature_list[:i+1]
    all_X = []
    for trl in data:
        curr_X = [trl[ft][:, np.newaxis] if trl[ft].ndim == 1 else trl[ft] for ft in current_feats]
        curr_X = np.concatenate(curr_X, axis=1)
        curr_X = _delay_time_series(curr_X, tmin, tmax, sfreq, fill_mean=False)
        curr_X = curr_X.reshape(curr_X.shape[0], -1)
        all_X.append(curr_X)
    
    y = data['resp']
    path_for_this_set = []
    best_alpha_r = -np.inf

    for alpha in alphas:
        trial_betas = [Ridge(alpha=alpha).fit(tx, ty).coef_ for tx, ty in zip(all_X, y)]
        loto_trial_rs = []
        for t_idx in range(n_trials):
            other_indices = [idx for idx in range(n_trials) if idx != t_idx]
            avg_coef = np.mean([trial_betas[idx] for idx in other_indices], axis=0)
            y_hat = (all_X[t_idx]/alpha) @ avg_coef.T
            r = nl.stats.pairwise_correlation(y[t_idx], y_hat)
            loto_trial_rs.append(np.mean(r))

        alpha_r = np.array(loto_trial_rs)
        avg_alpha_r = np.mean(loto_trial_rs)
        path_for_this_set.append(avg_alpha_r)
        
        if avg_alpha_r > best_alpha_r:
            best_alpha_r = avg_alpha_r
            best_alpha_r_all = alpha_r
            final_best_model = np.stack(trial_betas, axis=2) 
            _, p_val = ttest_1samp(alpha_r-prev_r_all, 0)
            
    standard_alpha_paths.append(path_for_this_set)
    standard_total_r.append(best_alpha_r)
    standard_delta_r.append(best_alpha_r - prev_r)
    standard_p.append(p_val)
    prev_r = best_alpha_r
    prev_r_all = best_alpha_r_all

###############################################################################
# 3. Fit Banded TRF
# -----------------
# The BandedTRF model allows each feature band to have its own optimal 
# regularization parameter, determined sequentially.

print("Fitting Banded TRF...")
banded_model = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
banded_model.fit(data=data, feature_order=feature_list, target='resp')

df_summary = banded_model.summary()

###############################################################################
# 4a. Comprehensive Comparison Plots & Statistics
# -----------------------------------------------
# Here we compare the cumulative correlation and marginal improvement.

# Print Statistics for Standard Model
print("\n" + "="*30)
print("STANDARD TRF STATISTICS")
print("="*30)
for i, feat in enumerate(feature_list):
    print(f"Feature: {feat:10} | Delta R: {standard_delta_r[i]:.4f} | Significance p: {standard_p[i]:.4f}")

# Print Statistics for Banded Model
print("\n" + "="*30)
print("BANDED TRF STATISTICS")
print("="*30)
print(df_summary)

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
axes[1].set_title(r'Marginal Improvement ($\Delta R$)', fontweight='bold')
axes[1].set_ylabel(r'Improvement in $R$')
axes[1].set_yscale('symlog', linthresh=1e-4)
axes[1].legend()

plt.tight_layout()
plt.show()

###############################################################################
# 4b. Visualization: Alpha Optimization Paths (Standard vs. Banded)
# -----------------------------------------------------------------
# We compare the optimization curves for each feature. For the Standard model,
# the path represents the best $R$ achievable using a global $\alpha$ as 
# features are added. For the Banded model, the path represents the marginal
# improvement ($\Delta R$) gained by optimizing that specific band's alpha.

colors = {'env': '#1f77b4', 'noise': '#7f7f7f', 'peak_rate': '#d62728'}

for b_idx, feat in enumerate(feature_list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    
    # --- Left Plot: Standard TRF (Global Alpha) ---
    # In the standard approach, we look at the R-path for the cumulative set
    std_path = np.array(standard_alpha_paths[b_idx])
    # Calculate marginal improvement for standard model
    prev_std_r = 0 if b_idx == 0 else standard_total_r[b_idx-1]
    std_delta_path = std_path - prev_std_r
    
    best_std_idx = np.argmax(std_delta_path)
    axes[0].semilogx(alphas, std_delta_path, 'o-', color='black', alpha=0.6, label=f'Global $\\alpha$ Path')
    axes[0].plot(alphas[best_std_idx], std_delta_path[best_std_idx], '*', 
                 markersize=14, markeredgecolor='k', label=f'Opt $\\alpha$: {alphas[best_std_idx]:.1e}')
    
    axes[0].set_title(f'Standard TRF - Step {b_idx+1}: {feat}')
    axes[0].set_xlabel(r'Global Regularization ($\alpha$)')
    axes[0].set_ylabel(r'Marginal Improvement ($\Delta R$)')
    axes[0].legend(fontsize='small')
    
    # --- Right Plot: Banded TRF (Independent Alpha) ---
    # In the banded approach, we look at the R-path for the specific feature band
    banded_path = banded_model.alpha_paths_[feat]
    # Calculate marginal improvement relative to previous bands' max R
    prev_banded_r = 0 if b_idx == 0 else np.max(banded_model.alpha_paths_[feature_list[b_idx-1]])
    banded_delta_path = banded_path - prev_banded_r
    
    best_banded_alpha = banded_model.feature_alphas_[feat]
    peak_banded_delta = np.max(banded_delta_path)
    
    axes[1].semilogx(alphas, banded_delta_path, 'o-', color=colors[feat], label=f'Band: {feat}')
    axes[1].plot(best_banded_alpha, peak_banded_delta, '*', 
                 markersize=14, markeredgecolor='k', label=f'Opt $\\alpha$: {best_banded_alpha:.1e}')
    
    axes[1].set_title(f'Banded TRF - Step {b_idx+1}: {feat}')
    axes[1].set_xlabel(r'Band-Specific Regularization ($\alpha$)')
    axes[1].legend(fontsize='small')

    all_deltas = np.concatenate([std_delta_path, banded_delta_path])
    ymax = all_deltas.max()
    ymin = max(all_deltas.min(), -0.005)
    axes[0].set_ylim([ymin, ymax+(ymax-ymin)*0.1])

    plt.tight_layout()
    plt.show()

###############################################################################
# 5. Kernel Comparison: Standard vs. Banded
# -----------------------------------------
# Inspecting the kernels reveals how Banded TRF better suppresses the 
# noise feature by applying an independent regularization penalty.

best_ch = 0
lags = np.linspace(tmin, tmax, banded_model._ndelays)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# Extract Standard TRF Kernels
std_coef = final_best_model[best_ch, :, :].reshape(len(feature_list), len(lags), n_trials)

# Extract Banded TRF Kernels (average across trials)
banded_coef = banded_model.coef_[best_ch]

colors = ['#1f77b4', '#7f7f7f', '#d62728'] 

for i, feat in enumerate(feature_list):
    # Plot TRF with error shading across trials/CV folds
    nl.visualization.shaded_error_plot(
        lags, std_coef[i, :],
        color=colors[i],
        ax=axes[0],
        plt_args={'label': f'Std: {feat}', 'lw': 2}
    )
    nl.visualization.shaded_error_plot(
        lags, banded_coef[i, :],
        color=colors[i],
        ax=axes[1],
        plt_args={'label': f'Banded: {feat}', 'lw': 2}
    )

axes[0].set_title(f'Standard TRF Kernels (Global $\\alpha$)\nChannel {best_ch}')
axes[1].set_title(f'Banded TRF Kernels (Independent $\\alpha$)\nChannel {best_ch}')

for ax in axes:
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel('Lag (s)')
    ax.legend(fontsize='small', frameon=False)

axes[0].set_ylabel('Weights (a.u.)')
plt.tight_layout()
plt.show()