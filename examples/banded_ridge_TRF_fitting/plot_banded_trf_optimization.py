r"""
===================================================
Banded Ridge: Robustness Check with Null Bands
===================================================

This example provides a rigorous sanity check for BandedTRF. We insert a 
"Null Band" (random Gaussian noise) between our meaningful features to 
ensure the model correctly regularizes irrelevant information.

Specifically, we examine:
1. Iterative Alpha Optimization (Alpha Paths) with peak markers.
2. Incremental predictive power (Delta R) for each band.
3. Model stability and weight suppression for irrelevant features.
4. Statistical significance of incremental gains across trials.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import naplib as nl
from naplib.encoding import BandedTRF

###############################################################################
# 1. Prepare the Data
# -------------------
# We compute features from a speech task dataset. We define a high-res
# extraction rate (spec_fs) and a target modeling rate (feat_fs).

data = nl.io.load_speech_task_data()

# Preprocess responses
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# Step A: Compute high-res auditory spectrogram
spec_fs, feat_fs = 11025, 100
data['spec'] = [nl.features.auditory_spectrogram(trl['sound'], spec_fs) for trl in data]

# Ensure spectrogram matches response length exactly
data['spec'] = [resample(trial['spec'], trial['resp'].shape[0]) for trial in data] 

# Step B: Compute Envelope and Peak Rate
data['env_raw'] = [np.sum(trl['spec'], axis=1) for trl in data]
data['pk_raw'] = [nl.features.peak_rate(trl['spec'], feat_fs, band=[1, 10]) for trl in data]

# Step C: Final alignment and "Null" Noise Injection
for i, trial in enumerate(data):
    # Standard features
    data[i]['env'] = resample(data[i]['env_raw'], trial['resp'].shape[0])
    data[i]['peak_rate'] = resample(data[i]['pk_raw'], trial['resp'].shape[0])
    
    # Null Band: Gaussian noise with same variance as envelope for "fair" competition
    noise = np.random.randn(trial['resp'].shape[0])
    data[i]['noise'] = (noise / np.std(noise)) * np.std(data[i]['env'])

###############################################################################
# 2. Fit Models with Injected Noise (Order Dependency)
# ----------------------------------------------------

tmin, tmax, sfreq = -0.2, 0.7, 100
alphas = np.logspace(-1, 8, 19) # Wide range to allow noise to be heavily penalized

# Order 1: Env -> Noise -> Peak Rate
order_1 = ['env', 'noise', 'peak_rate']
model1 = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
model1.fit(data=data[:-1], feature_order=order_1, target='resp')

# Order 2: Peak Rate -> Noise -> Env
order_2 = ['peak_rate', 'noise', 'env']
model2 = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
model2.fit(data=data[:-1], feature_order=order_2, target='resp')

###############################################################################
# 3. Analyze Delta R (Incremental Improvement)
# --------------------------------------------

def get_incremental_r(model, test_data, order):
    r_steps = []
    current_feats = []
    for feat in order:
        current_feats.append(feat)
        # Predict using a subset of features
        pred = model.predict(test_data, feature_names=current_feats)[0]
        # Diagonal of pairwise correlation gives per-channel r
        r_step = np.mean(np.diag(nl.stats.pairwise_correlation(test_data[0]['resp'], pred)))
        r_steps.append(r_step)
    
    return np.diff(r_steps, prepend=0)

dr1 = get_incremental_r(model1, data[-1:], order_1)
dr2 = get_incremental_r(model2, data[-1:], order_2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
colors = {'env': '#1f77b4', 'noise': '#7f7f7f', 'peak_rate': '#d62728'}

axes[0].bar(order_1, dr1, color=[colors[f] for f in order_1])
axes[0].set_title('Delta R (Order: Env -> Noise -> PK)')
axes[0].set_ylabel(r'Gain in Pearson $r$')

axes[1].bar(order_2, dr2, color=[colors[f] for f in order_2])
axes[1].set_title('Delta R (Order: PK -> Noise -> Env)')
plt.tight_layout()
plt.show()

###############################################################################
# 4. Alpha Optimization Paths
# ---------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
models = [model1, model2]
orders = [order_1, order_2]
titles = ['Alpha Paths (Order 1)', 'Alpha Paths (Order 2)']

for i, (mdl, ord_list) in enumerate(zip(models, orders)):
    for feat in ord_list:
        path = mdl.alpha_paths_[feat]
        best_alpha = mdl.feature_alphas_[feat]
        
        # Plot path
        axes[i].semilogx(alphas, path, marker='o', label=feat, color=colors[feat], alpha=0.6)
        
        # Mark peak
        peak_val = np.max(path)
        axes[i].plot(best_alpha, peak_val, '*', markersize=14, 
                     markeredgecolor='k', label=f'Best {feat}')

    axes[i].set_title(titles[i])
    axes[i].set_xlabel(r'Alpha ($\lambda$)')
    axes[i].legend(fontsize='small', ncol=2)

axes[0].set_ylabel(r'Cross-Validated Correlation ($r$)')
plt.tight_layout()
plt.show()

###############################################################################
# 5. Consistency and Kernel Visualization
# ---------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatterplot of full model r (Order 1 vs Order 2)
# Evaluate on test set
r_full_1_vec = np.diag(nl.stats.pairwise_correlation(data[-1]['resp'], model1.predict(data[-1:])[0]))
r_full_2_vec = np.diag(nl.stats.pairwise_correlation(data[-1]['resp'], model2.predict(data[-1:])[0]))

ax1.scatter(r_full_1_vec, r_full_2_vec, alpha=0.6, edgecolors='w')
max_r = max(r_full_1_vec.max(), r_full_2_vec.max())
ax1.plot([0, max_r], [0, max_r], 'k--', alpha=0.5, label='Unity')
ax1.set_title(r'Total Prediction Consistency ($r_{full}$)')
ax1.set_xlabel('Order 1: Env -> Noise -> Peak Rate')
ax1.set_ylabel('Order 2: Peak Rate -> Noise -> Env')
ax1.legend()

# Kernel comparison for best channel
elec = np.argmax(r_full_1_vec)
lags = np.linspace(tmin, tmax, model1.coef_.shape[-1])
ax2.plot(lags, model1.coef_[elec, 0, :], label='Envelope', lw=2.5, color=colors['env'])
ax2.plot(lags, model1.coef_[elec, 1, :], label='Noise', lw=2.5, color=colors['noise'], linestyle='--')
ax2.plot(lags, model1.coef_[elec, 2, :], label='Peak Rate', lw=2.5, color=colors['peak_rate'])
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.set_title(f'TRF Kernels (Electrode {elec})')
ax2.set_xlabel('Time (s)')
ax2.legend()

plt.tight_layout()
plt.show()

###############################################################################
# 6. Statistical Summary
# ----------------------
# We find the best channel and show its specific statistical metrics.

best_ch = np.argmax(r_full_1_vec)
print(f"Generating summary for the most responsive electrode (Channel {best_ch})...")

# model1.summary() performs the t-test across trials and channels
best_ch_summary = model1.summary(data[-1:], channel=best_ch)