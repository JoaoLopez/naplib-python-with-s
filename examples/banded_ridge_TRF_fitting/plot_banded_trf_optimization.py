"""
===================================================
Banded Ridge: Envelope vs. Acoustic Peak Rate
===================================================

This example demonstrates how to fit a BandedTRF model to neural data using
correlated acoustic features. We compare a broadband speech envelope with
discrete acoustic "peak rate" events.

Specifically, we examine:
1. Iterative Alpha Optimization (Alpha Paths) with peak detection.
2. Incremental predictive power (Delta R).
3. Model stability across different feature fitting orders.
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

data = nl.io.load_speech_task_data()

# Preprocess responses
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# Step A: Compute high-res auditory spectrogram
spec_fs, feat_fs = 11025, 100
data['spec'] = [nl.features.auditory_spectrogram(trl['sound'], spec_fs) for trl in data]
data['spec'] = [resample(trial['spec'], trial['resp'].shape[0]) for trial in data] 

# Step B: Compute Envelope and Peak Rate
data['env_raw'] = [np.sum(trl['spec'], axis=1) for trl in data]
data['pk_raw'] = [nl.features.peak_rate(trl['spec'], feat_fs, band=[1, 10]) for trl in data]

# Step C: Final alignment
data['env'] = [resample(e, r.shape[0]) for e, r in zip(data['env_raw'], data['resp'])]
data['peak_rate'] = [resample(p, r.shape[0]) for p, r in zip(data['pk_raw'], data['resp'])]

###############################################################################
# 2. Fit the BandedTRF (Order 1: Env -> Peak Rate)
# ------------------------------------------------

tmin, tmax, sfreq = -0.1, 0.6, 100
alphas = np.logspace(-2, 5, 15) 
order_1 = ['env', 'peak_rate']

model1 = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
model1.fit(data=data[:-1], feature_order=order_1, target='resp')

# Evaluate Order 1
r_mat_full1 = nl.stats.pairwise_correlation(data[-1]['resp'], model1.predict(data[-1:])[0])
r_full_1 = np.diag(r_mat_full1)

r_mat_env_only = nl.stats.pairwise_correlation(data[-1]['resp'], model1.predict(data[-1:], feature_names=['env'])[0])
r_env_only = np.diag(r_mat_env_only)
dr_peak_rate = r_full_1 - r_env_only

###############################################################################
# 3. Fit the BandedTRF (Order 2: Peak Rate -> Env)
# ------------------------------------------------

order_2 = ['peak_rate', 'env']
model2 = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
model2.fit(data=data[:-1], feature_order=order_2, target='resp')

# Evaluate Order 2
r_mat_full2 = nl.stats.pairwise_correlation(data[-1]['resp'], model2.predict(data[-1:])[0])
r_full_2 = np.diag(r_mat_full2)

r_mat_pk_only = nl.stats.pairwise_correlation(data[-1]['resp'], model2.predict(data[-1:], feature_names=['peak_rate'])[0])
r_pk_only = np.diag(r_mat_pk_only)
dr_env = r_full_2 - r_pk_only

###############################################################################
# 4. Visualization: Alpha Paths with Peak Markers
# -----------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Alpha Paths for Order 1
colors = {'env': '#1f77b4', 'peak_rate': '#d62728'}
for feat in order_1:
    path = model1.alpha_paths_[feat]
    best_alpha = model1.feature_alphas_[feat]
    
    # Plot the full line
    axes[0].semilogx(alphas, path, marker='o', label=feat, color=colors[feat], alpha=0.6)
    
    # Highlight the peak
    peak_val = np.max(path)
    axes[0].plot(best_alpha, peak_val, 'r*', markersize=15, 
                 markeredgecolor='k', label=f'Best {feat}')

axes[0].set_title('Optimization Paths (Order: Env → Peak Rate)')
axes[0].set_xlabel('Alpha ($\lambda$)')
axes[0].set_ylabel('Mean Cross-Validated $r$')
axes[0].legend()

# Delta R comparison
labels = ['Peak Rate Gain\n(After Env)', 'Envelope Gain\n(After Peak Rate)']
dr_values = [np.mean(dr_peak_rate), np.mean(dr_env)]
axes[1].bar(labels, dr_values, color=[colors['peak_rate'], colors['env']])
axes[1].set_title('Incremental Predictive Power ($\Delta R$)')
axes[1].set_ylabel('Mean Gain in Pearson $r$')

plt.tight_layout()
plt.show()

###############################################################################
# 5. Consistency and Kernel Visualization
# ---------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatterplot of full model r
ax1.scatter(r_full_1, r_full_2, alpha=0.6, edgecolors='w')
max_r = max(r_full_1.max(), r_full_2.max())
ax1.plot([0, max_r], [0, max_r], 'k--', alpha=0.5, label='Unity')
ax1.set_title('Total Prediction Consistency ($r_{full}$)')
ax1.set_xlabel('Order 1: Env → Peak Rate')
ax1.set_ylabel('Order 2: Peak Rate → Env')
ax1.legend()

# Kernel comparison for best channel
elec = np.argmax(r_full_1)
lags = np.linspace(tmin, tmax, model1.coef_.shape[-1])
ax2.plot(lags, model1.coef_[elec, 0, :], label='Envelope TRF', lw=2.5, color=colors['env'])
ax2.plot(lags, model1.coef_[elec, 1, :], label='Peak Rate TRF', lw=2.5, color=colors['peak_rate'])
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.set_title(f'TRF Kernels (Electrode {elec})')
ax2.set_xlabel('Time (s)')
ax2.legend()

plt.tight_layout()
plt.show()

###############################################################################
# 6. Summary Table
# ----------------
res_df = pd.DataFrame({
    'Order 1': [np.mean(r_full_1), np.mean(dr_peak_rate)],
    'Order 2': [np.mean(r_full_2), np.mean(dr_env)]
}, index=['Mean Full R', 'Mean Delta R'])

print("\n--- Model Performance Summary ---")
print(res_df)