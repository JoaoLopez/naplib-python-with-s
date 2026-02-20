r"""
===================================================
Banded Ridge: Robustness Check with Null Bands
===================================================

This example provides a rigorous sanity check for BandedTRF. We insert a 
"Null Band" (random Gaussian noise) between our meaningful features to 
ensure the model correctly regularizes irrelevant information.

Robustness Checks included:
1. Stimulus Alignment Visualization.
2. Step-wise Marginal Delta R optimization paths.
3. Order-invariance consistency (Scatter of Order 1 vs Order 2).
4. Kernel weight inspection for noise suppression.
5. Statistical significance via the .summary() method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.stats import zscore
import naplib as nl
from naplib.encoding import BandedTRF

###############################################################################
# 1. Prepare the Data
# -------------------

data = nl.io.load_speech_task_data()
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# Step A: Compute auditory spectrogram and align to modeling rate (100Hz)
spec_fs, feat_fs = 11025, 100
data['spec'] = [nl.features.auditory_spectrogram(trl['sound'], spec_fs) for trl in data]
data['spec'] = [resample(trial['spec'], trial['resp'].shape[0]) for trial in data] 

# Step B: Compute Envelope and Peak Rate
data['env'] = [zscore(np.sum(trl['spec'], axis=1)) for trl in data]
data['peak_rate'] = [nl.features.peak_rate(trl['spec'], feat_fs, band=[1, 10]) for trl in data]

# Step C: Final alignment and "Null" Noise Injection
for i, trial in enumerate(data):    
    # Null Band: Gaussian noise scaled to match envelope variance
    noise = np.random.randn(trial['resp'].shape[0])
    data[i]['noise'] = (noise / np.std(noise)) * np.std(data[i]['env'])

###############################################################################
# 2. Visualize Stimulus Features
# ------------------------------

fig, ax = plt.subplots(figsize=(12, 3))
t = np.arange(500) / feat_fs
ax.plot(t, data[0]['env'][:500], label='Envelope', color='#1f77b4')
ax.plot(t, data[0]['peak_rate'][:500], label='Peak Rate', color='#d62728')
ax.plot(t, data[0]['noise'][:500], label='Noise (Null)', color='#7f7f7f', alpha=0.5)
ax.set_title('Stimulus Features (First 5 Seconds)')
ax.set_xlabel('Time (s)')
ax.legend(loc='upper right', fontsize='small', ncol=3)
plt.show()

###############################################################################
# 3. Fit Models with Injected Noise (Order Dependency)
# ----------------------------------------------------

tmin, tmax, sfreq = -0.2, 0.7, 100
alphas = np.logspace(-2, 8, 6) 

# We fit two models with the noise band in different positions
order_1 = ['env', 'noise', 'peak_rate']
model1 = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
model1.fit(data=data[:-1], feature_order=order_1, target='resp')

order_2 = ['peak_rate', 'noise', 'env']
model2 = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)
model2.fit(data=data[:-1], feature_order=order_2, target='resp')

###############################################################################
# 4. Alpha Optimization Paths (Marginal Delta R)
# ----------------------------------------------

colors = {'env': '#1f77b4', 'noise': '#7f7f7f', 'peak_rate': '#d62728'}
n_bands = len(order_1)

for b_idx in range(n_bands):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    for i, (mdl, ord_list) in enumerate(zip([model1, model2], [order_1, order_2])):
        feat = ord_list[b_idx]
        path = mdl.alpha_paths_[feat]
        # Calculate Delta R Path relative to the max R of the previous band
        prev_r = 0 if b_idx == 0 else np.max(mdl.alpha_paths_[ord_list[b_idx-1]])
        delta_path = path - prev_r
        
        best_alpha = mdl.feature_alphas_[feat]
        peak_delta = np.max(delta_path)
        
        axes[i].semilogx(alphas, delta_path, marker='o', color=colors[feat], label=f'Path: {feat}')
        axes[i].plot(best_alpha, peak_delta, '*', markersize=14, markeredgecolor='k', label=f'Best Alpha')
        axes[i].set_title(f'Step {b_idx+1}: {feat} Optimization')
        axes[i].set_xlabel(r'Alpha ($\lambda$)')
        axes[i].legend()

    axes[0].set_ylabel(r'Marginal $\Delta R$')
    plt.tight_layout()
    plt.show()

###############################################################################
# 5. Global Consistency: Order 1 vs Order 2
# -----------------------------------------

# Evaluate both models on a held-out trial for all channels
test_trl = data[-1:]

r_full_1 = nl.stats.pairwise_correlation(test_trl[0]['resp'], model1.predict(test_trl)[0])
r_full_2 = nl.stats.pairwise_correlation(test_trl[0]['resp'], model2.predict(test_trl)[0])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(r_full_1, r_full_2, alpha=0.6, edgecolors='w', color='purple')
lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Unity (Perfect Consistency)')
ax.set_title('Global Consistency Check')
ax.set_xlabel('Predictive Accuracy $r$ (Order 1)')
ax.set_ylabel('Predictive Accuracy $r$ (Order 2)')
ax.legend()
plt.show()

###############################################################################
# 6. Final Model Kernels for the Best Channel
# -------------------------------------------

best_ch = np.argmax(r_full_2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
lags = np.linspace(tmin, tmax, model1._ndelays)

for i, (mdl, ord_list, title) in enumerate(zip([model1, model2], 
                                               [order_1, order_2], 
                                               ['Kernels (Order 1)', 'Kernels (Order 2)'])):
    for f_idx, feat in enumerate(ord_list):
        axes[i].plot(lags, mdl.coef_[best_ch, f_idx, :], 
                     label=feat, color=colors[feat], lw=2 if feat != 'noise' else 1,
                     linestyle='-' if feat != 'noise' else '--')
    
    axes[i].axhline(0, color='black', alpha=0.3)
    axes[i].set_title(f"{title} - Ch {best_ch}")
    axes[i].set_xlabel('Lag (s)')
    axes[i].legend(fontsize='small')

axes[0].set_ylabel('Weight (a.u.)')
plt.tight_layout()
plt.show()


# Final Summary Table for the Best Channel
print(f"\nFinal Statistics for Model 1, Electrode {best_ch}:")
model1.summary(test_trl, channel=best_ch)

# Final Summary Table for the Best Channel
print(f"\nFinal Statistics for Model 2, Electrode {best_ch}:")
model2.summary(test_trl, channel=best_ch)