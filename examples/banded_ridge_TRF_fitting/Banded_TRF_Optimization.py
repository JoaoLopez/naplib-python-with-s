"""
====================================
Iterative Banded Ridge TRF Modeling
====================================

This example demonstrates how to fit a Banded TRF model to neural data. 
Unlike standard Ridge regression which applies a single penalty to all 
features, Banded Ridge allows different feature sets (bands) to have 
independent regularization.

We use an iterative "greedy" approach:
1. Optimize alpha for Feature A.
2. Fix Feature A, then optimize alpha for Feature B.
3. Observe the incremental improvement (Delta R) in model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import naplib as nl
from naplib.encoding import BandedTRF

###############################################################################
# 1. Prepare the Data
# -------------------
# We load a speech task dataset and prepare two feature bands:
# Band A: High-dimensional auditory spectrogram (reduced to 32 bins)
# Band B: Low-dimensional speech envelope

data = nl.io.load_speech_task_data()

# Preprocess responses
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# Feature Band A: Spectrogram
data['spec'] = [nl.features.auditory_spectrogram(trl['sound'], 11025) for trl in data]
data['spec'] = [resample(trl['spec'], trl['resp'].shape[0]) for trl in data]
data['spec_32'] = nl.array_ops.concat_apply(data['spec'], resample, {'num': 32, 'axis': 1})

# Feature Band B: Envelope
data['env'] = [np.abs(nl.features.hilbert_transform(trl['sound'])) for trl in data]
data['env'] = [resample(trl['env'], trl['resp'].shape[0]) for trl in data]

###############################################################################
# 2. Fit the BandedTRF
# --------------------
# We define our feature order and a range of alpha values to sweep.

tmin, tmax, sfreq = 0, 0.4, 100
alphas = np.logspace(-1, 5, 7)
feature_order = ['spec_32', 'env']

model = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)

# We fit using the first 9 trials and hold out the last trial
model.fit(data=data[:-1], feature_order=feature_order, target='resp')

print(f"Optimized Alphas: {model.feature_alphas_}")

###############################################################################
# 3. Visualize Alpha Paths and Delta R
# ------------------------------------
# We can examine how each feature improved the model and the stability 
# of the regularization sweep.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot Alpha Paths
for feat in feature_order:
    axes[0].semilogx(alphas, model.alpha_paths_[feat], marker='o', label=feat)
axes[0].set_title('Regularization Sweep (Alpha Paths)')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Mean Correlation (r)')
axes[0].legend()

# Compute Delta R on test data
# Correlation with Band A only
pred_a = model.predict(data[-1:], feature_names=['spec_32'])
r_a = np.mean(nl.evaluation.correlation(data[-1]['resp'], pred_a[0]))

# Correlation with Band A + Band B
pred_all = model.predict(data[-1:])
r_all = np.mean(nl.evaluation.correlation(data[-1]['resp'], pred_all[0]))

axes[1].bar(['Spectrogram Only', 'Spectrogram + Envelope'], [r_a, r_all], color=['#1f77b4', '#ff7f0e'])
axes[1].set_title(f'Delta R: {r_all - r_a:.4f}')
axes[1].set_ylabel('Pearson r')

plt.tight_layout()
plt.show()

###############################################################################
# 4. Plot the Resulting TRFs
# --------------------------
# The weights are stored in the .coef_ attribute with shape (channels, features, lags).

elec = 10 # Example electrode/channel
full_coefs = model.coef_

# Slice spectrogram weights (first 32 indices)
spec_weights = full_coefs[elec, :32, :]

# Slice envelope weights (index 32)
env_weights = full_coefs[elec, 32, :]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [3, 1]})

nl.visualization.strf_plot(spec_weights, tmin=tmin, tmax=tmax, ax=ax1)
ax1.set_title(f'Spectral TRF (Elec {elec})')

lags = np.linspace(tmin, tmax, len(env_weights))
ax2.plot(lags, env_weights)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.set_title('Envelope TRF')
ax2.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()