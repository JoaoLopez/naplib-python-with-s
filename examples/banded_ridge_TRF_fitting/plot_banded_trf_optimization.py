"""
===================================================
Banded Ridge: Envelope vs. Acoustic Peak Rate
===================================================

This example demonstrates how to use BandedTRF to handle correlated features. 
We fit a model using the broadband speech envelope and the "peak rate" of 
the auditory spectrogram. By fitting the envelope first, we can determine 
if discrete peak rate events add unique predictive power (Delta R).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import naplib as nl
from naplib.encoding import BandedTRF

###############################################################################
# 1. Prepare the Data
# -------------------
# We compute the envelope by summing the auditory spectrogram over frequency
# bins, then compute peak rate using the dedicated naplib feature function.

data = nl.io.load_speech_task_data()

# Preprocess responses
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# Step A: Compute high-res auditory spectrogram (usually 128 bins)
# We use a sampling rate of 11025 Hz for the feature extraction
feat_fs = 11025
data['spec'] = [nl.features.auditory_spectrogram(trl['sound'], feat_fs) for trl in data]

# Step B: Compute Envelope and Peak Rate
# Peak rate uses the spectrogram to find acoustic landmarks
data['env_raw'] = [np.sum(trl['spec'], axis=1) for trl in data]
data['pk_raw'] = [nl.features.peak_rate(trl['spec'], feat_fs, band=[1, 10]) for trl in data]

# Step C: Resample features to match neural sampling rate (sfreq=100)
# Make sure the lengths match the response exactly
data['env'] = [resample(e, r.shape[0]) for e, r in zip(data['env_raw'], data['resp'])]
data['peak_rate'] = [resample(p, r.shape[0]) for p, r in zip(data['pk_raw'], data['resp'])]

# --- Visualization: Compare Stimulus Features ---
plt.figure(figsize=(10, 3))
plt.plot(data[0]['env'][:500] / np.max(data[0]['env']), label='Envelope (norm)', alpha=0.8)
plt.plot(data[0]['peak_rate'][:500] / np.max(data[0]['peak_rate']), label='Peak Rate (norm)', alpha=0.8)
plt.title('Stimulus Features: First 5 Seconds (Normalized for Viewing)')
plt.xlabel('Samples')
plt.legend()
plt.show()

###############################################################################
# 2. Fit the BandedTRF
# --------------------
# We fit the 'env' first, followed by 'peak_rate'.

tmin, tmax, sfreq = 0, 0.4, 100
alphas = np.logspace(-1, 5, 10) 
feature_order = ['env', 'peak_rate']

model = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=alphas)

# Fit on all but the last trial
model.fit(data=data[:-1], feature_order=feature_order, target='resp')

print(f"Optimized Alphas: {model.feature_alphas_}")

###############################################################################
# 3. Analyze Alpha Paths and Delta R
# ------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot Alpha Paths
for feat in feature_order:
    axes[0].semilogx(alphas, model.alpha_paths_[feat], marker='o', label=feat)
axes[0].set_title('Regularization Sweep (Alpha Paths)')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Mean Correlation (r)')
axes[0].legend()

# Compute Delta R on test data
pred_env = model.predict(data[-1:], feature_names=['env'])
r_env = np.mean(nl.evaluation.correlation(data[-1]['resp'], pred_env[0]))

pred_all = model.predict(data[-1:])
r_all = np.mean(nl.evaluation.correlation(data[-1]['resp'], pred_all[0]))

axes[1].bar(['Envelope Only', 'Env + Peak Rate'], [r_env, r_all], color=['#1f77b4', '#d62728'])
axes[1].set_ylim([min(r_env, r_all) * 0.9, max(r_env, r_all) * 1.1])
axes[1].set_title(f'Improvement (Delta R): {r_all - r_env:.4f}')
axes[1].set_ylabel('Pearson r')

plt.tight_layout()
plt.show()

###############################################################################
# 4. Compare TRF Kernels
# ----------------------
# Extract coefficients for the last electrode to see the temporal tuning.

elec = 10 
full_coefs = model.coef_ # (channels, features, lags)
lags = np.linspace(tmin, tmax, full_coefs.shape[-1])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lags, full_coefs[elec, 0, :], label='Envelope TRF', lw=2.5)
ax.plot(lags, full_coefs[elec, 1, :], label='Peak Rate TRF', lw=2.5)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_title(f'Comparison of TRF Kernels (Electrode {elec})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Weight (a.u.)')
ax.legend()
plt.tight_layout()
plt.show()