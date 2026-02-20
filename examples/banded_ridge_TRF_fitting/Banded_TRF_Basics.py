"""
====================
Banded STRF Basics
====================

Tutorial on fitting Banded TRF models.

This tutorial shows how to use the BandedTRF estimator to iteratively fit 
different feature sets (bands) with independent regularization.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from sklearn.linear_model import Ridge
import naplib as nl
from naplib.visualization import strf_plot
from naplib.encoding import BandedTRF

###############################################################################
# Set up the data
# ---------------

data = nl.io.load_speech_task_data()

# 1. Normalize neural responses
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# 2. Prepare Feature Band A: Auditory Spectrogram (32 channels)
data['spec'] = [nl.features.auditory_spectrogram(trial['sound'], 11025) for trial in data]
data['spec'] = [resample(trial['spec'], trial['resp'].shape[0]) for trial in data] 
resample_kwargs = {'num': 32, 'axis': 1}
data['spec_32'] = nl.array_ops.concat_apply(data['spec'], resample, function_kwargs=resample_kwargs)

# 3. Prepare Feature Band B: Temporal Envelope (1 channel)
# We use the Hilbert transform to get the broadband envelope
data['env'] = [np.abs(nl.features.hilbert_transform(trial['sound'])) for trial in data]
data['env'] = [resample(trial['env'], trial['resp'].shape[0]) for trial in data]

###############################################################################
# Fit Banded TRF Model
# --------------------
# 
# We will fit the features in order: first 'spec_32', then 'env'. 
# The model will optimize alpha for the spectrogram, fix it, and then 
# optimize alpha for the envelope.

tmin = 0 
tmax = 0.3 
sfreq = 100 

# Define the alpha sweep range
alphas = np.logspace(-2, 5, 8)

# Initialize the BandedTRF
banded_model = BandedTRF(tmin, tmax, sfreq, alphas=alphas)

# Split data
data_train = data[:-1]
data_test = data[-1:]

# Fit features iteratively
feature_order = ['spec_32', 'env']
banded_model.fit(data=data_train, feature_order=feature_order, target='resp')

print(f"Optimized Alphas: {banded_model.feature_alphas_}")

###############################################################################
# Analyze Banded Weights
# ----------------------
# 
# The .coef_ attribute returns weights for all features concatenated.
# For 32 spectral channels + 1 envelope channel, shape is (targets, 33, lags).

coefs = banded_model.coef_
elec = 9

# Split coefficients for visualization
# First 32 rows are the STRF, the 33rd row is the Envelope TRF
spec_coef = coefs[elec, :32, :]
env_coef = coefs[elec, 32:, :]



fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})

# Plot Spectrogram TRF
strf_plot(spec_coef, tmin=tmin, tmax=tmax, freqs=[171, 5000], ax=axes[0])
axes[0].set_title(f'Spectral Band TRF (α={banded_model.feature_alphas_["spec_32"]:.1f})')

# Plot Envelope TRF
lags = np.linspace(tmin, tmax, env_coef.shape[-1])
axes[1].plot(lags, env_coef.T)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_title(f'Envelope TRF\n(α={banded_model.feature_alphas_["env"]:.1f})')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()

###############################################################################
# Prediction Comparison
# ---------------------

# Standard TRF for comparison (joint optimization)
standard_model = nl.encoding.TRF(tmin, tmax, sfreq, estimator=Ridge(10))
# Combine features manually for standard TRF
data_train['combined'] = [np.hstack([s, e[:,None]]) for s, e in zip(data_train['spec_32'], data_train['env'])]
data_test['combined'] = [np.hstack([s, e[:,None]]) for s, e in zip(data_test['spec_32'], data_test['env'])]
standard_model.fit(data=data_train, X='combined', y='resp')

# Compute correlations
banded_preds = banded_model.predict(data=data_test)
standard_preds = standard_model.predict(data=data_test, X='combined')

r_banded = nl.evaluation.correlation(data_test['resp'][-1], banded_preds[-1])
r_standard = nl.evaluation.correlation(data_test['resp'][-1], standard_preds[-1])

print(f"Mean Banded Correlation: {np.mean(r_banded):.3f}")
print(f"Mean Standard Correlation: {np.mean(r_standard):.3f}")