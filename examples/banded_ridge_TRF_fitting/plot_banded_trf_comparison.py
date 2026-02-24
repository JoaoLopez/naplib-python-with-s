import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.stats import zscore
import naplib as nl
from naplib.encoding import TRF, BandedTRF
from sklearn.linear_model import Ridge

###############################################################################
# 1. Prepare Synthetic Data with Known Ground Truth
###############################################################################
# Load real speech data as a template for stimulus statistics
data = nl.io.load_speech_task_data()
feat_fs = 100

# Compute Features
data['aud_spec'] = [resample(nl.features.auditory_spectrogram(trl['sound'], 11025), trl['resp'].shape[0], axis=0) for trl in data]

data['env'] = [zscore(np.sum(trl['aud_spec'], axis=1)) for trl in data]

# Compute Peak Rate (Sparse events)
data['peak_rate'] = [nl.features.peak_rate(trl['aud_spec'], feat_fs) for trl in data]

# Inject Noise Band (Matches envelope variance but is unrelated to brain)
np.random.seed(42)
for i in range(len(data)):
    noise = np.random.randn(data[i]['resp'].shape[0])
    data[i]['noise'] = (noise / np.std(noise)) * np.std(data[i]['env'])

###############################################################################
# 2. Fit Standard TRF (Global Alpha)
###############################################################################
tmin, tmax, sfreq = -0.1, 0.5, 100
feature_list = ['env', 'noise', 'peak_rate']

# Standard TRF uses one alpha for all concatenated features
standard_model = TRF(tmin, tmax, sfreq, estimator=Ridge(alpha=1000))
standard_model.fit(data=data[:-1], X=feature_list, y='resp')
standard_scores = standard_model.score(data=data[-1:], X=feature_list, y='resp')

###############################################################################
# 3. Fit Banded TRF (Feature-Specific Alphas)
###############################################################################
# Banded TRF optimizes alpha per band sequentially
banded_model = BandedTRF(tmin=tmin, tmax=tmax, sfreq=sfreq, alphas=np.logspace(0, 7, 8))
banded_model.fit(data=data[:-1], feature_order=feature_list, target='resp')

###############################################################################
# 4. Compare Results
###############################################################################
# Generate Banded Summary (Delta R analysis)
print("\n--- Banded TRF Statistical Summary ---")
df_summary = banded_model.summary()

# Plotting the three requested comparisons
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# a) Total Correlation Comparison
axes[0].bar(['Standard TRF', 'Banded TRF'], 
            [np.mean(standard_scores), df_summary['Total R'].iloc[-1]], 
            color=['#7f7f7f', '#1f77b4'])
axes[0].set_title('a) Total Predictive Accuracy (R)\n(Standard vs. Final Banded)')
axes[0].set_ylabel('Pearson Correlation')

# b) Delta R when adding Peak Rate
# This shows the unique contribution of peaks above the envelope
axes[1].bar(df_summary.index[:2], df_summary['Delta R'].iloc[:2], color=['#1f77b4', '#d62728'])
axes[1].set_title('b) Unique Contribution (Delta R)\n(Env vs. Peak Rate)')
axes[1].set_ylabel('Improvement in R')

# c) Non-zero Delta R when adding Noise
# This highlights the model's robustness to irrelevant bands
axes[2].bar(df_summary.index, df_summary['Delta R'], color=['#1f77b4', '#d62728', '#bcbd22'])
axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[2].set_title('c) Robustness Check\n(Delta R for Noise Band)')
axes[2].set_ylabel('Improvement in R')

plt.tight_layout()
plt.show()

# Visualize Kernels for Banded Model
best_ch = 0
lags = np.linspace(tmin, tmax, banded_model._ndelays)
plt.figure(figsize=(10, 4))
for f_idx, feat in enumerate(feature_list):
    # coef_ is (targets, features, delays, trials) -> average over trials
    kernel = banded_model.coef_[best_ch, f_idx, :, :].mean(axis=-1)
    plt.plot(lags, kernel, label=feat, lw=2 if feat != 'noise' else 1)

plt.axhline(0, color='k', alpha=0.5)
plt.title(f'Banded TRF Kernels (Ch {best_ch}) - Noise is effectively Zeroed')
plt.xlabel('Lag (s)')
plt.legend()
plt.show()