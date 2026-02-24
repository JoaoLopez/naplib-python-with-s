import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from naplib import Data
from naplib.encoding import BandedTRF
from naplib.stats import pairwise_correlation

@pytest.fixture(scope='module')
def synth_data():
    """
    Generate synthetic data for testing.
    Matches the LOTO requirement where n_trials must remain consistent.
    """
    rng = np.random.default_rng(42)
    fs = 100
    n_samples = 1000
    n_trials = 3
    trials = []
    
    for _ in range(n_trials):
        # Features must be (samples, n_features)
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        
        # stim1 drives response at lag 0, stim2 at lag 2
        y1 = x1 * 1.0
        y2 = np.zeros_like(x2)
        y2[2:] = x2[:-2] * 0.5
        
        # response must be (samples, n_channels)
        resp = y1 + y2 + 0.05 * rng.standard_normal(y1.shape)
        trials.append({'resp': resp, 'stim1': x1, 'stim2': x2})
        
    return {
        'data': Data(trials),
        'feature_order': ['stim1', 'stim2'],
        'tmin': 0,
        'tmax': 0.03, # Resulting in 4 delays: 0, 0.01, 0.02, 0.03
        'sfreq': fs
    }

def test_banded_trf_loto_consistency(synth_data):
    """Verify alpha selection and coefficient storage."""
    alphas = [1e-1, 1e5]
    model = BandedTRF(tmin=synth_data['tmin'],
                      tmax=synth_data['tmax'],
                      sfreq=synth_data['sfreq'],
                      alphas=alphas)
    
    model.fit(data=synth_data['data'],
              feature_order=synth_data['feature_order'],
              target='resp')
    
    # 1. Check alpha_paths_ (naming fix from implementation)
    assert hasattr(model, 'alpha_paths_')
    assert 'stim1' in model.alpha_paths_
    assert len(model.alpha_paths_['stim1']) == len(alphas)
    
    # 2. Verify 4D coef_ shape: (n_targets, n_features, n_delays, n_trials)
    # 1 target, 2 features, 4 delays, 3 trials
    assert model.coef_.shape == (1, 2, 4, 3)

def test_summary_delta_r(synth_data):
    """Check if the summary table correctly computes incremental Delta R."""
    model = BandedTRF(tmin=synth_data['tmin'],
                      tmax=synth_data['tmax'],
                      sfreq=synth_data['sfreq'])
    
    model.fit(data=synth_data['data'],
              feature_order=synth_data['feature_order'],
              target='resp')
    
    df = model.summary()
    assert isinstance(df, pd.DataFrame)
    assert 'Delta R' in df.columns
    assert 'Total R' in df.columns
    # Feature 1 (stim1) should be significant
    assert df.loc['stim1', 'Delta R'] > 0
    assert df.loc['stim1', 'p-value'] < 0.05

def test_predict_loto_averaging(synth_data):
    """Ensure prediction uses LOTO (averaging weights from other trials)."""
    model = BandedTRF(tmin=synth_data['tmin'],
                      tmax=synth_data['tmax'],
                      sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'],
              feature_order=synth_data['feature_order'],
              target='resp')
    
    # LOTO implementation requires same n_trials for predict
    preds = model.predict(synth_data['data'])
    
    assert len(preds) == 3
    assert preds[0].shape == synth_data['data'][0]['resp'].shape
    
    # Check correlation
    r = pairwise_correlation(synth_data['data'][0]['resp'], preds[0])
    assert r[0] > 0.8

def test_not_fitted_error():
    """Accessing coef_ should raise AttributeError before fit."""
    model = BandedTRF(0, 0.1, 100)
    with pytest.raises(AttributeError, match="not been fitted"):
        _ = model.coef_

def test_single_trial_error(synth_data):
    """LOTO requires at least 2 trials for np.mean([indices j != i])."""
    single_trial_data = synth_data['data'][:1]
    model = BandedTRF(0, 0.1, 100)
    # The current fit loop will attempt np.mean on an empty slice
    with pytest.raises(Exception): 
        model.fit(data=single_trial_data, feature_order=['stim1'], target='resp')

def test_pairwise_correlation_logic():
    """Verify Pearson R returns 1D array for 2D inputs (samples, channels)."""
    a = np.array([[1, 2, 3], [4, 5, 6]]).T # (3 samples, 2 channels)
    b = np.array([[1, 2, 3], [4, 5, 6]]).T
    r = pairwise_correlation(a, b)
    # Should return shape (2,)
    assert r.shape == (2,)
    assert np.allclose(r, 1.0)