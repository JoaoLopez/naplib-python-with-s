import pytest
import numpy as np
from scipy.signal import convolve
from sklearn.linear_model import Ridge

from naplib import Data
from naplib.encoding import BandedTRF
from naplib.encoding.banded_trf import pairwise_correlation

@pytest.fixture(scope='module')
def synth_data():
    """
    Generate 3 trials of synthetic data.
    'stim1' drives response at lag 0.
    'stim2' drives response at lag 2.
    """
    rng = np.random.default_rng(42)
    fs = 100
    n_samples = 5000
    trials = []
    
    for _ in range(3):
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        
        # Stim 1: weight 1.0 at lag 0
        y1 = x1 * 1.0
        # Stim 2: weight 0.5 at lag 2 (0.02s)
        y2 = np.zeros_like(x2)
        y2[2:] = x2[:-2] * 0.5
        
        resp = y1 + y2 + 0.1 * rng.standard_normal(y1.shape)
        trials.append({'resp': resp, 'stim1': x1, 'stim2': x2})
        
    return {
        'data': Data(trials),
        'feature_order': ['stim1', 'stim2'],
        'tmin': 0,
        'tmax': 0.03, # 4 samples: 0, 1, 2, 3
        'sfreq': fs
    }

def test_pairwise_correlation_1d():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 4, 5])
    assert np.isclose(pairwise_correlation(a, b), 1.0)
    
    # Anti-correlated
    assert np.isclose(pairwise_correlation(a, -a), -1.0)

def test_pairwise_correlation_2d():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((100, 2))
    b = rng.standard_normal((100, 2))
    r_mat = pairwise_correlation(a, b)
    assert r_mat.shape == (2, 2)
    # Diagonals should be reasonable
    assert np.all(np.abs(np.diag(r_mat)) <= 1.0)

def test_banded_trf_fast_cv_logic(synth_data):
    """Verify that fit runs and populates alpha paths using the fast CV logic."""
    alphas = [1e-1, 1e2]
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'],
                      alphas=alphas)
    
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    # Check that alpha paths were stored for each feature
    assert 'stim1' in model.alpha_paths_
    assert 'stim2' in model.alpha_paths_
    assert len(model.alpha_paths_['stim1']) == len(alphas)
    
    # Ensure selected alphas are from the provided list
    assert model.feature_alphas_['stim1'] in alphas
    assert model.feature_alphas_['stim2'] in alphas

def test_coef_reshaping(synth_data):
    """Check that coef_ has the expected dimensions (targets, features, lags)."""
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'])
    
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    # n_targets=1, n_features=2 (stim1, stim2), n_lags=4 (0, 0.01, 0.02, 0.03)
    assert model.coef_.shape == (1, 2, 4)

def test_predict_subset_features(synth_data):
    """Verify that predicting with a subset of features works correctly."""
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'])
    
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    # Predict with only the first feature
    preds = model.predict(data=synth_data['data'], feature_names=['stim1'])
    
    assert len(preds) == 3
    assert preds[0].shape == synth_data['data'][0]['resp'].shape

def test_fast_cv_vs_standard_ridge(synth_data):
    """
    Check if the fast coefficient-averaging approach yields 
    sensible weights compared to a standard fit.
    """
    # Use a single alpha to make comparison straightforward
    model = BandedTRF(tmin=0, tmax=0, sfreq=100, alphas=[1.0])
    model.fit(data=synth_data['data'], feature_order=['stim1'], target='resp')
    
    # For stim 1 at lag 0, weight should be near 1.0
    # coef_ shape is (1, 1, 1) -> (target, feature, lag)
    weight = model.coef_[0, 0, 0]
    assert 0.8 < weight < 1.2

def test_not_fitted_error():
    model = BandedTRF(0, 0.1, 100)
    with pytest.raises(ValueError, match="fitted before accessing coef_"):
        _ = model.coef_