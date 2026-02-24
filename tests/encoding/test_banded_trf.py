import pytest
import numpy as np
from sklearn.linear_model import Ridge

from naplib import Data
from naplib.encoding import BandedTRF
from naplib.stats import pairwise_correlation

@pytest.fixture(scope='module')
def synth_data():
    """
    Generate synthetic data for testing.
    'resp' must be (samples, n_targets).
    """
    rng = np.random.default_rng(42)
    fs = 100
    n_samples = 1000
    n_trials = 3
    trials = []
    
    for _ in range(n_trials):
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        
        y1 = x1 * 1.0
        y2 = np.zeros_like(x2)
        y2[2:] = x2[:-2] * 0.5
        
        # Ensure resp is (samples, 1) to avoid broadcasting errors
        resp = y1 + y2 + 0.05 * rng.standard_normal(y1.shape)
        trials.append({'resp': resp, 'stim1': x1, 'stim2': x2})
        
    return {
        'data': Data(trials),
        'feature_order': ['stim1', 'stim2'],
        'tmin': 0,
        'tmax': 0.03, # 4 samples: 0, 1, 2, 3
        'sfreq': fs
    }

def test_banded_trf_loto_consistency(synth_data):
    """Verify LOTO logic: alpha selection and coefficient storage."""
    alphas = [1e-1, 1e5] 
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'],
                      alphas=alphas)
    
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    # FIX: Use alpha_paths_ (dict) instead of optimization_paths_ (list)
    assert hasattr(model, 'alpha_paths_'), "BandedTRF should store alpha paths in 'alpha_paths_'"
    assert 'stim1' in model.alpha_paths_
    assert len(model.alpha_paths_['stim1']) == len(alphas)
    
    # Verify 4D coef_ shape: (n_targets, n_features, n_delays, n_trials)
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
    assert 'Delta R' in df.columns
    # stim1 is the primary signal driver
    assert df.loc['stim1', 'Delta R'] > 0

def test_predict_functionality(synth_data):
    """Verify predictive accuracy on the trained data."""
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    preds = model.predict(synth_data['data'])
    
    assert isinstance(preds, list)
    # pairwise_correlation returns (n_channels,)
    r = pairwise_correlation(synth_data['data'][0]['resp'], preds[0])
    assert r[0] > 0.8

def test_not_fitted_error():
    """Accessing model weights before fit should raise AttributeError."""
    model = BandedTRF(0, 0.1, 100)
    # Check if the property raises AttributeError correctly
    with pytest.raises(AttributeError):
        _ = model.coef_