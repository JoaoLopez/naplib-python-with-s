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
    Generate synthetic data. 
    Crucially: resp is (samples, 1) and stims are (samples, 1).
    """
    rng = np.random.default_rng(42)
    fs = 100
    n_samples = 1000
    n_trials = 3
    trials = []
    
    for _ in range(n_trials):
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        
        y = x1 * 1.0 + 0.5 * np.roll(x2, 2)
        resp = y + 0.01 * rng.standard_normal(y.shape)
        
        trials.append({'resp': resp, 'stim1': x1, 'stim2': x2})
        
    return {
        'data': Data(trials),
        'feature_order': ['stim1', 'stim2'],
        'tmin': 0,
        'tmax': 0.03, # 4 delays
        'sfreq': fs
    }

def test_banded_trf_loto_consistency(synth_data):
    """Verify coefficient storage and shape."""
    model = BandedTRF(tmin=synth_data['tmin'],
                      tmax=synth_data['tmax'],
                      sfreq=synth_data['sfreq'],
                      alphas=[0.1, 10.0])
    
    model.fit(data=synth_data['data'],
              feature_order=synth_data['feature_order'],
              target='resp')
    
    # Calculation: 1 target * 2 features * 4 delays * 3 trials = 24 elements.
    # The reshape in your class: (n_targets, n_feats, n_delays, n_trials)
    assert model.coef_.shape == (1, 2, 4, 3)

def test_predict_loto_averaging(synth_data):
    """Ensure prediction handles the LOTO averaging and masking without IndexErrors."""
    model = BandedTRF(tmin=synth_data['tmin'],
                      tmax=synth_data['tmax'],
                      sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'],
              feature_order=synth_data['feature_order'],
              target='resp')
    
    # Test full prediction
    preds = model.predict(synth_data['data'])
    assert len(preds) == 3
    assert preds[0].shape == (1000, 1)

    # Test partial prediction (triggers the masking logic)
    preds_sub = model.predict(synth_data['data'], feature_names=['stim1'])
    assert len(preds_sub) == 3
    assert preds_sub[0].shape == (1000, 1)

def test_summary_output(synth_data):
    """Verify summary table structure and p-values."""
    model = BandedTRF(tmin=synth_data['tmin'],
                      tmax=synth_data['tmax'],
                      sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'],
              feature_order=synth_data['feature_order'],
              target='resp')
    
    df = model.summary()
    assert isinstance(df, pd.DataFrame)
    assert 'Delta R' in df.columns
    # With this SNR, stim1 should definitely be positive
    assert df.loc['stim1', 'Total R'] > 0.5

def test_not_fitted_error():
    """Accessing weights before fit should raise AttributeError."""
    model = BandedTRF(0, 0.1, 100)
    with pytest.raises(AttributeError, match="not been fitted"):
        _ = model.coef_