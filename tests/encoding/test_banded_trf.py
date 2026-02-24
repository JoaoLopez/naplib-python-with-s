import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from naplib import Data
from naplib.encoding import BandedTRF
from naplib.stats import pairwise_correlation

@pytest.fixture(scope='module')
def synth_data():
    rng = np.random.default_rng(42)
    fs, n_samples, n_trials = 100, 1000, 3
    trials = []
    for _ in range(n_trials):
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        # Ensure response is 2D (samples, 1 channel)
        resp = (x1 * 1.0 + np.roll(x2, 2) * 0.5) + 0.01 * rng.standard_normal((n_samples, 1))
        trials.append({'resp': resp, 'stim1': x1, 'stim2': x2})
    
    return {
        'data': Data(trials),
        'feature_order': ['stim1', 'stim2'],
        'tmin': 0, 'tmax': 0.03, 'sfreq': fs
    }

def test_banded_trf_loto_consistency(synth_data):
    """Test that coef_ property handles 1D vs 2D Ridge coefficients correctly."""
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'], alphas=[0.1, 10.0])
    model.fit(data=synth_data['data'], feature_order=synth_data['feature_order'], target='resp')
    
    # We have 1 target, 2 features, 4 delays, 3 trials. Total elements = 24.
    # If this fails, the internal 'n_targets' logic in the property is wrong.
    assert model.coef_.shape == (1, 2, 4, 3)

def test_predict_masking_logic(synth_data):
    """Verify that partial feature prediction doesn't cause IndexError."""
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], feature_order=synth_data['feature_order'], target='resp')
    
    # Full prediction
    preds_all = model.predict(synth_data['data'])
    assert len(preds_all) == 3
    
    # Partial prediction (Triggers the internal mask logic)
    # This specifically addresses the 'tuple index out of range' error
    preds_sub = model.predict(synth_data['data'], feature_names=['stim1'])
    assert len(preds_sub) == 3
    assert preds_sub[0].shape == (1000, 1)

def test_summary_p_values(synth_data):
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], feature_order=synth_data['feature_order'], target='resp')
    
    df = model.summary()
    assert 'p-value' in df.columns
    # With n_trials=3, t-test has 2 degrees of freedom
    assert df.loc['stim1', 'p-value'] < 0.1 

def test_unfitted_attribute_error():
    model = BandedTRF(0, 0.1, 100)
    with pytest.raises(AttributeError, match="not been fitted"):
        _ = model.coef_