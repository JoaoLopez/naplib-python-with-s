import pytest
import numpy as np
import pandas as pd
from naplib import Data
from naplib.encoding import BandedTRF

@pytest.fixture(scope='module')
def synth_data():
    """
    Generate synthetic data with 2 target channels.
    Matches BandedTRF expected input format: list of arrays or Data object.
    """
    rng = np.random.default_rng(42)
    fs, n_samples, n_trials = 100, 1000, 3
    trials = []
    for _ in range(n_trials):
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        
        # Create 2 target channels (multi-output)
        y1 = (x1 * 1.0 + np.roll(x2, 2) * 0.5)
        y2 = (x1 * 0.5 + np.roll(x2, 1) * 1.0)
        resp = np.hstack([y1, y2]) + 0.01 * rng.standard_normal((n_samples, 2))
        
        trials.append({'resp': resp, 'stim1': x1, 'stim2': x2})
    
    return {
        'data': Data(trials),
        'feature_order': ['stim1', 'stim2'],
        'tmin': 0, 'tmax': 0.03, 'sfreq': fs
    }

def test_banded_trf_loto_consistency(synth_data):
    """Test that coef_ property handles the 4D reshape correctly."""
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'], alphas=[0.1, 10.0])
    model.fit(data=synth_data['data'], X=synth_data['feature_order'], y='resp')
    
    # Shape calculation: 2 targets, 2 feature bands, 4 delays, 3 trials.
    # ndelays = round(0.03*100) - round(0*100) + 1 = 4.
    # Note: If stim1 and stim2 are single-dimension, total features = 2.
    assert model.coef_.shape == (2, 2, 4, 3)

def test_predict_masking_logic(synth_data):
    """Verify that partial feature prediction works (greedy logic)."""
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], X=synth_data['feature_order'], y='resp')
    
    # 1. Full prediction
    preds_all = model.predict(data=synth_data['data'], X=synth_data['feature_order'])
    assert len(preds_all) == 3
    assert preds_all[0].shape == (1000, 2)
    
    # 2. Partial prediction (subset of features)
    # This triggers the 'Using reduced TRF' print/logic in predict()
    preds_sub = model.predict(data=synth_data['data'], X=['stim1'])
    assert len(preds_sub) == 3
    assert preds_sub[0].shape == (1000, 2)

def test_summary_p_values(synth_data):
    """Verify summary table computes stats across channels correctly."""
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], X=synth_data['feature_order'], y='resp')
    
    # Test Global Summary
    df = model.summary()
    assert isinstance(df, pd.DataFrame)
    assert 'Delta R' in df.columns
    assert 'p-value' in df.columns
    assert not df['p-value'].isna().any()
    
    # Test specific channel summary
    df_ch0 = model.summary(channel=0)
    assert len(df_ch0) == 2

def test_unfitted_attribute_error():
    """Verify custom AttributeError message for unfitted models."""
    model = BandedTRF(0, 0.1, 100)
    with pytest.raises(AttributeError, match="BandedTRF has not been fitted yet."):
        _ = model.coef_

def test_predict_trial_mismatch(synth_data):
    """LOTO requires the same number of trials for predict as fit."""
    model = BandedTRF(tmin=synth_data['tmin'], tmax=synth_data['tmax'], sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], X=synth_data['feature_order'], y='resp')
    
    # Try predicting with only 2 trials instead of 3
    short_data = synth_data['data'][:2]
    with pytest.raises(ValueError, match="LOTO predict requires the same number of trials"):
        model.predict(data=short_data, X=synth_data['feature_order'])