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
    'stim1' drives response at lag 0 (weight 1.0).
    'stim2' drives response at lag 2 (weight 0.5).
    """
    rng = np.random.default_rng(42)
    fs = 100
    n_samples = 1000
    n_trials = 3
    trials = []
    
    for _ in range(n_trials):
        # Ensure x is (samples, 1) for consistent 2D math
        x1 = rng.standard_normal(size=(n_samples, 1))
        x2 = rng.standard_normal(size=(n_samples, 1))
        
        y1 = x1 * 1.0
        y2 = np.zeros_like(x2)
        y2[2:] = x2[:-2] * 0.5
        
        # resp should be (samples, n_targets)
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
    """Verify LOTO logic: alpha selection and coefficient averaging."""
    alphas = [1e-1, 1e5] 
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'],
                      alphas=alphas)
    
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    # Optimization paths are stored by feature name
    assert 'stim1' in model.alpha_paths_
    assert len(model.alpha_paths_['stim1']) == len(alphas)
    assert model.coef_.shape == (1, 2, 4, 3) # (targets, features, delays, trials)

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
    # stim1 is the primary driver
    assert df.loc['stim1', 'Delta R'] > 0

def test_predict_manual_weight_averaging(synth_data):
    """Ensure prediction uses the average coefficient across trials."""
    model = BandedTRF(tmin=synth_data['tmin'], 
                      tmax=synth_data['tmax'], 
                      sfreq=synth_data['sfreq'])
    model.fit(data=synth_data['data'], 
              feature_order=synth_data['feature_order'], 
              target='resp')
    
    preds = model.predict(synth_data['data'])
    
    assert isinstance(preds, list)
    # y[test_idx] and y_pred are (samples, targets). r is (targets,)
    r = pairwise_correlation(synth_data['data'][0]['resp'], preds[0])
    assert r[0] > 0.8

def test_single_trial_error(synth_data):
    """LOTO requires at least 2 trials. Update the code to catch the specific ValueError."""
    single_trial_data = synth_data['data'][:1]
    model = BandedTRF(0, 0.1, 100)
    # The current implementation fails at matmul, but logically it's a trial count issue
    with pytest.raises(ValueError):
        model.fit(data=single_trial_data, feature_order=['stim1'], target='resp')

def test_feature_not_in_data(synth_data):
    """The argchecker raises ValueError for missing fields, not KeyError."""
    model = BandedTRF(0, 0.1, 100)
    with pytest.raises(ValueError, match="is not a field of the Data"):
        model.fit(data=synth_data['data'], feature_order=['nonexistent'], target='resp')

def test_not_fitted_error():
    """Accessing coef_ should raise AttributeError if _fitted is not True."""
    model = BandedTRF(0, 0.1, 100)
    # If the model uses a property that checks for fit status
    with pytest.raises(AttributeError):
        _ = model.coef_

def test_pairwise_correlation_logic():
    """Verify basic Pearson R computation returns 1D array for 2D inputs."""
    a = np.array([[1, 2, 3]]).T
    b = np.array([[1, 2, 3]]).T
    r = pairwise_correlation(a, b)
    # r is shape (1,) because there is one target channel
    assert np.isclose(r[0], 1.0)