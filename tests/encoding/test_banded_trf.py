import pytest
import numpy as np
from scipy.signal import convolve
from sklearn.linear_model import Ridge

from naplib import Data
from naplib.encoding import BandedTRF # Assuming this is where it's saved

@pytest.fixture(scope='module')
def banded_data():
    """
    Generate synthetic data where 'resp' is a combination of two 
    distinct features ('stim1', 'stim2') with different optimal lags.
    """
    rng = np.random.default_rng(1)
    n_samples = 10000
    
    # Feature 1: Immediate response
    x1 = rng.random(size=(n_samples, 1))
    coef1 = np.array([[1.0], [0.0]]) # Lag 0
    y1 = convolve(x1, coef1, mode='same')
    
    # Feature 2: Delayed response
    x2 = rng.random(size=(n_samples, 1))
    coef2 = np.array([[0.0], [0.8]]) # Lag 1
    y2 = convolve(x2, coef2, mode='same')
    
    # Combined response with some noise
    resp = y1 + y2 + 0.05 * rng.standard_normal(y1.shape)
    
    # Multiple trials for cross-validation tests
    trial1 = {'resp': resp, 'stim1': x1, 'stim2': x2}
    trial2 = {'resp': resp, 'stim1': x1, 'stim2': x2}
    
    outstruct = Data([trial1, trial2])
    
    return {
        'outstruct': outstruct,
        'feature_order': ['stim1', 'stim2'],
        'sfreq': 100,
        'tmin': 0,
        'tmax': 0.01 # 2 samples at 100Hz
    }

def test_banded_fit_logic(banded_data):
    """Test if the model fits and populates the feature_alphas_ attribute."""
    model = BandedTRF(tmin=banded_data['tmin'], 
                      tmax=banded_data['tmax'], 
                      sfreq=banded_data['sfreq'],
                      alphas=[0.1, 1.0, 10.0])
    
    model.fit(data=banded_data['outstruct'], 
              feature_order=banded_data['feature_order'], 
              target='resp')
    
    # Check if all features in order have an assigned alpha
    assert len(model.feature_alphas_) == 2
    for feat in banded_data['feature_order']:
        assert feat in model.feature_alphas_

def test_banded_coef_shape(banded_data):
    """Verify the coefficient shape: (targets, features, lags)."""
    model = BandedTRF(tmin=banded_data['tmin'], 
                      tmax=banded_data['tmax'], 
                      sfreq=banded_data['sfreq'])
    
    model.fit(data=banded_data['outstruct'], 
              feature_order=banded_data['feature_order'], 
              target='resp')
    
    # 1 target, 2 features, 2 lags
    expected_shape = (1, 2, 2)
    assert model.coef_.shape == expected_shape

def test_banded_prediction(banded_data):
    """Verify that predictions are returned as a list of arrays (one per trial)."""
    model = BandedTRF(tmin=banded_data['tmin'], 
                      tmax=banded_data['tmax'], 
                      sfreq=banded_data['sfreq'])
    
    model.fit(data=banded_data['outstruct'], 
              feature_order=banded_data['feature_order'])
    
    preds = model.predict(data=banded_data['outstruct'])
    
    assert isinstance(preds, list)
    assert len(preds) == len(banded_data['outstruct'])
    assert preds[0].shape == banded_data['outstruct'][0]['resp'].shape

def test_feature_order_dependency(banded_data):
    """
    Ensure the model respects feature order. Fitting [A, B] should result 
    in different alpha selections/coefs than [B, A] due to the iterative nature.
    """
    model_ab = BandedTRF(tmin=0, tmax=0.01, sfreq=100, alphas=[0.1, 100.0])
    model_ab.fit(data=banded_data['outstruct'], feature_order=['stim1', 'stim2'])
    
    model_ba = BandedTRF(tmin=0, tmax=0.01, sfreq=100, alphas=[0.1, 100.0])
    model_ba.fit(data=banded_data['outstruct'], feature_order=['stim2', 'stim1'])
    
    # The order of features in the final coef_ property should match feature_order
    # We check if they differ in content because 'stim2' was optimized against a 
    # different 'fixed' background in each case.
    assert not np.array_equal(model_ab.coef_, model_ba.coef_)

def test_not_fitted_error():
    """Ensure predict and coef_ raise errors if called before fit."""
    model = BandedTRF(tmin=0, tmax=0.1, sfreq=100)
    
    with pytest.raises(ValueError, match="Model not fitted"):
        _ = model.coef_
        
    with pytest.raises(ValueError, match="Model not fitted"):
        model.predict(data=None)