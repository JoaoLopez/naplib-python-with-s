import numpy as np
import pytest
from naplib.stats import pairwise_correlation

def test_pairwise_correlation_1d():
    # Identical 1D signals
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    corr = pairwise_correlation(a, b)
    assert np.isclose(corr, 1.0, atol=1e-8)

    # Inverse 1D signals
    b_inv = -b
    corr_inv = pairwise_correlation(a, b_inv)
    assert np.isclose(corr_inv, -1.0, atol=1e-8)

def test_pairwise_correlation_2d_default_axis():
    # Default axis=0 (correlate columns over rows)
    A = np.array([[1, 2], 
                  [2, 1], 
                  [3, 3]])
    B = A.copy()
    corr = pairwise_correlation(A, B) # Expected shape (2,)
    assert corr.shape == (2,)
    assert np.allclose(corr, [1.0, 1.0], atol=1e-8)

def test_pairwise_correlation_2d_custom_axis():
    # axis=1 (correlate rows over columns)
    A = np.array([[1, 2, 3], 
                  [4, 5, 6]])
    B = A.copy()
    corr = pairwise_correlation(A, B, axis=1) # Expected shape (2,)
    assert corr.shape == (2,)
    assert np.allclose(corr, [1.0, 1.0], atol=1e-8)

def test_pairwise_correlation_3d_neural_data():
    # Simulation of (trials, channels, time) correlating over time axis
    rng = np.random.default_rng(42)
    A = rng.standard_normal((5, 10, 100)) # 5 trials, 10 channels, 100 samples
    B = A.copy()
    
    # Correlate over time (axis=2)
    corr = pairwise_correlation(A, B, axis=2)
    assert corr.shape == (5, 10) # One correlation per trial/channel
    assert np.allclose(corr, 1.0, atol=1e-8)

def test_pairwise_correlation_shape_mismatch():
    A = np.ones((10, 2))
    B = np.ones((10, 3))
    with pytest.raises(ValueError, match="A and B must have the same shape"):
        pairwise_correlation(A, B)

def test_pairwise_correlation_zero_variance():
    # Test epsilon handling for constant signals (prevents nan)
    A = np.array([1.0, 1.0, 1.0])
    B = np.array([2.0, 2.0, 2.0])
    corr = pairwise_correlation(A, B)
    # With 1e-15 in denominator and 0 in numerator, result is 0
    assert np.isclose(corr, 0.0, atol=1e-8)

def test_pairwise_correlation_random_precision():
    # Test against np.corrcoef for a single pair to ensure mathematical parity
    rng = np.random.default_rng(1)
    a = rng.standard_normal(100)
    b = rng.standard_normal(100)
    
    expected = np.corrcoef(a, b)[0, 1]
    actual = pairwise_correlation(a, b)
    assert np.isclose(actual, expected, atol=1e-8)