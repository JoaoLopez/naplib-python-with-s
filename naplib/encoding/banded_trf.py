import copy
import numpy as np
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from mne.decoding.receptive_field import _delay_time_series
from .utils import _parse_outstruct_args

def pairwise_correlation(A, B):
    """
    Computes Pearson correlation. Works for 1D vectors (returns scalar) 
    and 2D matrices (returns dot product covariance / normalization).
    """
    am = A - np.mean(A, axis=0)
    bm = B - np.mean(B, axis=0)
    
    # Using np.dot handles 1D vectors naturally
    coscale = np.dot(am, bm)
    a_ss = np.dot(am, am)
    b_ss = np.dot(bm, bm)
    
    return coscale / np.sqrt(a_ss * b_ss + 1e-15)

class BandedTRF(BaseEstimator):
    """
    Class for fitting iterative Banded Ridge TRF models to neural data.
    
    Features are added and optimized one-by-one. Each subsequent feature's 
    alpha is optimized while previously added features are held constant 
    at their optimal regularization levels.
    
    Parameters
    ----------
    tmin : float
        Starting lag (seconds).
    tmax : float
        Ending lag (seconds).
    sfreq : float
        Sampling frequency (Hz).
    alphas : ndarray, optional
        Alphas to sweep for each feature. Default is np.logspace(-2, 5, 8).
    basis_dict : dict, optional
        Basis expansion functions/objects for specific features.
    """
    def __init__(self, tmin, tmax, sfreq, alphas=None, basis_dict=None):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.alphas = alphas if alphas is not None else np.logspace(-2, 5, 8)
        self.basis_dict = basis_dict if basis_dict is not None else {}
        self.feature_alphas_ = {}
        self.feature_order_ = []
        self.model_ = None

    @property
    def _ndelays(self):
        return int(round(self.tmax * self.sfreq)) - int(round(self.tmin * self.sfreq)) + 1

    def _prepare_matrix(self, X_list, feature_names, alphas_dict):
        """Prepares design matrix by applying bases, scaling by alpha, and time-lagging."""
        processed_trials = []
        num_trials = len(X_list[0])
        
        for trl in range(num_trials):
            mats = []
            for i, name in enumerate(feature_names):
                x = X_list[i][trl]
                if x.ndim == 1:
                    x = x[:, np.newaxis]
                
                # Apply basis expansion
                if name in self.basis_dict:
                    # Logic for apply_bases or transformer object
                    x = apply_bases(x, self.basis_dict[name]) 
                
                alpha = alphas_dict.get(name, 1.0)
                mats.append(x / alpha)
            
            concatenated = np.concatenate(mats, axis=1)
            # Time lagging used by naplib internally
            
            delayed = _delay_time_series(concatenated, self.tmin, self.tmax, self.sfreq)
            processed_trials.append(delayed.reshape(delayed.shape[0], -1))
            
        return processed_trials

    def fit(self, data, feature_order, target='resp'):
        """
        Fit features iteratively.
        
        Parameters
        ----------
        data : naplib.Data
            The Data object containing trials.
        feature_order : list of str
            Order in which features are added and optimized.
        target : str
            Field name of the target response (e.g., 'eeg').
        """
        self.feature_order_ = feature_order
        _, y = _parse_outstruct_args(data, feature_order[0], target)
        self.n_targets_ = y[0].shape[1]
        
        # Load data once
        all_features_data = []
        for feat in feature_order:
            feat_data, _ = _parse_outstruct_args(data, feat, target)
            all_features_data.append(feat_data)

        for i, current_feat in enumerate(feature_order):
            best_alpha = None
            max_r = -np.inf
            
            for alpha in tqdm(self.alphas, desc=f"Optimizing {current_feat}", leave=False):
                temp_alphas = {**self.feature_alphas_, current_feat: alpha}
                X_mats = self._prepare_matrix(all_features_data[:i+1], feature_order[:i+1], temp_alphas)
                
                # Cross-validation over trials
                trial_corrs = []
                for test_idx in range(len(X_mats)):
                    X_train = np.concatenate([X_mats[j] for j in range(len(X_mats)) if j != test_idx])
                    y_train = np.concatenate([y[j] for j in range(len(y)) if j != test_idx])
                    
                    mdl = Ridge(alpha=1.0).fit(X_train, y_train)
                    y_pred = mdl.predict(X_mats[test_idx])
                    
                    # Compute mean correlation across channels
                    # For multi-channel y, pairwise_correlation returns a diagonal of r's
                    r = pairwise_correlation(y[test_idx], y_pred)
                    trial_corrs.append(np.mean(np.diag(r)) if r.ndim > 1 else r)
                
                avg_r = np.mean(trial_corrs)
                if avg_r > max_r:
                    max_r = avg_r
                    best_alpha = alpha
            
            self.feature_alphas_[current_feat] = best_alpha

        # Final fit on all data
        final_X = self._prepare_matrix(all_features_data, feature_order, self.feature_alphas_)
        self.model_ = Ridge(alpha=1.0).fit(np.concatenate(final_X), np.concatenate(y))
        return self

    @property
    def coef_(self):
        """
        TRF weights of shape (n_targets, n_features_total, n_lags).
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        return self.model_.coef_.reshape(self.n_targets_, -1, self._ndelays)

    def predict(self, data):
        """
        Returns predictions for each trial in data.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        feat_data_list = []
        for feat in self.feature_order_:
            fd, _ = _parse_outstruct_args(data, feat)
            feat_data_list.append(fd)
            
        X_mats = self._prepare_matrix(feat_data_list, self.feature_order_, self.feature_alphas_)
        return [self.model_.predict(x) for x in X_mats]