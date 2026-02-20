import numpy as np
import copy
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from mne.decoding.receptive_field import _delay_time_series
from ..utils import _parse_outstruct_args

def pairwise_correlation(A, B):
    """
    Computes Pearson correlation coefficient between corresponding columns of A and B.
    Works for 1D vectors (returns scalar) and 2D matrices (returns correlation matrix).
    
    Parameters
    ----------
    A : np.ndarray
        First array (time, channels)
    B : np.ndarray
        Second array (time, channels)
        
    Returns
    -------
    corr : float or np.ndarray
        Correlation(s). If 2D, the diagonal of the resulting matrix represents 
        the channel-wise correlations.
    """
    am = A - np.mean(A, axis=0)
    bm = B - np.mean(B, axis=0)
    
    # Use np.dot to handle both 1D and 2D cases
    coscale = np.dot(am.T, bm)
    a_ss = np.power(np.linalg.norm(am, axis=0), 2)
    b_ss = np.power(np.linalg.norm(bm, axis=0), 2)
    
    # For 1D inputs, am.T @ bm is a scalar. For 2D, we normalize by the outer product of norms.
    if np.isscalar(coscale):
        return coscale / np.sqrt(a_ss * b_ss + 1e-15)
    else:
        return coscale / np.sqrt(np.outer(a_ss, b_ss) + 1e-15)

class BandedTRF(BaseEstimator):
    """
    Iterative Banded Ridge TRF model. 
    
    Fits features sequentially in bands. For each band, the regularization (alpha) 
    is optimized via leave-one-trial-out cross-validation using coefficient averaging 
    for computational efficiency.
    
    Parameters
    ----------
    tmin : float
        Starting lag (seconds).
    tmax : float
        Ending lag (seconds).
    sfreq : float
        Sampling frequency (Hz).
    alphas : np.ndarray, optional
        Alphas to sweep for each feature. Default is np.logspace(-2, 5, 8).
    basis_dict : dict, optional
        Dictionary mapping feature names to basis objects (must have .transform() method).
    """
    def __init__(self, tmin, tmax, sfreq, alphas=None, basis_dict=None):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.alphas = alphas if alphas is not None else np.logspace(-2, 5, 8)
        self.basis_dict = basis_dict if basis_dict is not None else {}
        self.feature_alphas_ = {}
        self.alpha_paths_ = {}
        self.feature_order_ = []
        self.model_ = None

    @property
    def _ndelays(self):
        return int(round(self.tmax * self.sfreq)) - int(round(self.tmin * self.sfreq)) + 1

    def _prepare_matrix(self, X_list, feature_names, alphas_dict):
        """Prepares design matrix list (one per trial) scaled by alpha."""
        processed_trials = []
        for trl in range(len(X_list[0])):
            mats = []
            for i, name in enumerate(feature_names):
                x = X_list[i][trl]
                if x.ndim == 1:
                    x = x[:, np.newaxis]
                
                # Apply basis expansion
                if name in self.basis_dict:
                    x = self.basis_dict[name].transform(x) 
                
                alpha = alphas_dict.get(name, 1.0)
                mats.append(x / alpha)
            
            concatenated = np.concatenate(mats, axis=1)
            delayed = _delay_time_series(concatenated, self.tmin, self.tmax, self.sfreq)
            processed_trials.append(delayed.reshape(delayed.shape[0], -1))
        return processed_trials

    def fit(self, data, feature_order, target='resp'):
        """
        Fit features iteratively using fast coefficient-averaging cross-validation.
        
        Parameters
        ----------
        data : naplib.Data
            Data object containing trials.
        feature_order : list of str
            The order in which to optimize features.
        target : str
            Field name for the response variable.
        """
        self.feature_order_ = feature_order
        _, y = _parse_outstruct_args(data, feature_order[0], target)
        self.n_targets_ = y[0].shape[1]
        
        # Pre-load features from the Data object
        all_features_data = [(_parse_outstruct_args(data, f, target)[0]) for f in feature_order]

        for i, current_feat in enumerate(feature_order):
            best_alpha = None
            max_r = -np.inf
            r_history = []
            
            for alpha in tqdm(self.alphas, desc=f"Optimizing {current_feat}", leave=False):
                temp_alphas = {**self.feature_alphas_, current_feat: alpha}
                X_mats = self._prepare_matrix(all_features_data[:i+1], feature_order[:i+1], temp_alphas)
                
                # Fast CV: Fit each trial individually
                trial_betas = []
                for trl_x, trl_y in zip(X_mats, y):
                    mdl = Ridge(alpha=1.0).fit(trl_x, trl_y)
                    trial_betas.append(mdl.coef_)

                # Leave-One-Trial-Out CV via Coefficient Averaging
                trial_corrs = []
                for test_idx in range(len(X_mats)):
                    train_indices = [j for j in range(len(trial_betas)) if j != test_idx]
                    avg_beta = np.mean([trial_betas[j] for j in train_indices], axis=0)
                    
                    # Predict using averaged weights
                    y_pred = X_mats[test_idx] @ avg_beta.T
                    
                    # Extract channel-wise correlations
                    r_mat = pairwise_correlation(y[test_idx], y_pred)
                    r = np.mean(np.diag(r_mat)) if r_mat.ndim > 1 else r_mat
                    trial_corrs.append(r)
                
                avg_r = np.mean(trial_corrs)
                r_history.append(avg_r)
                
                if avg_r > max_r:
                    max_r = avg_r
                    best_alpha = alpha
            
            self.feature_alphas_[current_feat] = best_alpha
            self.alpha_paths_[current_feat] = np.array(r_history)

        # Final fit on all data combined using the optimized alphas
        final_X = self._prepare_matrix(all_features_data, feature_order, self.feature_alphas_)
        self.model_ = Ridge(alpha=1.0).fit(np.concatenate(final_X), np.concatenate(y))
        
        # Record feature dimensions for reshaping
        self.feat_dims_ = []
        for i, name in enumerate(feature_order):
            sample = all_features_data[i][0]
            if name in self.basis_dict:
                # Assuming the basis object has a property for output dimensionality
                self.feat_dims_.append(getattr(self.basis_dict[name], 'n_components', 1))
            else:
                self.feat_dims_.append(sample.shape[1] if sample.ndim > 1 else 1)

        return self

    @property
    def coef_(self):
        """
        The learned TRF weights.
        Returns
        -------
        coef : np.ndarray, shape (n_targets, n_features_total, n_lags)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before accessing coef_.")
        return self.model_.coef_.reshape(self.n_targets_, -1, self._ndelays)

    def predict(self, data, feature_names=None):
        """
        Predict response using the fitted model.
        
        Parameters
        ----------
        data : naplib.Data
            Data to predict.
        feature_names : list of str, optional
            Features to use for prediction. Defaults to all features in 
            `feature_order` used during fit.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        feats = feature_names if feature_names else self.feature_order_
        feat_data_list = [(_parse_outstruct_args(data, f)[0]) for f in feats]
            
        X_mats = self._prepare_matrix(feat_data_list, feats, self.feature_alphas_)
        return [self.model_.predict(x) for x in X_mats]