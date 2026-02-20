import numpy as np
import copy
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import ttest_1samp
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from mne.decoding.receptive_field import _delay_time_series
from ..stats import pairwise_correlation
from ..utils import _parse_outstruct_args

class BandedTRF(BaseEstimator):
    r"""
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
        Dictionary mapping feature names to basis objects.
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
        self.target_ = None

    @property
    def _ndelays(self):
        return int(round(self.tmax * self.sfreq)) - int(round(self.tmin * self.sfreq)) + 1

    @property
    def coef_(self):
        """
        Reshaped coefficients of shape (n_targets, n_features, n_delays).
        Assumes each feature has the same number of delays (tmin to tmax).
        Note: Only works if feat_dims_ are all 1. For multi-dim features, 
        this would require more complex indexing.
        """
        if self.model_ is None:
            return None
        
        # self.model_.coef_ is (n_targets, n_features_total)
        n_targets = self.model_.coef_.shape[0]
        n_feats = len(self.feature_order_)
        
        # Reshape to (n_targets, n_features, n_delays)
        # This works because _prepare_matrix concatenates features before delaying
        return self.model_.coef_.reshape(n_targets, n_feats, self._ndelays)
        # return self.model_.coef_.reshape(n_targets, self._ndelays, n_feats).transpose(0, 2, 1)

    def _prepare_matrix(self, X_list, feature_names, alphas_dict):
        """
        X_list is a list of lists: [feature_1_trials, feature_2_trials, ...]
        """
        processed_trials = []
        n_trials = len(X_list[0])
        
        for trl in range(n_trials):
            mats = []
            for i, name in enumerate(feature_names):
                x = X_list[i][trl]
                
                if np.isscalar(x):
                    continue 
                if x.ndim == 1:
                    x = x[:, np.newaxis]
                
                if name in self.basis_dict:
                    x = self.basis_dict[name].transform(x) 
                
                # Apply the band-specific scaling (Banded Ridge trick)
                alpha = alphas_dict.get(name, 1.0)
                mats.append(x / np.sqrt(alpha))
            
            if not mats:
                raise ValueError("No features were successfully processed.")
                
            concatenated = np.concatenate(mats, axis=1)
            delayed = _delay_time_series(concatenated, self.tmin, self.tmax, self.sfreq)
            processed_trials.append(delayed.reshape(delayed.shape[0], -1))
        return processed_trials

    def fit(self, data, feature_order, target='resp'):
        self.feature_order_ = feature_order
        self.target_ = target
        
        # Parse targets and all features into lists of trials
        y = _parse_outstruct_args(data, target)
        self.n_targets_ = y[0].shape[1]
        
        # Pre-parse all features once
        all_features_data = [_parse_outstruct_args(data, f) for f in feature_order]

        for i, current_feat in enumerate(feature_order):
            best_alpha = None
            max_r = -np.inf
            r_history = []
            
            for alpha in tqdm(self.alphas, desc=f"Optimizing {current_feat}", leave=False):
                temp_alphas = {**self.feature_alphas_, current_feat: alpha}
                # Slice the pre-parsed list of trial-lists
                X_mats = self._prepare_matrix(all_features_data[:i+1], feature_order[:i+1], temp_alphas)
                
                trial_betas = []
                for trl_x, trl_y in zip(X_mats, y):
                    mdl = Ridge(alpha=1.0).fit(trl_x, trl_y)
                    trial_betas.append(mdl.coef_)

                trial_corrs = []
                for test_idx in range(len(X_mats)):
                    train_indices = [j for j in range(len(trial_betas)) if j != test_idx]
                    avg_beta = np.mean([trial_betas[j] for j in train_indices], axis=0)
                    y_pred = X_mats[test_idx] @ avg_beta.T
                    
                    # New pairwise_correlation returns 1D array of correlations per channel
                    r_per_channel = pairwise_correlation(y[test_idx], y_pred)
                    trial_corrs.append(np.mean(r_per_channel))
                
                avg_r = np.mean(trial_corrs)
                r_history.append(avg_r)
                if avg_r > max_r:
                    max_r, best_alpha = avg_r, alpha
            
            self.feature_alphas_[current_feat] = best_alpha
            self.alpha_paths_[current_feat] = np.array(r_history)

        # Final fit
        final_X = self._prepare_matrix(all_features_data, feature_order, self.feature_alphas_)
        self.model_ = Ridge(alpha=1.0).fit(np.concatenate(final_X), np.concatenate(y))
        
        # Record feature dimensions
        self.feat_dims_ = []
        for i, name in enumerate(feature_order):
            x_sample = all_features_data[i][0]
            if x_sample.ndim == 1: x_sample = x_sample[:, None]
            if name in self.basis_dict:
                x_sample = self.basis_dict[name].transform(x_sample)
            self.feat_dims_.append(x_sample.shape[1])

        return self

    def predict(self, data, feature_names=None):
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        requested_features = feature_names if feature_names else self.feature_order_
        
        feat_data_list = [_parse_outstruct_args(data, f) for f in requested_features]
        X_mats = self._prepare_matrix(feat_data_list, requested_features, self.feature_alphas_)
        
        if feature_names is not None:
            preds = []
            full_coef = self.model_.coef_ 
            mask = np.zeros(full_coef.shape[1], dtype=bool)
            current_col = 0
            for i, f in enumerate(self.feature_order_):
                num_cols = self.feat_dims_[i] * self._ndelays
                if f in requested_features:
                    mask[current_col : current_col + num_cols] = True
                current_col += num_cols
            
            sliced_coef = full_coef[:, mask]
            for x_trl in X_mats:
                preds.append(x_trl @ sliced_coef.T + self.model_.intercept_)
            return preds
        
        return [self.model_.predict(x) for x in X_mats]

    def summary(self, data, channel=None):
        r"""
        Generate a statistical summary of the fitted BandedTRF model.
        """
        if not hasattr(self, 'feature_alphas_'):
            raise ValueError("Model must be fitted before calling summary.")

        resp_list = _parse_outstruct_args(data, self.target_) 
        
        n_trials = len(resp_list)
        n_channels = resp_list[0].shape[1]
        n_features = len(self.feature_order_)
        r_tensor = np.zeros((n_trials, n_channels, n_features))
        
        current_features = []
        for f_idx, feat in enumerate(self.feature_order_):
            current_features.append(feat)
            preds = self.predict(data, feature_names=current_features)
            for t_idx in range(n_trials):
                # Using new pairwise_correlation which returns per-channel values
                r_tensor[t_idx, :, f_idx] = pairwise_correlation(resp_list[t_idx], preds[t_idx])

        dr_tensor = np.diff(r_tensor, axis=2, prepend=0)

        if channel is not None:
            r_report, dr_report = r_tensor[:, channel, :], dr_tensor[:, channel, :]
            ch_label = f"Channel {channel}"
        else:
            r_report, dr_report = np.mean(r_tensor, axis=1), np.mean(dr_tensor, axis=1)
            ch_label = "Global Mean (All Channels)"

        summary_results = []
        for f_idx, feat in enumerate(self.feature_order_):
            # t-test across trials
            if channel is not None:
                sample = dr_report[:, f_idx]
            else:
                sample = np.mean(dr_tensor[:, :, f_idx], axis=1)
                
            _, p_val = ttest_1samp(sample, 0, alternative='greater')
            summary_results.append({
                'Feature': feat,
                'Total R': np.mean(r_report[:, f_idx]),
                'Delta R': np.mean(dr_report[:, f_idx]),
                'Alpha': self.feature_alphas_[feat],
                'p-value': p_val,
            })

        df = pd.DataFrame(summary_results).set_index('Feature')
        print(f"\nBandedTRF Summary | {ch_label}\n" + "-" * 70)
        print(df.to_string(formatters={'Total R': '{:,.4f}'.format, 'Delta R': '{:,.4f}'.format, 'Alpha': '{:,.2e}'.format}))
        return df