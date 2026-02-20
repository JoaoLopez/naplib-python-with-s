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
    
    The model iteratively solves for the optimal $\alpha_b$ for each band $b$ 
    by maximizing the cross-validated correlation:
    
    .. math::
        \rho = \text{corr}(y, \sum_{b=1}^{B} X_b \beta_b(\alpha_b))

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
        _, y = _parse_outstruct_args(data, feature_order[0], target)
        self.n_targets_ = y[0].shape[1]
        
        all_features_data = [(_parse_outstruct_args(data, f, target)[0]) for f in feature_order]

        for i, current_feat in enumerate(feature_order):
            best_alpha = None
            max_r = -np.inf
            r_history = []
            
            for alpha in tqdm(self.alphas, desc=f"Optimizing {current_feat}", leave=False):
                temp_alphas = {**self.feature_alphas_, current_feat: alpha}
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
                    
                    r_mat = pairwise_correlation(y[test_idx], y_pred)
                    r = np.mean(np.diag(r_mat))
                    trial_corrs.append(r)
                
                avg_r = np.mean(trial_corrs)
                r_history.append(avg_r)
                
                if avg_r > max_r:
                    max_r = avg_r
                    best_alpha = alpha
            
            self.feature_alphas_[current_feat] = best_alpha
            self.alpha_paths_[current_feat] = np.array(r_history)

        final_X = self._prepare_matrix(all_features_data, feature_order, self.feature_alphas_)
        self.model_ = Ridge(alpha=1.0).fit(np.concatenate(final_X), np.concatenate(y))
        
        # Record feature dimensions for slicing during prediction
        self.feat_dims_ = []
        temp_prep = self._prepare_matrix([[f[0]] for f in all_features_data], feature_order, self.feature_alphas_)
        # Logic to extract how many columns each feature occupies in the final matrix
        current_col = 0
        for i, name in enumerate(feature_order):
            # This accounts for basis expansion and lags
            x_sample = all_features_data[i][0]
            if x_sample.ndim == 1: x_sample = x_sample[:, None]
            if name in self.basis_dict:
                x_sample = self.basis_dict[name].transform(x_sample)
            self.feat_dims_.append(x_sample.shape[1])

        return self

    @property
    def coef_(self):
        if self.model_ is None:
            raise ValueError("Model must be fitted before accessing coef_.")
        return self.model_.coef_.reshape(self.n_targets_, -1, self._ndelays)

    def predict(self, data, feature_names=None):
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        requested_features = feature_names if feature_names else self.feature_order_
        feat_data_list = [_parse_outstruct_args(data, f)[0] for f in requested_features]
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

        n_trials = len(data)
        n_channels = data[0]['resp'].shape[1]
        n_features = len(self.feature_order_)
        r_tensor = np.zeros((n_trials, n_channels, n_features))
        
        current_features = []
        for f_idx, feat in enumerate(self.feature_order_):
            current_features.append(feat)
            preds = self.predict(data, feature_names=current_features)
            for t_idx in range(n_trials):
                r_tensor[t_idx, :, f_idx] = np.diag(pairwise_correlation(data[t_idx]['resp'], preds[t_idx]))

        dr_tensor = np.diff(r_tensor, axis=2, prepend=0)

        if channel is not None:
            r_report, dr_report = r_tensor[:, channel, :], dr_tensor[:, channel, :]
            ch_label = f"Channel {channel}"
        else:
            r_report, dr_report = np.mean(r_tensor, axis=1), np.mean(dr_tensor, axis=1)
            ch_label = "Global Mean (All Channels)"

        summary_results = []
        for f_idx, feat in enumerate(self.feature_order_):
            _, p_val = ttest_1samp(dr_report[:, f_idx], 0, alternative='greater')
            summary_results.append({
                'Feature': feat,
                'Total R': np.mean(r_report[:, f_idx]),
                'Delta R': np.mean(dr_report[:, f_idx]),
                'Alpha': self.feature_alphas_[feat],
                'p-value': p_val
            })

        df = pd.DataFrame(summary_results).set_index('Feature')
        print(f"\nBandedTRF Summary | {ch_label}\n" + "-" * 70)
        print(df.to_string(formatters={'Total R': '{:,.4f}'.format, 'Delta R': '{:,.4f}'.format, 'Alpha': '{:,.2e}'.format}))
        return df