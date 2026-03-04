import numpy as np
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
    """
    def __init__(self, tmin, tmax, sfreq, alphas=None):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.alphas = alphas if alphas is not None else np.logspace(-2, 5, 8)
        self.feature_alphas_ = []
        self.alpha_paths_ = []
        self.feature_order_ = []
        self.model_ = None # Will store a list of fitted Ridge models (one per trial)
        self.target_ = None
        self.scores_ = None # Shape: (n_trials, n_channels, n_features)

    @property
    def _ndelays(self):
        return int(round(self.tmax * self.sfreq)) - int(round(self.tmin * self.sfreq)) + 1
    
    @property
    def coef_(self):
        if self.model_ is None:
            raise AttributeError("BandedTRF has not been fitted yet.")
        
        n_trials = len(self.model_)
        n_feats = len(self.feature_order_)
        
        # Force coefficients to be 2D (n_targets, n_features_total)
        # This fixes the 3.8 vs 3.10 discrepancy
        trial_coefs = []
        for m in self.model_:
            c = m.coef_
            if c.ndim == 1:
                c = c[np.newaxis, :]
            trial_coefs.append(c)
            
        n_targets = trial_coefs[0].shape[0]
        all_coefs = np.stack(trial_coefs, axis=-1)
        n_feat_dim = sum(self.feat_dims_)
        
        return all_coefs.reshape(n_targets, n_feat_dim, self._ndelays, n_trials)

    def _prepare_matrix(self, X_list, alphas_list):
        processed_trials = []
        n_trials = len(X_list[0])
        
        for trl in range(n_trials):
            mats = []
            for i in range(len(X_list)):
                x = X_list[i][trl]
                
                if isinstance(x, list) and len(x) == 1:
                    x = x[0]
                
                if np.isscalar(x) or x is None:
                    continue 
                if x.ndim == 1:
                    x = x[:, np.newaxis]
                
                alpha = alphas_list[i]
                mats.append(x / np.sqrt(alpha))
            
            if not mats:
                raise ValueError("No features were successfully processed.")
                
            concatenated = np.concatenate(mats, axis=1)
            delayed = _delay_time_series(concatenated, self.tmin, self.tmax, self.sfreq)
            processed_trials.append(delayed.reshape(delayed.shape[0], -1))
        return processed_trials

    def fit(self, data=None, X=['aud'], y='resp'):
        r"""
        Fit the Iterative Banded Ridge model using leave-one-trial-out cross-validation.

        The model fits features sequentially according to `feature_order`. For each 
        new feature band, an optimal regularization parameter (alpha) is selected 
        from `self.alphas` by maximizing the average prediction correlation across 
        held-out trials.

        Parameters
        ----------
        data : naplib.Data instance, optional
            Data object containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : list of str | list of list of np.ndarrays
            Data to be used as predictor in the regression. Should be a list, 
            in which each element is a feature, corresponding to a list of trials, 
            each of which is a numpy array of shape (time, num_features).
            If a string, it must specify a list of fields of the Data
            provided in the first argument.
        y : str | list of np.ndarrays
            Data to be used as target(s) in the regression. Once arranged,
            should be of shape (time, num_targets).
            If a string, it must specify one of the fields of the Data
            provided in the first argument.

        Returns
        -------
        self : BandedTRF
            Returns the instance of the fitted model.

        Notes
        -----
        The cross-validation uses 'coefficient averaging' for efficiency. For 
        each alpha in the sweep, a model is fit to each trial individually. 
        The prediction for a held-out trial $i$ is generated using the mean 
        coefficients of all trials $j \neq i$.
        """
        if isinstance(X[0], str):
            self.feature_order_ = X
        else:
            self.feature_order_ = [chr(i+65) for i in range(len(X))]
        if isinstance(y, str):
            self.target_ = y
        else:
            self.target_ = 'target'
        
        y = _parse_outstruct_args(data, y)
        if not isinstance(y, list): y = [y]
        X = [_parse_outstruct_args(data, x) for x in X]
        
        n_trials = len(y)
        self.n_targets_ = y[0].shape[1]

        self.scores_ = np.zeros((n_trials, self.n_targets_, len(X)))
        self.feature_alphas_ = []
        self.alpha_paths = np.zeros((len(X), len(self.alphas)))

        for i in range(len(X)):
            best_alpha = None
            max_r = -np.inf
            r_history = []
            best_r_per_trial_ch = None
            
            for alpha in tqdm(self.alphas, desc=f"Optimizing {self.feature_order_[i]}", leave=False):
                temp_alphas = self.feature_alphas_ + [alpha]
                X_mats = self._prepare_matrix(X[:i+1], temp_alphas)
                
                trial_betas = [Ridge(alpha=1.0).fit(tx, ty.reshape(-1, self.n_targets_)).coef_ for tx, ty in zip(X_mats, y)]

                current_alpha_trial_r = np.zeros((n_trials, self.n_targets_))
                for test_idx in range(n_trials):
                    train_indices = [j for j in range(n_trials) if j != test_idx]
                    avg_beta = np.mean([trial_betas[j] for j in train_indices], axis=0)
                    y_pred = X_mats[test_idx] @ avg_beta.T
                    
                    # Ensure y is 2D: (samples, targets)
                    y_true = y[test_idx]
                    if y_true.ndim == 1:
                        y_true = y_true[:, np.newaxis]
                        
                    # Ensure y_pred is 2D: (samples, targets)
                    if y_pred.ndim == 1:
                        y_pred = y_pred[:, np.newaxis]

                    # This returns an array of shape (n_targets,)
                    r_values = pairwise_correlation(y_true, y_pred)
                    current_alpha_trial_r[test_idx, :] = r_values
                
                avg_r = np.nanmean(current_alpha_trial_r)
                r_history.append(avg_r)
                if avg_r > max_r or np.isclose(avg_r, max_r):
                    max_r, best_alpha = avg_r, alpha
                    best_r_per_trial_ch = current_alpha_trial_r
            
            self.feature_alphas_.append(best_alpha)
            self.alpha_paths_[i, :] = np.array(r_history)
            self.scores_[:, :, i] = best_r_per_trial_ch

        # Final fit on each trial separately
        final_X = self._prepare_matrix(X, self.feature_alphas_)
        self.model_ = [Ridge(alpha=1.0).fit(tx, ty) for tx, ty in zip(final_X, y)]
        
        self.feat_dims_ = []
        for i in range(len(X)):
            x_sample = X[i][0]
            if isinstance(x_sample, list): x_sample = x_sample[0]
            if x_sample.ndim == 1: x_sample = x_sample[:, None]
            self.feat_dims_.append(x_sample.shape[1])

        return self

    def predict(self, data=None, X=['aud']):
        """
        Predict target responses using the fitted Banded Ridge model.

        This method performs Leave-One-Trial-Out (LOTO) prediction. For each 
        trial in the input data, it averages the regression coefficients 
        from all *other* trials (fitted during training) to generate the 
        prediction for the current trial.
        
        Parameters
        ----------
        data : naplib.Data object, optional
            Data object containing data to predict from in one of the fields.
            If not given, must give the X data directly.
        X : list of str | list of list of np.ndarrays
            Data to be used as predictor in the regression. Should be a list, 
            in which each element is a feature, corresponding to a list of trials, 
            each of which is a numpy array of shape (time, num_features).
            If a list of strings, it must specify a list of fields of the Data
            provided in the first argument.
        
        Returns
        -------
        preds : list of np.ndarray
            Predicted target values for each trial. Each element is an 
            array of shape (n_samples, n_targets).

        Raises
        ------
        ValueError
            If the model has not been fitted, or if the number of trials 
            in `data` does not match the number of models in `self.model_`.

        Notes
        -----
        Because this model stores a separate fit for every trial to enable 
        efficient cross-validation, the `predict` step requires the input 
        to have a one-to-one mapping with the training trials.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        
        X = [_parse_outstruct_args(data, x) for x in X]

        X_mats = self._prepare_matrix(X, self.feature_alphas_)
        n_trials = len(X_mats)
        n_feat_dim = X_mats[0].shape[1]
        
        if n_trials != len(self.model_):
            raise ValueError(
                f"LOTO predict requires the same number of trials ({len(self.model_)}) "
                f"as used in fit. Found {n_trials} trials."
            )

        all_coefs = np.array([m.coef_ for m in self.model_])
        if all_coefs.ndim == 2:
            # Expand (trials, features) -> (trials, 1_target, features)
            all_coefs = all_coefs[:, np.newaxis, :]

        if n_feat_dim < all_coefs.shape[2]:
            all_coefs = all_coefs[:, :, :n_feat_dim]
            print('Using reduced TRF')
        elif n_feat_dim > all_coefs.shape[2]:
            raise ValueError('Too many features for trained model.')

        preds = []
        for i in range(n_trials):
            # Indices for all trials except the current one
            loto_indices = [j for j in range(n_trials) if j != i]
            
            # Average coefficients and intercepts from the other trials
            loto_coef = np.mean(all_coefs[loto_indices], axis=0)
            
            # Predict for the current trial
            preds.append(X_mats[i] @ loto_coef.T)
            
        return preds

    def summary(self, channel=None):
        r"""
        Generate a statistical report of feature contributions and model performance.

        Calculates the incremental improvement (Delta R) for each feature band 
        added to the model and performs a one-sample t-test (alternative='greater') 
        across trials to determine if the contribution is significantly greater 
        than zero.

        Parameters
        ----------
        channel : int, optional
            The specific target channel (e.g., electrode or sensor) to summarize. 
            If None (default), results are averaged across all channels.

        Returns
        -------
        df : pandas.DataFrame
            A summary table indexed by 'Feature' containing:
            - Total R: Cumulative correlation after adding this feature.
            - Delta R: Incremental correlation increase attributed to this feature.
            - Alpha: The optimized regularization parameter for the band.
            - p-value: Significance of the Delta R across trials (t-test).

        Notes
        -----
        The Delta R for the first feature is its Total R. For subsequent 
        features, Delta R is calculated as:
        $ \Delta R_{n} = R_{n} - R_{n-1} $
        
        Significant p-values suggest that the addition of a specific feature 
        band significantly improves the model's predictive power on 
        held-out data.
        """
        if self.scores_ is None:
            raise ValueError("Model must be fitted before calling summary.")

        dr_tensor = np.diff(self.scores_, axis=2, prepend=0)

        if channel is not None:
            r_report = self.scores_[:, channel, :]
            dr_report = dr_tensor[:, channel, :]
            ch_label = f"Channel {channel}"
        else:
            r_report = np.nanmean(self.scores_, axis=1)
            dr_report = np.nanmean(dr_tensor, axis=1)
            ch_label = "Global Mean (All Channels)"

        summary_results = []
        for f_idx, feat in enumerate(self.feature_order_):
            sample = dr_report[:, f_idx]
            clean_sample = sample[~np.isnan(sample)]
            if len(clean_sample) < 2 or np.all(clean_sample == clean_sample[0]):
                p_val = 1.0 if np.mean(clean_sample) <= 0 else 0.0
            else:
                t_val, p_val = ttest_1samp(clean_sample, 0, alternative='greater')
            
            summary_results.append({
                'Feature': feat,
                'Total R': np.nanmean(r_report[:, f_idx]),
                'Delta R': np.nanmean(dr_report[:, f_idx]),
                'Alpha': self.feature_alphas_[f_idx],
                't-value': t_val,
                'p-value': p_val,
            })

        df = pd.DataFrame(summary_results).set_index('Feature')
        print(f"\nBandedTRF Summary | {ch_label}\n" + "-" * 70)
        print(df.to_string(formatters={'Total R': '{:,.4f}'.format, 
                                      'Delta R': '{:,.4f}'.format, 
                                      'Alpha': '{:,.2e}'.format}))
        return df