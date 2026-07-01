import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.linear_model import Ridge

import naplib as nl
import sys
sys.path.insert(0, '/app')
from speedupy.speedupy import deterministic, initialize_speedupy

@deterministic
def compute_auditory_spectrogram(sound, sfreq, func_globals=None):
    return nl.features.auditory_spectrogram(sound, sfreq)


@deterministic
def fit_reconstruction_model(X_train, y_train, tmin, tmax, sfreq, func_globals=None):
    trials = []
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        trial = {'resp': x, 'reshaped_spec': y}
        trials.append(trial)
    data_train = nl.Data(trials)
    mdl = nl.encoding.TRF(tmin=tmin, tmax=tmax, sfreq=sfreq, estimator=Ridge(), show_progress=False)
    mdl.fit(data_train, X='resp', y='reshaped_spec')
    return mdl


@deterministic
def fit_reconstruction_bychannel(X_train, y_train, tmin, tmax, sfreq, func_globals=None):
    trials = []
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        trial = {'resp': x, 'spec_32': y}
        trials.append(trial)
    data_train = nl.Data(trials)
    mdl = nl.encoding.TRF(tmin=tmin, tmax=tmax, sfreq=sfreq, estimator=Ridge(), show_progress=True)
    mdl.fit(data_train, X='resp', y='spec_32')
    return mdl


def plot_reconstruction(true_data, reconstructed_data, title_true, title_pred):
    region = slice(0, 500)
    true_region  = true_data.squeeze()[region].T
    pred_region  = reconstructed_data.squeeze()[region].T
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
    ax0.imshow(true_region, aspect=3, origin='lower')
    ax0.set_title(title_true)
    ax1.imshow(pred_region, aspect=3, origin='lower')
    ax1.set_title(title_pred)
    plt.tight_layout()
    plt.show()


@initialize_speedupy
def main(tmin):
    data = nl.io.load_speech_task_data()
    print(f"Trials carregados: {len(data)}")

    data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

    t0 = time.time()
    specs = []
    for trial in data:
        sound = trial['sound']
        spec = compute_auditory_spectrogram(sound, 11025, func_globals=globals())
        specs.append(spec)
    data['spec'] = specs
    print(f"Tempo auditory_spectrogram: {time.time() - t0:.2f}s")

    aligned_specs = []
    for trial in data:
        trial_spec = trial['spec']
        trial_resp = trial['resp']
        n = trial_resp.shape[0]
        spec_resampled = resample(trial_spec, n)
        aligned_specs.append(spec_resampled)
    data['spec'] = aligned_specs

    spec_0 = data['spec'][0]
    print(f"before resampling: {spec_0.shape}")
    resample_kwargs = {'num': 32, 'axis': 1}
    data['spec_32'] = nl.array_ops.concat_apply(data['spec'], resample, function_kwargs=resample_kwargs)
    spec_32_0 = data['spec_32'][0]
    print(f"after resampling:  {spec_32_0.shape}")

    reshaped = []
    for x in data['spec_32']:
        x_reshaped = x[:, np.newaxis, :]
        reshaped.append(x_reshaped)
    data['reshaped_spec'] = reshaped

    data_train = data[1:]
    data_test  = data[:1]

    tmax  = 0
    sfreq = 100

    X_train           = data_train['resp']
    y_train_unified   = data_train['reshaped_spec']
    y_train_bychannel = data_train['spec_32']

    data_test_list = list(data_test)
    first_test     = data_test_list[0]
    test_resp      = first_test['resp']
    test_reshaped  = first_test['reshaped_spec']
    test_spec_32   = first_test['spec_32']

    t0 = time.time()
    mdl_unified = fit_reconstruction_model(
        X_train, y_train_unified, tmin, tmax, sfreq, func_globals=globals()
    )
    print(f"Tempo fit_reconstruction_model (unificado): {time.time() - t0:.2f}s")

    test_trial_unified = {'resp': test_resp, 'reshaped_spec': test_reshaped}
    data_test_unified  = nl.Data([test_trial_unified])
    reconstructed      = mdl_unified.predict(data_test_unified, X='resp')
    corr_unified       = mdl_unified.corr(data_test_unified, X='resp', y='reshaped_spec')
    corr_unified_val   = corr_unified.item()

    reconstructed_first = reconstructed[0]


    t0 = time.time()
    mdl_bychannel = fit_reconstruction_bychannel(
        X_train, y_train_bychannel, tmin, tmax, sfreq, func_globals=globals()
    )
    print(f"Tempo fit_reconstruction_bychannel (32 modelos): {time.time() - t0:.2f}s")

    test_trial_bychannel    = {'resp': test_resp, 'spec_32': test_spec_32}
    data_test_bychannel     = nl.Data([test_trial_bychannel])
    reconstructed_bychannel = mdl_bychannel.predict(data_test_bychannel, X='resp')
    corr_bychannel_raw      = mdl_bychannel.corr(data_test_bychannel, X='resp', y='spec_32')
    corr_bychannel          = corr_bychannel_raw.mean()

    reconstructed_bychannel_first = reconstructed_bychannel[0]

    title_unified = f'Reconstrução (modelo unificado), corr={corr_unified_val:.3f}'
    plot_reconstruction(test_reshaped, reconstructed_first, 'Estímulo real', title_unified)

    title_bychannel = f'Reconstrução (por canal, 32 modelos), corr={corr_bychannel:.3f}'
    plot_reconstruction(test_spec_32, reconstructed_bychannel_first, 'Estímulo real', title_bychannel)


if __name__ == '__main__':
    tmin = float(sys.argv[1])
    start = time.perf_counter()
    main(tmin)
    print(f"\nTempo total: {time.perf_counter() - start:.2f}s")