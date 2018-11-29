#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Plot quadrature 1d and 2d into files
"""
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from raw_data_async import get_dbs
import scipy.signal as sps
import multiprocessing as mp
from numpy.fft import fftshift as fsh

plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})
sns.set(style='ticks', palette='Set2')


# %%
def calc_coh_spec(data, t, nperseg=1024, nfft=1024, overlap=0.5) \
        -> ('Coherent spectra', 'freq', 'time'):
    """Calculate coherent spectra"""
    fs = 1 / abs(np.mean(np.diff(t)))
    da = data.real
    db = data.imag
    noverlap = int(nperseg * overlap)
    freq, time, Sa = sps.spectrogram(da, nperseg=nperseg, nfft=nfft,
                                     noverlap=noverlap, fs=fs)
    freq, coh = sps.coherence(da, db, nperseg=nperseg, nfft=nfft,
                              noverlap=noverlap, fs=fs)
    Scoh = Sa * coh[:, :, None]
    return Scoh, freq, time + t[0]


def plot_coh_spec_2d(Scoh, freq, time, shot) -> None:
    print(f"Plotting 2d coherent power spec shot {shot}.")
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='all', figsize=(8, 8))
    for i, ax in enumerate(axs.flatten(order='F')):
        ax.pcolormesh(time, freq, np.log10(Scoh[i, :, :]),
                      cmap='viridis')
        ax.set(title=f'coh spec {shot} ch{i + 1}')
    plt.tight_layout()
    fig.savefig(f'../fig/coh_pwr_spec_{shot}_2d.png', dpi=300,
                transparent=True)
    plt.close()


def plot_coh_spec_1d(Scoh, freq, time, shot) -> None:
    print(f"Plotting 1d coherent power spec shot {shot}.")
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='all', figsize=(8, 8))
    for i, ax in enumerate(axs.flatten(order='F')):
        Sm = np.mean(Scoh[i, :, :], axis=-1)
        fav = np.sum(freq * Sm) / np.sum(Sm)
        ax.semilogy(freq, Sm)
        ax.axvline(fav, ls='--', c='C3')
        ax.set(title=f'coh spec {shot} ch{i + 1} f={fav:.0f}kHz')
    plt.tight_layout()
    fig.savefig(f'../fig/coh_pwr_spec_{shot}_1d.png', dpi=300,
                transparent=True)
    plt.close()


def load_data(shot):
    fname = f"../rawdata/DBS_{shot}.nc"
    t1, t2 = 2500, 3500
    data, t = get_dbs(shot, (t1, t2))
    print(f"Read {fname} data from file.")
    return data, t


def plots(shot):
    nperseg = 1024 * 8
    nfft = nperseg * 1
    overlap = 0.5
    data, t = load_data(shot)
    Sx, freq, time = calc_coh_spec(data, t, nperseg=nperseg, nfft=nfft,
                                   overlap=overlap)
    plot_coh_spec_2d(Sx, freq, time, shot)
    plot_coh_spec_1d(Sx, freq, time, shot)


# %%
if __name__ == "__main__":
    # import concurrent.futures as cof
    import multiprocessing as mp

    # plots(171956)
    shots = [171956, 171957, 171958, 171959,
             171963, 171964, 171965,
             171967, 171968, 171969]
    with mp.Pool(processes=len(shots)) as p:
        p.map(plots, shots)
