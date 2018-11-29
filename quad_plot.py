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
def calc_quad(data, t, nperseg=1024, nfft=1024, overlap=0.5) \
        -> ('Quadrature', 'freq', 'time'):
    """Calculate quadrature"""
    fs = 1 / abs(np.mean(np.diff(t)))
    noverlap = int(nperseg * overlap)
    freq, time, Sx = sps.spectrogram(data, nperseg=nperseg, nfft=nfft,
                                     noverlap=noverlap, fs=fs,
                                     return_onesided=False)
    return Sx, freq, time + t[0]


def plot_quad_2d(Sx, freq, time, shot) -> None:
    """Plot 2d Quadrature"""
    print(f"Plotting {shot} quadrature 2d")
    fig, axs = plt.subplots(4, 2, sharex='col', figsize=(8, 8))
    for i, ax in enumerate(axs.flatten(order='F')):
        ax.pcolormesh(time, fsh(freq), np.log10(fsh(Sx[i, :, :], axes=0)),
                      cmap='viridis')
        ax.set_ylabel('f (kHz)')
        ax.set_title(f'Quadrature {shot} ch{i + 1}')
    plt.tight_layout()
    fig.savefig(f'../fig/quadrature_2d_{shot}.png', dpi=300, transparent=True)
    plt.close()


def plot_quad_1d(Sx, freq, time, shot) -> None:
    print(f"Plotting {shot} quadrature 1d")
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='all', figsize=(8, 8))
    for i, ax in enumerate(axs.flatten(order='F')):
        Sm = np.mean(Sx[i, :, :], axis=-1)
        fav = np.sum(Sm * freq) / np.sum(Sm)
        ax.semilogy(fsh(freq), fsh(Sm))
        ax.axvline(0, ls='--', c='gray')
        ax.axvline(fav, ls='--', c='C3')
        ax.set(title=f'Spec {shot} ch{i + 1} f={fav:.0f}kHz')
    plt.tight_layout()
    fig.savefig(f'../fig/quadrature_1d_{shot}.png', dpi=300, transparent=True)
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
    Sx, freq, time = calc_quad(data, t, nperseg=nperseg, nfft=nfft,
                               overlap=overlap)
    plot_quad_2d(Sx, freq, time, shot)
    plot_quad_1d(Sx, freq, time, shot)


# %%
if __name__ == "__main__":
    # import concurrent.futures as cof
    import multiprocessing as mp

    shots = [171956, 171957, 171958, 171959,
             171963, 171964, 171965,
             171967, 171968, 171969]
    with mp.Pool(processes=len(shots)) as p:
        p.map(plots, shots)
