#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Check DBS data
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from raw_data_async import get_dbs
import scipy.signal as sps
import multiprocessing as mp
from numpy.fft import fftshift as fsh

# import os
# import xarray as xr
# import _axes_prop

# plt.switch_backend('Agg')
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


def plot_coh_spec(Scoh, freq, time) -> None:
    """Plot coherent power spectra"""
    import multiprocessing as mp
    with mp.Pool(processes=8) as p:
        p.map(plot_coh_spec_ch, range(Scoh.shape[0]))


def plot_coh_spec_ch(i):
    i_low = 10
    print(f"Plotting channel {i + 1}")
    fig = plt.figure(i + 1)
    plt.pcolormesh(time, freq[i_low:], np.log10(Scoh[i, i_low:, :]),
                   cmap='viridis')
    plt.title(f"coh power spec {shot} ch{i + 1} {t1}-{t2}")
    plt.xlabel('time (ms)')
    plt.ylabel('f (kHz)')
    plt.tight_layout()
    fig.savefig(f'../fig/coh_pwr_spec_{shot}_ch{i + 1}.png')
    plt.close()


def calc_quad(data, t, nperseg=1024, nfft=1024, overlap=0.5) \
        -> ('Quadrature', 'freq', 'time'):
    """Calculate quadrature"""
    fs = 1 / abs(np.mean(np.diff(t)))
    noverlap = int(nperseg * overlap)
    freq, time, Sx = sps.spectrogram(data, nperseg=nperseg, nfft=nfft,
                                     noverlap=noverlap, fs=fs,
                                     return_onesided=False)
    return Sx, freq, time + t[0]


def plot_quad_2d(Sx, freq, time) -> None:
    """Plot 2d Quadrature"""
    with mp.Pool(processes=8) as p:
        p.map(plot_quad_ch_2d, range(Sx.shape[0]))


def plot_quad_ch_2d(i) -> None:
    print(f"Plotting channel {i + 1}")
    fig = plt.figure(i + 1)
    plt.pcolormesh(time, fsh(freq), np.log10(fsh(Sx[i, :, :], axes=0)),
                   cmap='viridis')
    plt.title(f'Quadrature {shot} ch{i + 1} {t1}-{t2}ms')
    plt.xlabel('time (ms)')
    plt.ylabel('f (kHz)')
    plt.tight_layout()
    fig.savefig(f'../fig/quadrature_{shot}_ch{i + 1}.png')
    plt.close()


def plot_quad_1d(Sx, freq) -> None:
    """Plot 1d quad"""
    with mp.Pool(processes=8) as p:
        p.map(plot_quad_ch_1d, range(Sx.shape[0]))


def plot_quad_ch_1d(i) -> None:
    print(f"Plotting channel {i + 1}")
    fig = plt.figure(i + 1)
    Sm = np.mean(Sx[i, :, :], axis=-1)
    plt.semilogy(fsh(freq), fsh(Sm))
    plt.axvline(0, ls='--', c='gray')
    plt.title(f'Spectra {shot} ch{i + 1} {t1}-{t2}ms')
    plt.ylabel('Spectra')
    plt.xlabel('f (kHz)')
    plt.tight_layout()
    fig.savefig(f'../fig/quadrature_1d_{shot}_ch{i + 1}.pdf')
    plt.close()


# %%
def main():
    shot = 171956
    t1, t2 = 2500, 3400
    nperseg = 1024 * 4
    nfft = nperseg * 2
    overlap = 0.5

    data, t = get_dbs(shot, (t1, t2))
    print("Read data from file.")

    # Calculate quadrature
    Sx, freq, time = calc_quad(data, t, nperseg=nperseg, nfft=nfft,
                               overlap=overlap)
    # Plot 1D quadrature
    plot_quad_1d(Sx, freq)

    # # Plot 2D quadrature
    # plot_quad_2d(Sx, freq, time)

    # # Calculate coherent power spectra
    # Scoh, freq, time = calc_coh_spec(data, t, nperseg=nperseg, nfft=nfft,
    #                                  overlap=overlap)
    # # Plotting coherent power spectra
    # plot_coh_spec(Scoh, freq, time)


# %%
if __name__ == "__main__":
    main()
