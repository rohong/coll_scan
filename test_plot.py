#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Check DBS data
"""
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from raw_data_async import get_dbs
import scipy.signal as sps
import os

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
    # print(Sx.shape, freq.shape)
    Scoh = Sa * coh[:, :, None]
    return Scoh, freq, time + t[0]


def plot_coh_spec(Scoh, freq, time) -> None:
    """Plot coherent power spectra"""
    import multiprocessing as mp
    with mp.Pool(processes=8) as p:
        p.map(plot_coh_spec_ch, range(Scoh.shape[0]))


def plot_coh_spec_ch(i):
    i_low = 10
    print(f"Plotting channel {i+1}")
    plt.figure(i + 1)
    plt.pcolormesh(time, freq[i_low:], np.log10(Scoh[i, i_low:, :]),
                   cmap='viridis')
    plt.title(f'coh power spec {shot} ch{i+1} {t1}-{t2}')
    plt.xlabel('time (ms)')
    plt.ylabel('f (kHz)')
    plt.tight_layout()
    plt.savefig(f'../fig/coh_pwr_spec_{shot}_ch{i+1}.png')
    plt.show()


# %%
if __name__ == "__main__":
    shot = 171956
    t1, t2 = 3000, 3500
    nperseg = 1024 * 2
    nfft = nperseg * 2
    overlap = 0.5

    data, t = get_dbs(shot, (t1, t2))
    print("Read data from file.")

    # %%
    Scoh, freq, time = calc_coh_spec(data, t, nperseg=nperseg, nfft=nfft,
                                     overlap=overlap)
    # %% Plotting coherent power spectra
    plot_coh_spec(Scoh, freq, time)
