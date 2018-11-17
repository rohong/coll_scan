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
if __name__ == "__main__":
    shot = 171956
    t1, t2 = 3000, 3500
    nperseg = 1024
    nfft = nperseg * 2
    overlap = 0.9

    data, t = get_dbs(shot, (t1, t2))
    print("Read data from file.")

    # %% Calculate coherent spectra
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

    # %% Plotting coherent power spectra
    i_low = 10
    for i in range(Scoh.shape[0]):
        plt.figure(i + 1)
        plt.pcolormesh(time + t1, freq[i_low:], np.log10(Scoh[i, i_low:, :]),
                       cmap='cubehelix_r')
        plt.title(f'coh power spec {shot} ch{i+1} {t1}-{t2}')
        plt.xlabel('time (ms)')
        plt.ylabel('f (kHz)')
        plt.tight_layout()
        plt.savefig(f'../fig/coh_pwr_spec_{shot}.png')
        plt.show()
