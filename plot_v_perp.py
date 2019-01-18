import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import readsav
from scipy.interpolate import interp1d
from _data import get_bfields, get_psi

plt.rcParams.update({'axes.formatter.use_mathtext': True,
                     'axes.formatter.limits': [-3, 4],
                     'pdf.fonttype': 42})
sns.set(style='ticks', palette='Set2')


# %%
def read_idl(shot_number: int) -> object:
    """
    Read GENRAY data
    :type shot_number: int
    :rtype: dict
    """
    idl_file: str = f"../proc_data/{shot_number}_dbs8_positions_vs_time.sav"
    idl_data: dict = readsav(idl_file)
    return idl_data


def plot_vel(shots):
    fig, axs = plt.subplots(1, 1, figsize=[4, 4])

    for shot in shots:
        # Bt, Bp, R, Z = get_bfields(shot)
        # b0, r, z = Bt[:, 32], R[:], np.array([Z[32] for i in range(len(R))])
        # psin = get_psi(shot, r, z)

        idl_dat: dict = read_idl(shot)
        t_sc = idl_dat['time']
        psi_sc = idl_dat['psiarray']
        k_sc = idl_dat['kscarray']

        # import k-f quadrature from spec files
        dat = np.load(f'../proc_data/quad_spec_{shot}_2500_3400.npz')
        Sx, freq, time = dat['Sx'], dat['freq'], dat['time']
        del dat

        # interpolate genray data to fit dbs data
        k_time = np.zeros((8, len(time)))
        psi_time = np.zeros_like(k_time)
        for i in range(8):
            foo_intp = interp1d(t_sc, k_sc[:, i], kind='nearest')
            k_time[i, :] = foo_intp(time)
            foo_intp = interp1d(t_sc, psi_sc[:, i], kind='nearest')
            psi_time[i, :] = foo_intp(time)

        f_av = np.sum(Sx * freq[None, :, None], axis=1) / np.sum(Sx, axis=1)
        vel_perp = 2 * np.pi * f_av / k_time / 1e2

        # get time mean and std values for each channel
        vel_perp_m = vel_perp.mean(axis=1)
        vel_perp_err = vel_perp.std(axis=1) / np.sqrt(vel_perp.shape[1] - 1)
        psi_m = psi_time.mean(axis=1)
        psi_err = psi_time.std(axis=1) / np.sqrt(psi_time.shape[1] - 1)

        # for i in range(8):
        #     axs.plot(psi_time[i, :], vel_perp[i, :], '.', c='gray', alpha=0.1)
        axs.errorbar(psi_m, vel_perp_m, yerr=vel_perp_err, xerr=psi_err,
                     fmt='o-', label=f'#{shot}')

    axs.set_xlabel(r'$\psi_n$')
    axs.set_title(r"$V_\perp$ (km/s)")
    axs.set(ylim=[-8, 1], xlim=[0, 1])
    axs.legend(loc=1, fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'../fig/vel_perp_{shots[0]}_{shots[-1]}.pdf',
                transparent=True)
    plt.show()


# %%
# save mean and std into csv files (for pandas)
# plot all Bt cases in one figure
if __name__ == "__main__":
    shots1 = [171956, 171957, 171958, 171959]
    shots2 = [171963, 171964, 171965]
    shots3 = [171967, 171968, 171969]
    shots_vec = [shots1, shots2, shots3]
    # shots_vec = [[171968]]
    for shots in shots_vec:
        plot_vel(shots)
