#!/usr/bin/env python3
import numpy as np
from MDSplus import Connection
from scipy.interpolate import interp1d
conn = Connection('atlas')


# %%
shot = 171957
t0 = 3000


def get_bfields(shot, t0=3000, flux_current=False):
    conn.openTree('efit01', shot)
    mw = int(conn.get('\\mw')[0])
    mh = int(conn.get('\\mh')[0])
    time = conn.get('\\gtime')

    tind = np.argmin(abs(time - t0))
    psirz1 = np.array(conn.get(r'\psirz'))
    r_g = np.array(conn.get(r'\r'))
    z_g = np.array(conn.get(r'\z'))
    if flux_current:
        ssimag = np.array(conn.get(r'\ssimag'))[tind]
        ssibry = np.array(conn.get(r'\ssibry'))[tind]
        fpol1 = np.array(conn.get(r'\fpol'))
        fpol = fpol1[tind, :]
    bcentr = np.array(conn.get(r'\bcentr'))[tind]
    rbcent = np.array(conn.get(r'\rbcent'))[tind]

    conn.closeAllTrees()

    psirz = psirz1[tind, :, :]

    bp, br, bz = [np.empty_like(psirz) for i in range(3)]
    dpsidx, dpsidy = [np.empty_like(psirz) for i in range(2)]

    # vertical derivative of psi
    dpsidy = np.gradient(psirz, z_g, axis=1)
    # for i in range(mw):
    #     dpsidy[i, :] = np.gradient(psirz[i, :], z_g)

    # horizontal derivative of psi
    dpsidx = np.gradient(psirz, r_g, axis=0)
    # for i in range(mh):
    #     dpsidx[:, i] = np.gradient(psirz[:, i], r_g)

    # calculate Br, Bz, and Bp
    br = dpsidy / r_g[:, None]
    bz = -dpsidx / r_g[:, None]
    # for i in range(mh):
    #     br[:, i] = dpsidy[:, i] / r_g
    #     bz[:, i] = -dpsidx[:, i] / r_g

    bp = np.sqrt(br**2 + bz**2)

    # calculate toroidal field
    # vacuum field
    bt1 = bcentr * rbcent / r_g
    btvac = np.tile(np.reshape(bt1, (mw, -1)), (1, mh))
    bt = btvac.copy()

    if flux_current:
        psi_n = (psirz - ssimag) / (ssibry - ssimag)
        pn = np.linspace(0, 1, mw)
        for i in range(mh):
            func = interp1d(pn, fpol, kind='cubic')
            ind = abs(psi_n[:, i]) < 1.0
            bt[ind, i] = func(psi_n[ind, i]) / r_g[ind]
    return bt, bp, r_g, z_g
