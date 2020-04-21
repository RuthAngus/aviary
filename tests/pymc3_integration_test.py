import numpy as np
import pandas as pd

import pymc3 as pm
import corner
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

import aviary as av
import starspot as ss


def test_pymc3_gets_reasonable_results():

    # Load data.
    df0 = pd.read_csv("../data/gaia_mc5_velocities.csv")
    m = df0.radial_velocity.values != 0
    m &= np.isfinite(df0.basic_vx.values)
    m &= np.isfinite(df0.basic_vy.values)
    m &= np.isfinite(df0.basic_vz.values)
    df1 = df0.iloc[m]
    df = df1.iloc[0]

    pos = [df["ra"], df["dec"], df["parallax"]]
    pos_err = [df["ra_error"], df["dec_error"], df["parallax_error"]]
    proper = [df["pmra"], df["pmdec"]]
    proper_err = [df["pmra_error"], df["pmdec_error"]]

    # Now move star so it's near the Galactic pole.
    c = SkyCoord('12h51.4m', '+27.13',  unit=(u.hourangle, u.deg),
                 frame='icrs')
    pos[0] = c.ra.value
    pos[1] = c.dec.value

    mu, cov = av.get_prior()
    trace = av.run_pymc3_model(pos, pos_err, proper, proper_err, mu, cov)
    samples = pm.trace_to_dataframe(trace)
    # fig = corner.corner(samples.iloc[:, :4]);
    # fig.savefig("pymc3_corner")


def test_get_prior():
    mu, cov = av.get_prior()
    df = pd.read_csv("../aviary/mc_san_gaia_lam.csv", low_memory=False)

    lnD = np.log(1./df.parallax)
    finite = np.isfinite(df.vx.values) & np.isfinite(df.vy.values) \
        & np.isfinite(df.vz.values) & np.isfinite(lnD)
    nsigma = 3
    mx = ss.sigma_clip(df.vx.values[finite], nsigma=nsigma)
    my = ss.sigma_clip(df.vy.values[finite], nsigma=nsigma)
    mz = ss.sigma_clip(df.vz.values[finite], nsigma=nsigma)
    md = ss.sigma_clip(lnD[finite], nsigma=nsigma)
    m = mx & my & mz & md

    mu, cov = av.mean_and_var(df.vx.values[finite][m],
                              df.vy.values[finite][m],
                              df.vz.values[finite][m], lnD[finite][m])

    # plt.hist(df.vz.values[finite][m], 50)
    # plt.axvline(mu[2], color="k", lw=2)
    # plt.axvline(mu[2] + np.sqrt(cov[2, 2]), color="k", ls="--", lw=2)
    # plt.axvline(mu[2] - np.sqrt(cov[2, 2]), color="k", ls="--", lw=2)
    # plt.axvline(np.mean(df.vz.values[finite][m]))
    # plt.axvline(np.mean(df.vz.values[finite][m])
    #             + np.std(df.vz.values[finite][m]))
    # plt.axvline(np.mean(df.vz.values[finite][m])
    #             - np.std(df.vz.values[finite][m]))
    # plt.savefig("test")

    assert np.isclose(mu[0], np.mean(df.vx.values[finite][m]), atol=.1)
    assert np.isclose(mu[1], np.mean(df.vy.values[finite][m]), atol=.1)
    assert np.isclose(mu[2], np.mean(df.vz.values[finite][m]), atol=.1)
    assert np.isclose(np.sqrt(cov[0, 0]), np.std(df.vx.values[finite][m]),
                      atol=.1)
    assert np.isclose(np.sqrt(cov[1, 1]), np.std(df.vy.values[finite][m]),
                      atol=.1)
    assert np.isclose(np.sqrt(cov[2, 2]), np.std(df.vz.values[finite][m]),
                      atol=.1)

test_pymc3_gets_reasonable_results()
test_get_prior()
