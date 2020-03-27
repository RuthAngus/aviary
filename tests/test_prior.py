import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import emcee
import corner

import astropy.stats as aps
import astropy.coordinates as coord
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors
import astropy.units as u

import aviary as av

sun_xyz = [-8.122, 0, 0] * u.kpc
sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s

galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                    galcen_v_sun=sun_vxyz,
                                    z_sun=0*u.pc)

# Pre-compute the rotation matrix to go from Galactocentric to ICRS
# (ra/dec) coordinates
R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

# Calculate prior parameters from vx, vy, vz distributions
df = pd.read_csv("../data/gaia_mc5_velocities.csv")
m = df.radial_velocity.values != 0
df = df.iloc[m]


def test_model():
    df0 = df.iloc[0]
    xyz, vxyz = av.simple_calc_vxyz(df0.ra, df0.dec, 1./df0.parallax,
                                    df0.pmra, df0.pmdec, df0.radial_velocity)
    vx, vy, vz = vxyz.value
    pos = [df0.ra, df0.dec, df0.parallax]

    pm, rv = av.proper_motion_model([vx, vy, vz, np.log(1./df0.parallax)],
                                    pos)

    assert np.isclose(pm[0].value, df0.pmra, atol=1e-3)
    assert np.isclose(pm[1].value, df0.pmdec, atol=1e-3)
    assert np.isclose(rv.value, df0.radial_velocity, atol=1e-3)


def plot_prior():
    vx = np.linspace(mu_vx - 100, mu_vx + 100, 1000)
    params = [vx, mu_vy, mu_vz, np.log(.5)]
    prior = np.exp(av.lnprior(params))
    plt.clf()
    plt.plot(vx, prior)
    plt.savefig("test")
    plt.close()


def plot_lnlike():

    df1 = df.iloc[0]
    pm = [df1.pmra, df1.pmdec]
    pm_err = [df1.pmra_error, df1.pmdec_error]
    pos = [df1.ra, df1.dec, df1.parallax]
    pos_err = [df1.ra_error, df1.dec_error, df1.parallax_error]

    vx = np.linspace(df1.basic_vx - .01, df1.basic_vx + .01, 200)

    ll, lp, pr = [], [], []
    for x in vx:
        params = [x, df1.basic_vy, df1.basic_vz, np.log(1./df1.parallax)]
        ll.append(av.lnlike_one_star(params, pm, pm_err, pos, pos_err))
        lp.append(av.lnprob(params, pm, pm_err, pos, pos_err))
        pr.append(av.lnprior(params))

    df2 = pd.read_csv("../code/all_stars_cluster.csv")
    print(df1.kepid, df2.kepid.values[4])
    # print(vx[ll == max(ll)], df1.basic_vx, df2.vx_inferred.values[4])
    plt.plot(vx, ll)
    plt.axvline(df1.basic_vx, color="C0")
    plt.axvline(vx[ll == max(ll)], ls="--", color="C1")
    # plt.axvline(df2.vx_inferred.values[4], ls="--", color="C2")
    print(df2.vx_inferred.values[4])
    plt.plot(vx, lp, ls="--")
    plt.plot(vx, pr, ls="--")
    plt.savefig("test_lnprob")


def test_similar_distributions():

    np.random.seed(42)

    # Calculate prior parameters from vx, vy, vz distributions
    df = pd.read_csv("../data/gaia_mc5_velocities.csv")
    m = df.radial_velocity.values != 0
    df = df.iloc[m]

    # Calculate covariance between velocities
    VX = np.stack((df.basic_vx.values, df.basic_vy.values,
                   df.basic_vz.values, np.log(1./df.parallax.values)), axis=0)
    mean = np.mean(VX, axis=1)
    cov = np.cov(VX)

    Nstars = 1000
    vxs, vys, vzs, lnds = np.random.multivariate_normal(mean, cov, Nstars).T

    ra = np.random.uniform(280, 300, Nstars)
    dec = np.random.uniform(36, 52, Nstars)

    pms, rvs = np.zeros((Nstars, 2)), np.zeros(Nstars)
    for i in range(Nstars):
        params = [vxs[i], vys[i], vzs[i], lnds[i]]
        pos = [ra[i], dec[i], 1./np.exp(lnds[i])]
        pm, rv = av.proper_motion_model(params, pos)
        pms[i, :] = pm.value
        rvs[i] = rv.value

    assert np.isclose(np.mean(df.radial_velocity), np.mean(rvs), atol=2)
    assert np.isclose(np.mean(df.pmra), np.mean(pms[:, 0]), atol=2)
    assert np.isclose(np.mean(df.pmdec), np.mean(pms[:, 1]), atol=2)

    assert np.isclose(np.std(df.radial_velocity), np.std(rvs), atol=2)
    assert np.isclose(np.std(df.pmra), np.std(pms[:, 0]), atol=2)
    assert np.isclose(np.std(df.pmdec), np.std(pms[:, 1]), atol=2)

    # n = 50
    # plt.hist(df.radial_velocity, n, density=True, alpha=.5)
    # plt.hist(rvs, n, density=True, alpha=.5)
    # plt.xlabel("RV")
    # plt.savefig("rv_test")
    # plt.close()

    # plt.hist(df.pmra, n, density=True, alpha=.5)
    # plt.hist(pms[:, 0], n, density=True, alpha=.5)
    # plt.xlabel("pmra")
    # plt.savefig("pmra_test")
    # plt.close()

    # plt.hist(df.pmdec, n, density=True, alpha=.5)
    # plt.hist(pms[:, 1], n, density=True, alpha=.5)
    # plt.xlabel("pmdec")
    # plt.savefig("pmdec_test")
    # plt.close()


test_model()
test_similar_distributions()
