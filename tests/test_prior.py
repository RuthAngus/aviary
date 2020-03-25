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
mu_vx = np.median(df.basic_vx.values)
mu_vy = np.median(df.basic_vy.values)
mu_vz = np.median(df.basic_vz.values)
sigma_vx = 1.5*aps.median_absolute_deviation(df.basic_vx.values)
sigma_vy = 1.5*aps.median_absolute_deviation(df.basic_vy.values)
sigma_vz = 1.5*aps.median_absolute_deviation(df.basic_vz.values)
# print(mu_vx, sigma_vx)
# print(mu_vy, sigma_vy)
# print(mu_vz, sigma_vz)

df = pd.read_csv("../data/gaia_mc5_velocities.csv")
m = df.radial_velocity.values != 0
df = df.iloc[m]


def test_model():
    df0 = df.iloc[0]
    xyz, vxyz = av.simple_calc_vxyz(df0.ra, df0.dec, 1./df0.parallax,
                                    df0.pmra, df0.pmdec, df0.radial_velocity)
    print(vxyz)
    vx, vy, vz = vxyz.value
    pos = [df0.ra, df0.dec, df0.parallax]

    pm, rv = av.proper_motion_model([vx, vy, vz, np.log(1./df0.parallax)],
                                    pos)

    assert np.isclose(pm[0].value, df0.pmra, atol=1e-3)
    assert np.isclose(pm[1].value, df0.pmdec, atol=1e-3)
    assert np.isclose(rv.value, df0.radial_velocity, atol=1e-3)


def test_prior():
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


def test_mcmc():
    c = coord.SkyCoord(ra=61.342*u.deg,
                       dec=17*u.deg,
                       distance=3*u.kpc,
                       pm_ra_cosdec=4.2*u.mas/u.yr,
                       pm_dec=-7.2*u.mas/u.yr,
                       radial_velocity=17*u.km/u.s)

    test_galcen = c.transform_to(galcen_frame)

    test_pm, test_rv = av.get_icrs_from_galactocentric(
        test_galcen.data.xyz, test_galcen.velocity.d_xyz, R_gal, sun_xyz,
        sun_vxyz)

    pm = np.array([4.2, -7.2])
    pm_err = np.array([.01, .01])
    pos = np.array([61.342, 17., 1./3])
    pos_err = np.array([.1, .1, .001])
    vx = test_galcen.velocity.d_xyz[0].value
    vy = test_galcen.velocity.d_xyz[1].value
    vz = test_galcen.velocity.d_xyz[2].value

    # # Run MCMC.
    ndim, nwalkers = 4, 16
    inits = [vx, vy, vz, np.log(3)]
    p0 = np.random.randn(nwalkers, ndim)*1e-2 + inits
    sampler = emcee.EnsembleSampler(nwalkers, ndim, av.lnprob,
                                    args=(pm, pm_err, pos, pos_err))
    sampler.run_mcmc(p0, 1000, progress=True);
    flat_samples = sampler.get_chain(discard=500, flat=True)
    params_inferred = np.median(flat_samples, axis=0)

    print(inits)
    params_inferred = [161.24323368, 117.5419518, 55.55548381, 1.09888655]
    print(params_inferred)
    print(av.lnprob(inits, pm, pm_err, pos, pos_err))
    print(av.lnprob(params_inferred, pm, pm_err, pos, pos_err))



test_mcmc()
# test_lnlike()
# test_model()
