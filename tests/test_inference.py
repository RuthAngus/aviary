import numpy as np
import pandas as pd
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


def test_lnlike():

    # Test the function by transforming a coordinate to galactocentric
    # coords and back:
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

    assert u.allclose(test_pm[0], c.pm_ra_cosdec)
    assert u.allclose(test_pm[1], c.pm_dec)
    assert u.allclose(test_rv, c.radial_velocity)

    pm = np.array([4.2, -7.2])
    pm_err = np.array([.01, .01])
    pos = np.array([61.342, 17., 1./3])
    pos_err = np.array([.1, .1, .001])
    vx = test_galcen.velocity.d_xyz[0].value
    vy = test_galcen.velocity.d_xyz[1].value
    vz = test_galcen.velocity.d_xyz[2].value

    params = [vx, vy, vz, np.log(3.)]
    best_lnlike = av.lnlike_one_star(params, pm, pm_err, pos, pos_err)

    params = [vx, vy, vz, np.log(10.)]
    wrong_dist = av.lnlike_one_star(params, pm, pm_err, pos, pos_err)

    params = [vx, vy, vz, np.log(3.)]
    wrong_pm = av.lnlike_one_star(params, pm+.5, pm_err, pos, pos_err)

    assert best_lnlike > wrong_dist
    assert best_lnlike > wrong_pm

    av.lnprior(params)


def test_model():

    # Test the function by transforming a coordinate to galactocentric
    # coords and back:
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

    vx = test_galcen.velocity.d_xyz[0].value
    vy = test_galcen.velocity.d_xyz[1].value
    vz = test_galcen.velocity.d_xyz[2].value
    params = [vx, vy, vz, np.log(3)]
    pos = [61.342, 17, 1./3]
    pm_from_v, rv_from_v = av.proper_motion_model(params, pos)

    assert np.isclose(4.2, test_pm[0].value, atol=1e-4)
    assert np.isclose(4.2, pm_from_v[0].value, atol=1e-4)
    assert np.isclose(-7.2, test_pm[1].value, atol=1e-4)
    assert np.isclose(-7.2, pm_from_v[1].value, atol=1e-4)
    assert np.isclose(17, test_rv.value, atol=1e-4)
    assert np.isclose(17, rv_from_v.value, atol=1e-4)

    df0 = pd.read_csv("../data/gaia_mc5_velocities.csv")
    m = np.isfinite(df0.vz.values)
    df = df0.iloc[m]

    i = 0
    df1 = df.iloc[i]

    ra, dec, plx = df1.ra, df1.dec, df1.parallax
    ra_err, dec_err, plx_err = df1.ra_error, df1.dec_error, df1.parallax_error
    pmra, pmdec, rv = df1.pmra, df1.pmdec, df1.radial_velocity
    pmra_err, pmdec_err, rv_err = df1.pmra_error, df1.pmdec_error, \
            df1.radial_velocity_error

    #Test the function by transforming a coordinate to galactocentric and back:
    c = coord.SkyCoord(ra=ra*u.deg,
                    dec=dec*u.deg,
                    distance=(1./plx)*u.kpc,
                    pm_ra_cosdec=pmra*u.mas/u.yr,
                    pm_dec=pmdec*u.mas/u.yr,
                    radial_velocity=rv*u.km/u.s)

    test_galcen = c.transform_to(galcen_frame)
    test_pm, test_rv = av.get_icrs_from_galactocentric(
        test_galcen.data.xyz, test_galcen.velocity.d_xyz, R_gal, sun_xyz,
        sun_vxyz)

    vx = test_galcen.velocity.d_xyz[0].value
    vy = test_galcen.velocity.d_xyz[1].value
    vz = test_galcen.velocity.d_xyz[2].value

    xyz, vxyz = av.simple_calc_vxyz(ra, dec, 1./plx, pmra, pmdec, rv)
    _vx, _vy, _vz = vxyz

    params = [vx, vy, vz, np.log(1./plx)]
    pos = [ra, dec, 1./plx]
    pm_from_v, rv_from_v = av.proper_motion_model(params, pos)
    vpars = [df1.basic_vx, df1.basic_vy, df1.basic_vz,
             np.log(1./df1.parallax)]
    pm2, rv2 = av.proper_motion_model(vpars, pos)

    assert np.isclose(pmra, test_pm[0].value, atol=1e-4)
    assert np.isclose(pmra, pm_from_v[0].value, atol=1e-4)
    assert np.isclose(pmra, pm2[0].value, atol=1e-4)
    assert np.isclose(pmdec, test_pm[1].value, atol=1e-4)
    assert np.isclose(pmdec, pm_from_v[1].value, atol=1e-4)
    assert np.isclose(pmdec, pm2[1].value, atol=1e-4)
    assert np.isclose(rv, test_rv.value, atol=1e-4)
    assert np.isclose(rv, rv_from_v.value, atol=1e-4)
    assert np.isclose(rv, rv2.value, atol=1e-4)


test_lnlike()
test_model()
