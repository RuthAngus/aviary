import numpy as np
import aviary as av
import astropy.coordinates as coord
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors
import astropy.units as u
import time

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


test_lnlike()
test_model()
