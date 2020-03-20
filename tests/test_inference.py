import numpy as np
import aviary as av
import astropy.coordinates as coord
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors
import astropy.units as u
import time

def test_lnlike():
    sun_xyz = [-8.122, 0, 0] * u.kpc
    sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s

    galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                        galcen_v_sun=sun_vxyz,
                                        z_sun=0*u.pc)

    # Pre-compute the rotation matrix to go from Galactocentric to ICRS
    # (ra/dec) coordinates
    R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

    #Test the function by transforming a coordinate to galactocentric and back:
    c = coord.SkyCoord(ra=61.342*u.deg,
                    dec=17*u.deg,
                    distance=3*u.kpc,
                    pm_ra_cosdec=4.2*u.mas/u.yr,
                    pm_dec=-7.2*u.mas/u.yr,
                    radial_velocity=17*u.km/u.s)

    test_galcen = c.transform_to(galcen_frame)

    test_pm, test_rv = av.get_icrs_from_galactocentric(
        test_galcen.data.xyz,
        test_galcen.velocity.d_xyz, R_gal, sun_xyz, sun_vxyz)

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
    start1 = time.time()
    print(av.lnlike_one_star(params, pm, pm_err, pos, pos_err))
    end1 = time.time()

    params = [vx, vy, vz, np.log(10.)]
    start2 = time.time()
    print(av.lnlike_one_star(params, pm, pm_err, pos, pos_err))
    end2 = time.time()

    params = [vx, vy, vz, np.log(3.)]
    start3 = time.time()
    print(av.lnlike_one_star(params, pm+.5, pm_err, pos, pos_err))
    end3 = time.time()
    print(end1 - start1, "s")
    print(end2 - start2, "s")
    print(end3 - start3, "s")


test_lnlike()
