import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import aviary as av

from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors


def test_convert_vel_to_pm():
    # Set up Solar position and motion.
    sun_xyz = [-8.122, 0, 0] * u.kpc
    sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s

    galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                        galcen_v_sun=sun_vxyz,
                                        z_sun=0*u.pc)

    # Rotation matrix from Galactocentric to ICRS
    R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

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


test_convert_vel_to_pm()
