import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np

from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors

# Set up Solar position and motion.
sun_xyz = [-8.122, 0, 0] * u.kpc
sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s

galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                    galcen_v_sun=sun_vxyz,
                                    z_sun=0*u.pc)

# Rotation matrix from Galactocentric to ICRS
R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)


def get_tangent_basis(ra, dec):
    """
    row vectors are the tangent-space basis at (alpha, delta, r)
    ra, dec in radians
    """
    M = np.array([
        [-np.sin(ra), np.cos(ra), 0.],
        [-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)],
        [np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]
    ])
    return M


def get_icrs_from_galactocentric(xyz, vxyz):
    dx = xyz - sun_xyz
    dv = vxyz - sun_vxyz

    # galcen_frame, R_gal = get_rotation_matrix()

    x_icrs = coord.ICRS(
        coord.CartesianRepresentation(R_gal @ dx))

    M = get_tangent_basis(x_icrs.ra, x_icrs.dec)
    proj_dv = M @ R_gal @ dv

    pm = (proj_dv[:2] / x_icrs.distance).to(u.mas/u.yr,
                                            u.dimensionless_angles())
    rv = proj_dv[2]

    return pm, rv
