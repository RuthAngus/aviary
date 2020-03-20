import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np

from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors


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


def get_icrs_from_galactocentric(xyz, vxyz, R_gal, sun_xyz, sun_vxyz):
    dx = xyz - sun_xyz
    dv = vxyz - sun_vxyz

    x_icrs = coord.ICRS(
        coord.CartesianRepresentation(R_gal @ dx))

    M = get_tangent_basis(x_icrs.ra, x_icrs.dec)
    proj_dv = M @ R_gal @ dv

    pm = (proj_dv[:2] / x_icrs.distance).to(u.mas/u.yr,
                                            u.dimensionless_angles())
    rv = proj_dv[2]

    return pm, rv
