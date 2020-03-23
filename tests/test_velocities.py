import numpy as np
import pandas as pd

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors

import aviary as av


def test_calc_vxyz():

    # Set up Solar position and motion.
    sun_xyz = [-8.122, 0, 0] * u.kpc
    sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s

    galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                        galcen_v_sun=sun_vxyz,
                                        z_sun=0*u.pc)

    # Rotation matrix from Galactocentric to ICRS
    R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

    ra, dec, D = 61.342, 17, 3
    pmra, pmdec, rv = 4.2, -7.2, 17

    c = coord.SkyCoord(ra=ra*u.deg,
                       dec=dec*u.deg,
                       distance=D*u.kpc,
                       pm_ra_cosdec=pmra*u.mas/u.yr,
                       pm_dec=pmdec*u.mas/u.yr,
                       radial_velocity=ra*u.km/u.s)

    test_galcen = c.transform_to(galcen_frame)

    print(test_galcen.velocity.d_xyz)

    # Load mcquillan-gaia
    df0 = pd.read_csv("../data/gaia_mc5.csv")
    m = np.isfinite(df0.radial_velocity.values)

    # Downsize data frame to first two rows
    df = df0.iloc[m][:2]

    # Copy 1st row and replace values
    new_row = df.iloc[0].copy(deep=True)
    new_row.loc["ra"] = ra
    new_row.loc["dec"] = dec
    new_row.loc["pmra"] = pmra
    new_row.loc["pmdec"] = pmdec
    new_row.loc["parallax"] = 1./D
    new_row.loc["radial_velocity"] = rv
    df1 = df.append(new_row)

    # Calculate velocities
    vx, vx_err, vy, vy_err, vz, vz_err = av.calc_vxyz(df1)
    print(vx[-1], vy[-1], vz[-1])


test_calc_vxyz()
