# Functions needed for inferring velocities.

import numpy as np
import pandas as pd
import astropy.stats as aps
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.builtin_frames.galactocentric \
    import get_matrix_vectors
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import aviary as av

# Solar coords
sun_xyz = [-8.122, 0, 0] * u.kpc
sun_vxyz = [12.9, 245.6, 7.78] * u.km/u.s

galcen_frame = coord.Galactocentric(galcen_distance=np.abs(sun_xyz[0]),
                                    galcen_v_sun=sun_vxyz,
                                    z_sun=0*u.pc)

# Pre-compute the rotation matrix to go from Galactocentric to ICRS
# (ra/dec) coordinates
R_gal, _ = get_matrix_vectors(galcen_frame, inverse=True)

# Calculate prior parameters from vx, vy, vz distributions.
import pkg_resources
vel_data = pkg_resources.resource_filename(__name__,
                                           "../data/gaia_mc5_velocities.csv")
vels = pd.read_csv(vel_data)
mu_vx = np.median(vels.basic_vx.values)
mu_vy = np.median(vels.basic_vy.values)
mu_vz = np.median(vels.basic_vz.values)
sigma_vx = 1.5*aps.median_absolute_deviation(vels.basic_vx.values)
sigma_vy = 1.5*aps.median_absolute_deviation(vels.basic_vy.values)
sigma_vz = 1.5*aps.median_absolute_deviation(vels.basic_vz.values)


def proper_motion_model(params, pos):
    """
    The model. Calculates proper motion from velocities and positions.

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).
        pos (list): Positional coordinates, RA [deg], dec [deg] and parallax
            [mas].

    Returns:
        pm_from_v (list): The ra and dec proper motions, calculated from
            velocity model parameters and observed positions.
        rv_from_v (list): The RV, calculated from velocity model parameters
            and observed positions.

    """

    # Unpack parameters and make distance linear.
    vx, vy, vz, lnD = params
    D = np.exp(lnD)

    # Calculate XYZ position from ra, dec and parallax
    c = coord.SkyCoord(ra = pos[0]*u.deg,
                       dec = pos[1]*u.deg,
                       distance = D*u.kpc)
    galcen = c.transform_to(galcen_frame)

    # Calculate pm and rv from XYZ and V_XYZ
    V_xyz_units = [vx, vy, vz]*u.km*u.s**-1
    pm_from_v, rv_from_v = av.get_icrs_from_galactocentric(galcen.data.xyz,
                                                           V_xyz_units,
                                                           R_gal, sun_xyz,
                                                           sun_vxyz)
    return pm_from_v, rv_from_v


def lnlike_one_star(params, pm, pm_err, pos, pos_err):
    """
    log-likelihood of proper motion and position, given velocity & distance.

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).
        pm (list): Proper motion in RA and dec in mas/yr. [pmra, pmdec].
        pm_err (list): Uncertainties on proper motion in RA and dec in
            mas/yr. [pmra_err, pmdec_err]
        pos (list): Positional coordinates, RA [deg], dec [deg] and parallax
            [mas].
        pos_err (list): Uncertainties on positional coordinates, RA_err
            [deg], dec_err [deg] and parallax_err [mas].

    Returns:
        The log-likelihood.

    """

    # Unpack parameters and make distance linear.
    vx, vy, vz, lnD = params
    D = np.exp(lnD)

    pm_from_v, rv_from_v = proper_motion_model(params, pos)

    # Compare this proper motion with observed proper motion.
    return -.5*(pm_from_v[0].value - pm[0])**2/pm_err[0]**2 \
           - .5*(pm_from_v[1].value - pm[1])**2/pm_err[1]**2 \
           - .5*(1./D - pos[2])**2/pos_err[2]**2


def lnGauss(x, mu, sigma):
    """
    A log-Gaussian.

    """
    ivar = 1./sigma**2
    return -.5*(x - mu)**2 * ivar


def lnprior(params):
    """
    The log-prior over distance and velocity parameters.

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).

    Returns:
        The log-prior.

    """

    vx, vy, vz, lnD = params

    # A log uniform prior over distance
    if lnD < 5 and -5 < lnD:

        # And a Gaussian prior over X, Y and Z velocities
        return lnGauss(vx, mu_vx, sigma_vx) \
             + lnGauss(vy, mu_vy, sigma_vy) \
             + lnGauss(vz, mu_vz, sigma_vz)
        # return lnGauss(vx, 84, 33) \
        #      + lnGauss(vy, 521, 14) \
        #      + lnGauss(vz, 68, 24)

    else:
        return -np.inf


def lnprob(params, pm, pm_err, pos, pos_err):
           # , R_gal, galcen_frame,
           # sun_xyz=[-8.122, 0, 0]*u.kpc,
           # sun_vxyz=[12.9, 245.6, 7.78]*u.km/u.s):
    """
    log-probability of distance and velocity, given proper motion and position

    Args:
        params (list): A list of vx [km/s], vy [km/s], vz [km/s] and
            ln(distance [kpc]).
        pm (list): Proper motion in RA and dec in mas/yr. [pmra, pmdec].
        pm_err (list): Uncertainties on proper motion in RA and dec in
            mas/yr. [pmra_err, pmdec_err]
        pos (list): Positional coordinates, RA [deg], dec [deg] and parallax
            [mas].
        pos_err (list): Uncertainties on positional coordinates, RA_err [deg],
            dec_err [deg] and parallax_err [mas].

    Returns:
        The log-probability.

    """
    return lnlike_one_star(params, pm, pm_err, pos, pos_err) + lnprior(params)
