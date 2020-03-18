import numpy as np
import pandas as pd

import pkg_resources
mpf = pkg_resources.resource_filename(__name__,
                                      '../data/metal_poor_avr.csv')
mrf = pkg_resources.resource_filename(__name__,
                                      '../data/metal_rich_avr.csv')
af = pkg_resources.resource_filename(__name__,
                                     '../data/all_stars_avr.csv')

def get_avr_coefficients():
    """
    Fit the parameters of the AVR using data from Yu & Liu:
    https://arxiv.org/pdf/1712.03965.pdf

    Fit to the relationship between log(sigma_z) and age.
    Returns:
        p_mp (array): coefficients [slope, intercept] for straight line fit to
            metal poor stars.
        p_mr (array): coefficients [slope, intercept] for straight line fit to
            metal rich stars.
        p_a (array): coefficients [slope, intercept] for straight line fit to
            all stars.

    """
    mp = pd.read_csv(mpf)
    mr = pd.read_csv(mrf)
    a = pd.read_csv(af)

    # p_mp = np.polyfit(mp.Age_Gyr, np.log(mp.sigma_z_kms), 1)
    # p_mr = np.polyfit(mr.Age_Gyr, np.log(mr.sigma_z_kms), 1)
    # p_a = np.polyfit(a.Age_Gyr, np.log(a.sigma_z_kms), 1)

    # p_mp = np.polyfit(np.log(mp.Age_Gyr), np.log(mp.sigma_z_kms), 1)
    # p_mr = np.polyfit(np.log(mr.Age_Gyr), np.log(mr.sigma_z_kms), 1)
    # p_a = np.polyfit(np.log(a.Age_Gyr), np.log(a.sigma_z_kms), 1)

    p_mp = np.polyfit(np.log(mp.sigma_z_kms), np.log(mp.Age_Gyr), 1)
    p_mr = np.polyfit(np.log(mr.sigma_z_kms), np.log(mr.Age_Gyr), 1)
    p_a = np.polyfit(np.log(a.sigma_z_kms), np.log(a.Age_Gyr), 1)

    return p_mp, p_mr, p_a


def v_to_age(v, coeffs):
    """
    Convert z velocity dispersion [km/s] to age [Gyr]

    Args:
        v (array): Z Velocity dispersion [km/s].
        coeffs (array): coefficients for straight line fit. [intercept, slope]

    Returns:
        age (array): age in Gyr.

    """
    b, a = coeffs
    # return (np.log(v) - a) / b
    # return (v/a)**(1./b)
    # logt = (np.log(v) - np.log(a))/b
    # return np.exp(logt)
    # logt = a + b*np.log(v)
    logt = np.polyval(coeffs, np.log(v))
    return np.exp(logt)

def age_to_v(t, coeffs):
    """
    Convert age [Gyr] to velocity dispersion [km/s]

    Args:
        t (array): age [Gyr].
        coeffs (array): coefficients for straight line fit. [intercept, slope]

    Returns:
        sigma_vz (array): Z velocity dispersion [km/s].

    """
    # a, b = coeffs[1], coeffs[0]
    # return np.exp(a + b*t)
    b, a = coeffs
    # return a * t**b
    # return np.exp(np.polyval(coeffs, np.log(t)))
    logv = (np.log(t) - a)/b
    return np.exp(logv)
