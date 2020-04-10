import numpy as np
from tqdm import trange
import astropy.stats as aps
import matplotlib.pyplot as plt


def err_to_log10_err(value, err):
    return err/value/np.log(10)


def err_on_sample_std_dev(std_dev_of_distribution, n):
    """
    from https://stats.stackexchange.com/questions/156518/what-is-the-
    standard-error-of-the-sample-standard-deviation

    Which takes the derivation from
    Rao (1973) Linear Statistical Inference and its Applications 2nd Ed, John
    Wiley & Sons, NY

    Derivation for standard error on the variance is here:
    https://math.stackexchange.com/questions/72975/variance-of-sample-variance

    Args:
        std_dev_of_distribution (float): The standard deviation of the
            Gaussian distribution
        n (int): The number of data points.

    Returns:
        The standard error of the sample standard deviation (not variance).
    """

    sig = std_dev_of_distribution
    return 1./(2*sig) * np.sqrt(2*sig**4/(n-1))


# Calculate tangential distance.
def tan_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Find N nearest points
def n_nearest_points(x1, y1, x2, y2, z2, N):
    td = tan_dist(x1, y1, x2, y2)
    inds = np.argsort(td)
    return x2[inds[:N]], y2[inds[:N]], z2[inds[:N]]


def make_bin(x1, y1, x2, y2, z2, xrange, yrange):
    dx, dy = xrange/2., yrange/2.
    xlower, xupper = x1 - dx, x1 + dx
    ylower, yupper = y1 - dy, y1 + dy

    m = (xlower < x2) * (x2 < xupper) * (ylower < y2) * (y2 < yupper)
    return x2[m], y2[m], z2[m]


# Run on each star
def calc_dispersion_nearest(x, y, z, N):

    dispersions = np.zeros(len(x))
    for i in trange(len(x)):
        nx, ny, nz = n_nearest_points(x[i], y[i], x, y, z, N)
        dispersions[i] = 1.5*aps.median_absolute_deviation(nz, ignore_nan=True)

    return dispersions


def calc_dispersion_bins(x, y, z, xrange, yrange, method="mad"):

    dispersions = np.zeros(len(x))
    for i in trange(len(x)):
        nx, ny, nz = make_bin(x[i], y[i], x, y, z, xrange, yrange)

        if method == "mad" or method == "MAD":
            dispersions[i] = 1.5*aps.median_absolute_deviation(
                nz, ignore_nan=True)

        elif method == "std" or method == "STD":
            dispersions[i] = np.nanstd(nz)

    return dispersions


def calc_dispersion_bins_target(target_x, target_y, x, y, z, xrange, yrange):
    dispersions = np.zeros(len(target_x))
    for i in trange(len(target_x)):
        nx, ny, nz = make_bin(target_x[i], target_y[i], x, y, z, xrange,
                                 yrange)
        dispersions[i] = 1.5*aps.median_absolute_deviation(nz,
                                                           ignore_nan=True)
    return dispersions
