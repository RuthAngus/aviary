import numpy as np
import matplotlib.pyplot as plt
import aviary
import scipy.stats as sps
import pandas as pd


def test_err_to_log10_err():
    value = 20
    err = .1
    assert np.isclose(10**(np.log10(value)
                           + aviary.err_to_log10_err(value, err)),
                      value + err, atol=.01*value)


def test_tan_dist():
    x1, y1 = 1, 1
    x2, y2 = 2, 2
    assert aviary.tan_dist(x1, y1, x2, y2) == np.sqrt(2)


def test_n_nearest_points():
    x1, y1 = 10, 12
    np.random.seed(42)
    x2, y2 = [np.random.randn(1000) + 10 for i in range(2)]
    z2 = np.random.randn(1000)*y2
    nx, ny, nz = aviary.n_nearest_points(x1, y1, x2, y2, z2, 50)


def test_make_bin():
    np.random.seed(42)
    x, y, z = [np.random.randn(1000) + 10 for i in range(3)]
    bx, by, bz = aviary.make_bin(10, 10, x, y, z, 1, 1)
    plt.plot(x, y, ".")
    plt.plot(bx, by, ".")


def test_calc_dispersion():
    x2, y2 = [np.random.randn(1000) + 10 for i in range(2)]
    z2 = np.random.randn(1000)*y2

    dispersions_nearest = aviary.calc_dispersion_nearest(x2, y2, z2, 100);
    dispersions_bins = aviary.calc_dispersion_bins(x2, y2, z2, .5, .5);

    return dispersions_nearest, dispersions_bins


if __name__ == "__main__":
    test_err_to_log10_err()
    test_tan_dist()
    test_n_nearest_points()
    test_make_bin()
    test_calc_dispersion()
