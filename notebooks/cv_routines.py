import numpy as np
import pickle
import pandas as pd
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib as mpl

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
import astropy.modeling as apm


class AgePosterior(object):

    def __init__(self, filename):

        # Load the saved GP gyro model
        with open(filename, "rb") as f:
            model, map_soln = pickle.load(f)

        # Pull out the model and parameters
        with model:
            func = xo.get_theano_function_for_var(model.y_test)
            args = xo.utils.get_args_for_theano_function(map_soln)
            ind1 = model.vars.index(model.x1_test)
            ind2 = model.vars.index(model.x2_test)

        self.func, self.args, self.ind1, self.ind2 = func, args, ind1, ind2

    def get_post(self, c, prot, prot_err, age_lim=[-3, 3]):
        """

        Args:
            c (float): color
            prot (float): prot (days)
            prot_err (float): absolute prot uncertainty (days)
        """

        self.args[self.ind2] = np.linspace(age_lim[0], age_lim[1], 5000)
        self.args[self.ind1] = c + np.zeros_like(self.args[self.ind2])

        posterior = np.exp(-0.5 * (self.func(*self.args) - np.log(prot))**2 \
                           / (prot_err / prot) ** 2)
        return np.exp(self.args[self.ind2]), posterior


def assemble_data(young_limit, old_limit, hot_limit, dp, dc, cluster_old_err,
                  kinerr, sun_err, sun_color=.82, sun_prot=26.,
                  sun_age=4.56):
    """
    Load and assemble calibration data using variables that will be varied in
    cross validation.
    """

    k = pd.read_csv("../data/mcquillan_kinematic_ages.csv")
    k_lucy = pd.read_csv("../data/Gyrokinage2020_Prot.csv")
    kl = pd.DataFrame(dict({"kepid": k_lucy.kepid.values,
                            "kin_age_lucy": k_lucy.kin_age.values,
                            "kin_age_err": k_lucy.kin_age_err.values}))
    k = pd.merge(k, kl, on="kepid", how="left")

    # Remove subgiants and photometric binaries
    kin = k.iloc[k.flag.values == 1]
    finite = np.isfinite(kin.Prot.values) \
        & np.isfinite(kin.bprp_dered.values) \
        & np.isfinite(kin.kin_age_lucy.values)
    kin = kin.iloc[finite]

    # Remove stars bluer than 1.5 and with kinematic ages greater than 6 as
    # these are likely to be subgiants.
    subs = (kin.bprp_dered.values < 1.5) & (kin.kinematic_age.values > 6)
    kin = kin.iloc[~subs]

    # Remove stars that fall beneath the lower envelope using the
    # Angus + (2019) gyro relation and stars kinematically older than
    # old_limit.
    no_young = (kin.age.values > young_limit) \
        & (kin.kin_age_lucy.values < old_limit)
    kin = kin.iloc[no_young]

    # Remove hot stars as the clusters provide better coverage.
    cool = kin.bprp_dered.values > hot_limit
    akin = kin.iloc[cool]

    # Create grid of kinematic data
    logp = np.log10(akin.Prot.values)
    pgrid = np.arange(min(logp), max(logp), dp)
    cgrid = np.arange(min(akin.bprp_dered.values),
                      max(akin.bprp_dered.values), dc)
    P, C = np.meshgrid(pgrid, cgrid)
    A = np.zeros_like(P)
    prot_errs, npoints = [np.zeros_like(P) for i in range(2)]
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            b = (logp - .5*dp < P[i, j]) & (P[i, j] < logp + .5*dp)
            b &= (akin.bprp_dered.values - .5*dc < C[i, j]) \
                & (C[i, j] < akin.bprp_dered.values + .5*dc)
            A[i, j] = np.median(akin.kin_age_lucy.values[b])
            prot_errs[i, j] = np.sqrt(sum((akin.Prot_err.values[b]
                                           /akin.Prot.values[b])**2)) \
                / np.sqrt(float(len(akin.kin_age_lucy.values[b])))
            npoints[i, j] = len(akin.kin_age_lucy.values[b])

    finite = np.isfinite(C) & np.isfinite(P) & np.isfinite(A) \
        & np.isfinite(prot_errs)
    morethan = npoints[finite] > 1
    C, P, A, prot_errs = C[finite][morethan], P[finite][morethan], \
        A[finite][morethan], prot_errs[finite][morethan]

    # Load cluster data from Get_cluster_scatter and add the Sun.
    cluster_uncert = pd.read_csv("../data/clusters_with_uncertainties.csv")
    cluster_x = np.concatenate((cluster_uncert.bprp.values,
                                np.array([sun_color])))
    cluster_prot = np.concatenate((cluster_uncert.prot.values,
                                   np.array([sun_prot])))
    cluster_age = np.concatenate((cluster_uncert.age.values,
                                  np.array([sun_age])))
    cluster_prot_errs = np.concatenate((cluster_uncert.prot_err.values,
                                        np.array([sun_err])))

    # Decrease the uncertainties on the oldest clusters to cluster_old_err
    select_old = cluster_age > 2.
    cluster_prot_errs[select_old] = np.ones(len(
        cluster_prot_errs[select_old])) * cluster_old_err

    # Combine clusters with kinematic grid
    x = np.concatenate((cluster_x, np.ndarray.flatten(C)))
    prot_err = np.concatenate((cluster_prot_errs,
                               np.ndarray.flatten(10**P)*kinerr))
    prot = np.concatenate((cluster_prot, np.ndarray.flatten(10**P)))
    age = np.concatenate((cluster_age, np.ndarray.flatten(A)))
    ID = np.concatenate((np.zeros_like(cluster_age),
                         np.ones_like(np.ndarray.flatten(A))))
    # 0s are clusters, 1s are kinematics
    return x, age, prot, prot_err, ID, akin


def get_stellar_ages(x, prot, prot_err, filename):
    """
    Loop over stars and get maximum a-posteriori ages.
    """

    ap = AgePosterior(filename)

    # Loop over stars.
    mu, mu_fit, sig = [np.zeros(len(x)) for i in range(3)]
    for i in trange(len(x)):

        # Get the age posterior
        age_array, posterior = ap.get_post(x[i], prot[i], prot_err[i])

        # Adopt MAP as mean
        mu[i] = age_array[posterior == max(posterior)]

        g_init = apm.models.Gaussian1D(amplitude=1., mean=mu[i], stddev=.5)
        fit_g = apm.fitting.LevMarLSQFitter()
        g = fit_g(g_init, age_array, posterior)
        mu_fit[i], sig[i] = g.mean.value, g.stddev.value
    return mu, sig, mu_fit


def make_plot(kin, x, age, prot, prot_err, cluster_x, cluster_prot,
              cluster_age, filename):
    # Format data for GP fit.
    inds = np.argsort(x)
    x1 = np.array(x[inds])
    x2 = np.log(np.array(age[inds]))
    y = np.log(np.array(prot[inds]))
    y_err = prot_err[inds]/prot[inds]

    x2_min = np.min(x2)
    xp1 = np.linspace(x1.min() - .2, x1.max() + .2, 1000)
    xg2 = np.array([np.log(.12), np.log(.67), np.log(1), np.log(1.6),
                    np.log(2.7), np.log(4.56), np.log(5), np.log(8),
                    np.log(10), np.log(14)])


    # Load the saved GP gyro model
    with open(filename, "rb") as f:
        model, map_soln = pickle.load(f)

    # Pull out the model and parameters
    with model:
        func = xo.get_theano_function_for_var(model.y_test)
        args = xo.utils.get_args_for_theano_function(map_soln)
        ind1 = model.vars.index(model.x1_test)
        ind2 = model.vars.index(model.x2_test)

    cmap = mpl.cm.get_cmap("plasma_r")

    vmin = np.exp(x2).min()
    vmax = np.exp(x2).max()

    def get_color(x2):
        return cmap((np.exp(x2) - vmin) / (vmax - vmin))

    plt.figure(figsize=(12, 7), dpi=200)
    plt.plot(kin.bprp_dered, kin.Prot, ".", color=".8", mec="none", ms=10,
             alpha=.2, zorder=0, label="$\mathrm{Field~Star~distribution}$")
    plt.scatter(x1, np.exp(y), c=np.exp(x2), vmin=vmin, vmax=vmax, s=8,
                cmap="plasma_r",
                label="$\mathrm{Kinematic~Ages~(Field~Stars)}$")
    plt.scatter(cluster_x, cluster_prot, c=cluster_age, vmin=vmin, vmax=vmax,
                s=20, edgecolor="k", cmap="plasma_r",
                label="$\mathrm{Cluster~Stars}$")

    for i in range(len(xg2)):
        plt.plot(xp1, np.exp(map_soln["pred_{0}".format(i)]), color="k",
                 lw=1.25)
        plt.plot(xp1, np.exp(map_soln["pred_{0}".format(i)]),
                 color=get_color(xg2[i]), lw=0.75)

    plt.xlabel("$\mathrm{G_{BP} - G_{RP}}$")
    plt.ylabel("$\mathrm{P_{rot}~[days]}$")
    plt.ylim(0.5, 180)
    plt.yscale("log")
    plt.colorbar(label="$\mathrm{Age~[Gyr]}$");
    plt.xlim(.5, 2.8)
    leg = plt.legend()
    leg.legendHandles[0]._legmarker.set_alpha(1)
    plt.tight_layout()
