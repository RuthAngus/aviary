import numpy as np
import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def make_plot(kin, x, prot, prot_err, age, cluster_x, cluster_prot,
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
