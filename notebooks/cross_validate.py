import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo

import aviary as av
import astropy.modeling as apm

from tqdm import trange
import pickle

from cv_routines import *

def RMS(ypred, yobs):
    return np.sqrt(sum((ypred - yobs)**2) / float(len(ypred)))

def MAD(ypred, yobs):
    return np.median(abs(ypred - yobs))

filename = "cross_val/kinerr_gp_model.pkl"

# Variables
young_limit = .7  # .1 is worse.
old_limit = 20.
hot_limit = 1.
dp, dc = .05, .05
cluster_old_err = .05  # relative prot uncert on Ruprecht 147 and NGC6819
sun_err = .01  # absolute prot uncertainty on the Sun's rotation period.

# The test variable
abs_kinerrs = np.linspace(1, 10, 9)  # The absolute prot uncertainty "
variables = abs_kinerrs
nvar = len(variables)

# The number of cross validation batches
nbatches = 10

# Perform cross validation on this variable
rms, mad = [np.zeros(nvar) for i in range(2)]
results, all_true_ages, all_pred_ages, all_sigmas = [], [], [], []
for j, var in enumerate(variables):
    print(j+1, "of", len(variables), "kinerr = ", var)

    # Create data arrays using variables
    x, age, prot, prot_err, ID, akin, cluster_x, cluster_prot, cluster_age \
        = assemble_data(young_limit, old_limit, hot_limit, dp, dc,
                        cluster_old_err, var, sun_err)

    # Cross validate.
    true_ages_cv, pred_ages_cv, sigmas_cv, ids_cv, results_cv \
        = cross_validate(x, age, prot, prot_err, ID, filename,
                         nbatches=nbatches)

    rms[j] = RMS(true_ages_cv, pred_ages_cv)
    mad[j] = MAD(true_ages_cv, pred_ages_cv)
    results.append(results_cv)
    all_true_ages.append(true_ages_cv)
    all_pred_ages.append(pred_ages_cv)
    all_sigmas.append(sigmas_cv)


# Plot inferred vs. true age
plt.figure(dpi=200)
nvars = len(all_true_ages)
var = abs_kinerrs
for i in range(nvars):
    plt.errorbar(all_true_ages[i], all_pred_ages[i], yerr=all_sigmas[i],
                 fmt="k.", zorder=0)
    plt.scatter(all_true_ages[i], all_pred_ages[i],
                c=np.ones(len(all_true_ages[i]))*var[i],
                vmin=min(var), vmax=max(var), zorder=1)
xs = np.linspace(0,13, 100)
plt.plot(xs, xs, "k-", lw=.5)
plt.colorbar(label="Variable value")
plt.xlabel("True age [Gyr]")
plt.ylabel("inferred age [Gyr]")
plt.legend()
plt.tight_layout()
plt.savefig("cross_val/cv_results_kinerr.png")
plt.close()

rms_med, mad_med = np.median(rms), np.median(mad)
rms_norm, mad_norm = rms/rms_med, mad/mad_med
sump = rms_norm + mad_norm

# Plot the RMS stats.
plt.figure(dpi=200)
plt.plot(abs_kinerrs, rms_norm, "C0o")
plt.plot(abs_kinerrs, rms_norm, "C0-", label="RMS")
plt.plot(abs_kinerrs, mad_norm, "C1o")
plt.plot(abs_kinerrs, mad_norm, "C1-", label="MAD")
plt.plot(abs_kinerrs, mad_norm, "C1o")
plt.plot(abs_kinerrs, sump, "C2-", label="Sum")
plt.plot(abs_kinerrs, sump, "C2o")
plt.ylabel("RMS or MAD")
plt.xlabel("Variable value")
plt.yscale("log")
plt.legend()
plt.title("RMS = {0:.3f}, MAD = {1:.3f}, both = {2:.3f}".format(
    float(abs_kinerrs[rms_norm==min(rms_norm)]),
    float(abs_kinerrs[mad_norm==min(mad_norm)]),
    float(abs_kinerrs[sump==min(sump)])))
plt.tight_layout()
plt.savefig("cross_val/rms_kinerr.png")
plt.close()
