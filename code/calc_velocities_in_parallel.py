#!/usr/bin/python3


import numpy as np
import pandas as pd

import matplotlib as mpl
import emcee

from multiprocessing import Pool

# Path needed for SLURM
import sys
import os
sys.path.append(os.getcwd())

import aviary as av


def infer_velocity(df):

    # Format parameter and data arrays.
    pos = [df["ra"], df["dec"], df["parallax"]]
    pos_err = [df["ra_error"], df["dec_error"], df["parallax_error"]]
    pm = [df["pmra"], df["pmdec"]]
    pm_err = [df["pmra_error"], df["pmdec_error"]]

    # Run MCMC.
    ndim, nwalkers = 4, 16
    inits = [df["basic_vx"], df["basic_vy"], df["basic_vz"],
             np.log(1./df["parallax"])]
    p0 = np.random.randn(nwalkers, ndim) + inits
    sampler = emcee.EnsembleSampler(nwalkers, ndim, av.lnprob,
                                    args=(pm, pm_err, pos, pos_err))
    sampler.run_mcmc(p0, 100, progress=True);

    # Extract inferred parameters and uncertainties.
    flat_samples = sampler.get_chain(discard=50, thin=5, flat=True)
    params_inferred = np.median(flat_samples, axis=0)
    upper = np.percentile(flat_samples, 84, axis=0)
    lower = np.percentile(flat_samples, 16, axis=0)
    errp = upper - params_inferred
    errm = params_inferred - lower
    std = np.std(flat_samples, axis=0)

    df1 = pd.DataFrame(dict({
        "kepid": df["kepid"],
        "vx_inferred": params_inferred[0],
        "vx_inferred_errp": errp[0],
        "vx_inferred_errm": errm[0],
        "vx_inferred_err": std[0],
        "vy_inferred": params_inferred[1],
        "vy_inferred_errp": errp[1],
        "vy_inferred_errm": errm[1],
        "vy_inferred_err": std[1],
        "vz_inferred": params_inferred[2],
        "vz_inferred_errp": errp[2],
        "vz_inferred_errm": errm[2],
        "vz_inferred_err": std[2],
        "distance_inferred": params_inferred[3],
        "distance_inferred_errp": errp[3],
        "distance_inferred_errm": errm[3],
        "distance_inferred_err": std[3]
        }), index=[0])

    df1.to_csv("velocities/{}.csv".format(df["kepid"]))


# Load the data
df0 = pd.read_csv("../data/gaia_mc5_velocities.csv")

# For now, just run on stars with RV measurements.
m = np.isfinite(df0["vz"].values)

# Just the first 10 stars for now.
df = df0.iloc[m][:10]

list_of_dicts = []
for i in range(len(df)):
    list_of_dicts.append(df.iloc[i].to_dict())

print(list_of_dicts[0])
print(len(list_of_dicts))

p = Pool(24)
list(p.map(infer_velocity, list_of_dicts))
