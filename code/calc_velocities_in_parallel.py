#!/usr/bin/python3

import numpy as np
import pandas as pd

import matplotlib as mpl
import emcee
import corner

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

    # # Real data
    # pm = [df["pmra"], df["pmdec"]]
    # pm_err = [df["pmra_error"], df["pmdec_error"]]
    # inits = [df["basic_vx"], df["basic_vy"], df["basic_vz"],
    #          np.log(1./df["parallax"])]

    # Mock data
    pm = [df["mock_pmra"], df["mock_pmdec"]]
    pm_err = [df["mock_pmra_error"], df["mock_pmdec_error"]]
    inits = [df["mock_vx"], df["mock_vy"], df["mock_vz"],
             np.log(1./df["parallax"])]

    # Run MCMC.
    ndim, nwalkers = 4, 16
    p0 = np.random.randn(nwalkers, ndim)*1e-2 + inits
    sampler = emcee.EnsembleSampler(nwalkers, ndim, av.lnprob,
                                    args=(pm, pm_err, pos, pos_err))

    # nsteps = 10000
    nsteps = 1000
    sampler.run_mcmc(p0, 1000, progress=True);

    # Extract inferred parameters and uncertainties.
    flat_samples = sampler.get_chain(discard=int(nsteps/2), flat=True)
    # fig = corner.corner(flat_samples)
    # fig.savefig("corner")

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
        "lndistance_inferred": params_inferred[3],
        "lndistance_inferred_errp": errp[3],
        "lndistance_inferred_errm": errm[3],
        "lndistance_inferred_err": std[3]
        }), index=[0])

    # df1.to_csv("velocities/{}.csv".format(df["kepid"]))
    df1.to_csv("velocities/mock/{}.csv".format(df["kepid"]))


# Load the data
# df0 = pd.read_csv("../data/gaia_mc5_velocities.csv")
df0 = pd.read_csv("mock_df.csv")

# For now, just run on stars with RV measurements.
m = df0.radial_velocity.values != 0

# Just the first 10 stars for now.
df = df0.iloc[m][:10]

list_of_dicts = []
for i in range(len(df)):
    list_of_dicts.append(df.iloc[i].to_dict())

print("Running on ", len(list_of_dicts), "stars")

p = Pool(24)
list(p.map(infer_velocity, list_of_dicts))
# infer_velocity(df.iloc[0])
