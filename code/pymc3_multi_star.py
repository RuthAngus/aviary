import numpy as np
import pandas as pd

import pymc3 as pm
import corner
from astropy.coordinates import SkyCoord
import astropy.units as u

import aviary as av

from multiprocessing import Pool

# Path needed for SLURM
import sys
import os
sys.path.append(os.getcwd())

if len(sys.argv) < 2:
    raise RuntimeError("you must provide a directory to save results and" \
                       "a KIC number")

kepid = int(sys.argv[1])

# Get that faint or bright prior.
# prior = "bright"
# prior = "all"
prior = "faint"
DIR = "{}_prior".format(prior)
mu, cov = av.get_prior(cuts=prior)
print(mu)
assert 0

# Find the target star in the sample
vels = pd.read_csv("../data/mcquillan_santos_gaia_lamost_velocities.csv")
m = vels.kepid.values == kepid
df = vels.iloc[m]

pos = [float(df["ra"]), float(df["dec"]), float(df["parallax"])]
pos_err = [float(df["ra_error"]), float(df["dec_error"]),
           float(df["parallax_error"])]
proper = [float(df["pmra"]), float(df["pmdec"])]
proper_err = [float(df["pmra_error"]), float(df["pmdec_error"])]

trace = av.run_pymc3_model(pos, pos_err, proper, proper_err, mu, cov)
flat_samples = pm.trace_to_dataframe(trace)

params_inferred = np.median(flat_samples, axis=0)
upper = np.percentile(flat_samples, 84, axis=0)
lower = np.percentile(flat_samples, 16, axis=0)
errp = upper - params_inferred
errm = params_inferred - lower
std = np.std(flat_samples, axis=0)

df1 = pd.DataFrame(dict({
    "kepid": kepid,
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

df1.to_csv("velocities/{0}/{1}.csv".format(DIR, int(df["kepid"])))
