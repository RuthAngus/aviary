from multiprocessing import Pool
from subprocess import check_call

import numpy as np
import pandas as pd

# Path needed for SLURM
import sys
import os
sys.path.append(os.getcwd())


def run_kepid(kepid):
    print("running {0}".format(kepid))
    os.environ["THEANO_FLAGS"] = "compiledir=./cache/{0}".format(os.getpid())
    check_call("python pymc3_multi_star.py {0}".format(kepid), shell=True)

# Load data.
vels = pd.read_csv("../data/gaia_mc5_velocities.csv")
m = vels.radial_velocity.values != 0
m &= np.isfinite(vels.basic_vx.values)
m &= np.isfinite(vels.basic_vy.values)
m &= np.isfinite(vels.basic_vz.values)
vels = vels.iloc[m]

kepids = vels.kepid.values

with Pool(8) as pool:
    pool.map(run_kepid, kepids)
