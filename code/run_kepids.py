from multiprocessing import Pool
from subprocess import check_call, CalledProcessError

import numpy as np
import pandas as pd

# Path needed for SLURM
import sys
import os
sys.path.append(os.getcwd())

import starspot as ss


def run_kepid(kepid):
    print("running {0}".format(kepid))
    os.environ["THEANO_FLAGS"] = "compiledir=./cache/{0}".format(os.getpid())
    try:
        check_call("python3 pymc3_multi_star.py {0}".format(kepid), shell=True)
    except CalledProcessError as e:
        print("running {0} failed:".format(kepid) + str(e))
    else:
        print("running {0} succeeded".format(kepid))


# Load data.
vels = pd.read_csv("../data/mcquillan_santos_gaia_lamost_velocities.csv")
finite = vels.rv.values != 0
finite &= np.isfinite(vels.vx.values)
finite &= np.isfinite(vels.vy.values)
finite &= np.isfinite(vels.vz.values)

lnD = np.log(1./vels.parallax.values)

# Get rid of outliers
nsigma = 3
mx = ss.sigma_clip(vels.vx.values[finite], nsigma=nsigma)
my = ss.sigma_clip(vels.vy.values[finite], nsigma=nsigma)
mz = ss.sigma_clip(vels.vz.values[finite], nsigma=nsigma)
md = ss.sigma_clip(lnD[finite], nsigma=nsigma)
m = mx & my & mz & md

# Just choose some really faint stars to test on.
gmag = vels.phot_g_mean_mag.values[finite][m]
m_really_faint = gmag > 15.5

vels = vels.iloc[finite][m][m_really_faint]
kepids = vels.kepid.values[:2]

with Pool(8) as pool:
    pool.map(run_kepid, kepids)
