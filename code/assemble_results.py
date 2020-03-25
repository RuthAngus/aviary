# Combine information from multiple .csv files.
# Intended to be run on the cluster.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


files = glob.glob("velocities/mock/*csv")
# files = glob.glob("velocities/cluster/*csv")
df = pd.read_csv(files[0])

for i in range(1, len(files)):
    df = pd.concat([df, pd.read_csv(files[i])])

df.to_csv("all_stars_mock.csv")
# df.to_csv("all_stars_cluster.csv")

df1 = pd.read_csv("mock_df.csv")
# df1 = pd.read_csv("../data/gaia_mc5_velocities.csv")
df2 = pd.merge(df1, df, on="kepid", how="right")

# vx = df2.mock_vx
# vz = df2.mock_vz
vx = df2.basic_vx
vz = df2.basic_vz

plt.errorbar(vx, df2.vx_inferred,
             yerr=[df2.vx_inferred_errm, df2.vx_inferred_errp],
             fmt=".")
xs = np.linspace(min(vx), max(vx), 100)
plt.plot(xs, xs)
plt.savefig("testvx_mock")
plt.close()

plt.errorbar(vz, df2.vz_inferred,
             yerr=[df2.vz_inferred_errm, df.vz_inferred_errp],
             fmt=".")
xs = np.linspace(min(vz), max(vz), 100)
plt.plot(xs, xs)
plt.savefig("testvz_mock")
plt.close()
