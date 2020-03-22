# Script for calculating vx, vy, vz for stars with RVs.

import numpy as np
import pandas as pd
import aviary as av


df0 = pd.read_csv("../data/gaia_mc5.csv")
m = df0.radial_velocity.values != 0
df = df0.iloc[m]

# for i, row in enumerate(df):
xyz, vxyz = av.simple_calc_vxyz(df.ra.values, df.dec.values,
                                1./df.parallax.values, df.pmra.values,
                                df.pmdec.values, df.radial_velocity.values)

vx, vy, vz = vxyz
x, y, z = xyz

df = df.copy(deep=True)
df["basic_vx"] = vx.value
df["basic_vy"] = vy.value
df["basic_vz"] = vz.value

df["x"] = x.value
df["y"] = y.value
df["z"] = z.value

df.to_csv("gaia_mc5_velocities.csv")
