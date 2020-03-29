# Combine information from multiple .csv files.
# Intended to be run on the cluster.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import aviary as av

plotpar = {'axes.labelsize': 20,
           'font.size': 22,
           'legend.fontsize': 15,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)


files = glob.glob("velocities/*csv")
df = pd.read_csv(files[0])

for i in range(1, len(files)):
    df = pd.concat([df, pd.read_csv(files[i])])

df1 = pd.read_csv("../data/gaia_mc5_velocities.csv")
df2 = pd.merge(df1, df, on="kepid", how="right")


# Calculate vx vz and vy from the data and rv and pm from the inferred params.
vx_true, vy_true, vz_true = [], [], []
pmra_inf, pmdec_inf, rv_inf = [], [], []
for i in range(len(df2)):
    params = [df2.vx_inferred.values[i], df2.vy_inferred.values[i],
              df2.vz_inferred.values[i], df2.lndistance_inferred.values[i]]
    pos = [df2.ra.values[i], df2.dec.values[i], df2.parallax.values[i]]
    # pos = [df2.ra.values[i], df2.dec.values[i],
    #        1./np.exp(df2.lndistance_inferred.values[i])]
    _pm, _rv = av.proper_motion_model(params, pos)
    pmra_inf .append(_pm[0].value)
    pmdec_inf .append(_pm[1].value)
    rv_inf .append(_rv.value)

    xyz, vxyz = av.simple_calc_vxyz(df2.ra[i], df2.dec[i], 1./df2.parallax[i],
                                    df2.pmra[i], df2.pmdec[i],
                                    df2.radial_velocity[i])
    vx_true.append(vxyz.value[0])
    vy_true.append(vxyz.value[1])
    vz_true.append(vxyz.value[2])

xs = np.linspace(min(pmra_inf), max(pmra_inf), 100)
plt.plot(xs, xs)
plt.plot(df2.pmra, np.array(pmra_inf), ".")
plt.savefig("compare_pmra")
plt.close()

xs = np.linspace(min(pmdec_inf), max(pmdec_inf), 100)
plt.plot(xs, xs)
plt.plot(df2.pmdec, pmdec_inf, ".")
plt.savefig("compare_pmdec")
plt.close()

xs = np.linspace(min(rv_inf), max(rv_inf), 100)
plt.plot(xs, xs)
plt.plot(df2.radial_velocity, rv_inf, ".")
plt.savefig("compare_rv")
plt.close()

xs = np.linspace(min(1./df2.parallax), max(1./df2.parallax), 100)
plt.plot(xs, xs)
plt.plot(1./df2.parallax, np.exp(df2.lndistance_inferred), ".")
plt.savefig("compare_distance")
plt.close()

plt.errorbar(vx_true, df2.vx_inferred,
             yerr=[df2.vx_inferred_errm, df2.vx_inferred_errp],
             fmt=".")
xs = np.linspace(min(vx_true), max(vx_true), 100)
plt.plot(xs, xs)
plt.savefig("testvx")
plt.savefig("testvx.pdf")
plt.close()

plt.figure(figsize=(5, 4), dpi=200)
xs = np.linspace(-25, 25, 100)
plt.plot(xs, xs, ls="--", color="k", lw=.8, zorder=0)
plt.errorbar(vz_true, df2.vz_inferred,
             yerr=[df2.vz_inferred_errm, df.vz_inferred_errp],
             fmt=".", zorder=1)
plt.xlabel("$\mathrm{V_z~from~6D~[kms^{-1}]}$")
plt.ylabel("$\mathrm{V_z~from~5D~[kms^{-1}]}$")
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.tight_layout()
plt.savefig("testvz")
plt.savefig("testvz.pdf")
plt.close()

plt.errorbar(vy_true, df2.vy_inferred,
             yerr=[df2.vy_inferred_errm, df.vy_inferred_errp],
             fmt=".")
xs = np.linspace(min(vy_true), max(vy_true), 100)
plt.plot(xs, xs)
plt.xlabel("$\mathrm{V_y~from~6D~[kms^{-1}]}$")
plt.ylabel("$\mathrm{V_y~from~5D~[kms^{-1}]}$")
plt.savefig("testvy")
plt.savefig("testvy.pdf")
plt.close()
