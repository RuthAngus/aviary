# This script is for simulating and trying to recover velocities.

import numpy as np
import pandas as pd
import astropy.stats as aps
import aviary as av
from multiprocessing import Pool
import emcee

def infer_velocity(df):

    # Format parameter and data arrays.
    pos = [df["ra"], df["dec"], df["parallax"]]
    pos_err = [df["ra_error"], df["dec_error"], df["parallax_error"]]
    pm = [df["pmra"], df["pmdec"]]
    pm_err = [df["pmra_error"], df["pmdec_error"]]

    # Run MCMC.
    ndim, nwalkers = 4, 16
    inits = [df["vx"], df["vy"], df["vz"], np.log(1./df["parallax"])]
    p0 = np.random.randn(nwalkers, ndim)*1e-2 + inits
    sampler = emcee.EnsembleSampler(nwalkers, ndim, av.lnprob,
                                    args=(pm, pm_err, pos, pos_err))

    nsteps = 1000
    sampler.run_mcmc(p0, nsteps, progress=True);

    flat_samples = sampler.get_chain(discard=int(nsteps/2), flat=True)
    params_inferred = np.median(flat_samples, axis=0)
    upper = np.percentile(flat_samples, 84, axis=0)
    lower = np.percentile(flat_samples, 16, axis=0)
    errp = upper - params_inferred
    errm = params_inferred - lower
    std = np.std(flat_samples, axis=0)

    df["ID"] = df["ID"]
    df["vx_inferred"] = params_inferred[0],
    df["vx_inferred_errp"] = errp[0],
    df["vx_inferred_errm"] = errm[0],
    df["vx_inferred_err"] = std[0],
    df["vy_inferred"] = params_inferred[1],
    df["vy_inferred_errp"] = errp[1],
    df["vy_inferred_errm"] = errm[1],
    df["vy_inferred_err"] = std[1],
    df["vz_inferred"] = params_inferred[2],
    df["vz_inferred_errp"] = errp[2],
    df["vz_inferred_errm"] = errm[2],
    df["vz_inferred_err"] = std[2],
    df["lndistance_inferred"] = params_inferred[3],
    df["lndistance_inferred_errp"] = errp[3],
    df["lndistance_inferred_errm"] = errm[3],
    df["lndistance_inferred_err"] = std[3]
    df = pd.DataFrame(df)
    df.to_csv("{}.csv".format(int(df["ID"])))

    # df1 = pd.DataFrame(dict({
    #     "ID": df["ID"],
    #     "vx_inferred": params_inferred[0],
    #     "vx_inferred_errp": errp[0],
    #     "vx_inferred_errm": errm[0],
    #     "vx_inferred_err": std[0],
    #     "vy_inferred": params_inferred[1],
    #     "vy_inferred_errp": errp[1],
    #     "vy_inferred_errm": errm[1],
    #     "vy_inferred_err": std[1],
    #     "vz_inferred": params_inferred[2],
    #     "vz_inferred_errp": errp[2],
    #     "vz_inferred_errm": errm[2],
    #     "vz_inferred_err": std[2],
    #     "lndistance_inferred": params_inferred[3],
    #     "lndistance_inferred_errp": errp[3],
    #     "lndistance_inferred_errm": errm[3],
    #     "lndistance_inferred_err": std[3]
    #     }), index=[0])

    # df1.to_csv("{}.csv".format(int(df["ID"])))


if __name__ == "__main__":

    np.random.seed(42)

    # Calculate prior parameters from vx, vy, vz distributions
    df = pd.read_csv("../data/gaia_mc5_velocities.csv")
    mu_vx = np.median(df.basic_vx.values)
    mu_vy = np.median(df.basic_vy.values)
    mu_vz = np.median(df.basic_vz.values)
    sigma_vx = 1.5*aps.median_absolute_deviation(df.basic_vx.values)
    sigma_vy = 1.5*aps.median_absolute_deviation(df.basic_vy.values)
    sigma_vz = 1.5*aps.median_absolute_deviation(df.basic_vz.values)

    # Calculate covariance between velocities
    VX = np.stack((df.basic_vx.values, df.basic_vy.values,
                   df.basic_vz.values), axis=0)
    mean = np.mean(VX, axis=1)
    cov = np.cov(VX)

    # Draw parameters from the prior.
    Nstars = 10
    # # vxs = np.random.randn(Nstars)*sigma_vx + mu_vx
    # # vys = np.random.randn(Nstars)*sigma_vy + mu_vy
    # # vzs = np.random.randn(Nstars)*sigma_vz + mu_vz
    # vxs = np.random.randn(Nstars)*np.sqrt(cov[0, 0]) + mean[0]
    # vys = np.random.randn(Nstars)*np.sqrt(cov[1, 1]) + mean[1]
    # vzs = np.random.randn(Nstars)*np.sqrt(cov[2, 2]) + mean[2]

    # vxs, vys, vzs = np.random.multivariate_normal(mean, cov, Nstars).T
    # print(np.shape(vxs))
    # assert 0

    # lnds = np.random.uniform(np.log(1e-5), np.log(2), Nstars)
    # ra = np.random.uniform(280, 300, Nstars)
    # dec = np.random.uniform(36, 52, Nstars)
    # parallax_error = np.ones(Nstars)*1e-5
    # ra_error = np.ones(Nstars)*1e-5
    # dec_error = np.ones(Nstars)*1e-5

    # # Replace with real data
    # vxs = df.basic_vx.values[:Nstars]
    # vys = df.basic_vy.values[:Nstars]
    # vzs = df.basic_vz.values[:Nstars]
    # lnds = 1./df.parallax.values[:Nstars]
    # ra = df.ra.values[:Nstars]
    # ra_error = df.ra_error.values[:Nstars]
    # dec = df.dec.values[:Nstars]
    # dec_error = df.dec_error.values[:Nstars]
    # parallax_error = df.parallax_error.values[:Nstars]
    # parallax = df.parallax.values[:Nstars]

    df1 = pd.DataFrame(dict({"ID": np.arange(Nstars),
                            "vx": vxs,
                            "vy": vys,
                            "vz": vzs,
                            "lnd": lnds,
                            "parallax": 1./np.exp(lnds),
                            "parallax_error": parallax_error,
                            "ra": ra,
                            "ra_error": ra_error,
                            "dec": dec,
                            "dec_error": dec_error
                            }))

    # # Generate mock proper motion and RV.
    # pmras, pmdecs, rvs = [], [], []
    # for i in range(Nstars):
    #     pos = [ra[i], dec[i], 1./np.exp(lnds[i])]
    #     params = [vxs[i], vys[i], vzs[i], lnds[i]]

    #     # Calculate observables for these stars.
    #     pm, rv = av.proper_motion_model(params, pos)
    #     pmras.append(pm[0].value)
    #     pmdecs.append(pm[1].value)
    #     rvs.append(rv.value)
    # pmra_errs = np.ones(Nstars)*1e-5
    # pmdec_errs = np.ones(Nstars)*1e-5

    # Replace with real data
    pmras = df.pmra.values[:Nstars]
    pmra_errs = df.pmra_error.values[:Nstars]
    pmdecs = df.pmdec.values[:Nstars]
    pmdec_errs = df.pmdec_error.values[:Nstars]
    rvs = df.radial_velocity.values[:Nstars]

    df1["pmra"] = np.array(pmras)
    df1["pmra_error"] = pmra_errs
    df1["pmdec"] = np.array(pmdecs)
    df1["pmdec_error"] = pmdec_errs
    df1["rv"] = np.array(rvs)

    list_of_dicts = []
    for i in range(len(df1)):
        list_of_dicts.append(df1.iloc[i].to_dict())

    p = Pool(24)
    list(p.map(infer_velocity, list_of_dicts))
    # for i in range(Nstars):
    #     infer_velocity(df.iloc[i])
