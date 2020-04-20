# Converting Exploring data into a script.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.utils as au
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery
import astropy.units as units
from tools import getDust
from stardate.lhf import age_model
from calc_velocities import calc_vb, calc_vz, calc_vl
import astropy.units as u
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
from astropy.table import Table
from pyia import GaiaData
import astropy.coordinates as coord
from photometric_teff import bprp_to_teff

plotpar = {'axes.labelsize': 30,
           'font.size': 30,
           'legend.fontsize': 15,
           'xtick.labelsize': 30,
           'ytick.labelsize': 30,
           'text.usetex': True}
plt.rcParams.update(plotpar)


def load_and_merge_data():
    # Load Gaia-Kepler crossmatch.
    with fits.open("../data/kepler_dr2_1arcsec.fits") as data:
        gaia = pd.DataFrame(data[1].data, dtype="float64")
    m = gaia.parallax.values > 0
    gaia = gaia.iloc[m]

    # Load Santos stars
    santos = pd.read_csv("../data/santos.csv", skiprows=41)
    santos["kepid"] = santos.KIC

    # Load McQuillan stars
    mc1 = pd.read_csv("../data/Table_1_Periodic.txt")
    mc1["kepid"] = mc1.KID.values

    # Merge santos, mcquillan and Gaia
    mc_santos = pd.merge(santos, mc1, how="outer", on="kepid",
                        suffixes=["_santos", ""])
    mc_sant_gaia = pd.merge(mc_santos, gaia, how="left", on="kepid",
                            suffixes=["KIC", ""])
    df0 = mc_sant_gaia.drop_duplicates(subset="kepid")

    # Add LAMOST RVs
    lamost = pd.read_csv("../data/KeplerRot-LAMOST.csv")
    lamost["kepid"] = lamost.KIC.values
    lam = pd.merge(df0, lamost, on="kepid", how="left",
                   suffixes=["", "_lamost"])
    df = lam.drop_duplicates(subset="kepid")

    # Add Travis Berger's masses
    travis = pd.read_csv("../data/Ruth_McQuillan_Masses_Out.csv")
    masses = pd.DataFrame(dict({"kepid": travis.KIC.values,
                                "Mass": travis.iso_mass.values}))
    masses.head()
    df = pd.merge(masses, df, how="right", on="kepid",
                  suffixes=["_berger", ""])

    df = df.drop_duplicates(subset="kepid")
    return df


# S/N cuts
def sn_cuts(df):
    sn = df.parallax.values/df.parallax_error.values

    m = (sn > 10)
    m &= (df.parallax.values > 0) * np.isfinite(df.parallax.values)
    m &= df.astrometric_excess_noise.values < 5
    print(len(df.iloc[m]), "stars after S/N cuts")

    # Jason's wide binary cuts
    # m &= df.astrometric_excess_noise.values > 0
    # m &= df.astrometric_excess_noise_sig.values > 6

    # Jason's short-period binary cuts
    # m &= radial_velocity_error < 4
    # print(len(df.iloc[m]), "stars after Jason's binary cuts")
    # assert 0

    df = df.iloc[m]
    return df


def deredden(df):
    print("Loading Dustmaps")
    bayestar = BayestarQuery(max_samples=2, version='bayestar2019')

    print("Calculating Ebv")
    coords = SkyCoord(df.ra.values*units.deg, df.dec.values*units.deg,
                    distance=df.r_est.values*units.pc)

    ebv, flags = bayestar(coords, mode='percentile', pct=[16., 50., 84.],
                        return_flags=True)

    # Calculate Av
    Av_bayestar = 2.742 * ebv
    print(np.shape(Av_bayestar), "shape")
    Av = Av_bayestar[:, 1]
    Av_errm = Av - Av_bayestar[:, 0]
    Av_errp = Av_bayestar[:, 2] - Av
    Av_std = .5*(Av_errm + Av_errp)

    # Catch places where the extinction uncertainty is zero and default to an
    # uncertainty of .05
    m = Av_std == 0
    Av_std[m] = .05

    df["ebv"] = ebv[:, 1]  # The median ebv value.
    df["Av"] = Av
    df["Av_errp"] = Av_errp
    df["Av_errm"] = Av_errm
    df["Av_std"] = Av_std

    # Calculate dereddened photometry
    AG, Abp, Arp = getDust(df.phot_g_mean_mag.values,
                        df.phot_bp_mean_mag.values,
                        df.phot_rp_mean_mag.values, df.ebv.values)

    df["bp_dered"] = df.phot_bp_mean_mag.values - Abp
    df["rp_dered"] = df.phot_rp_mean_mag.values - Arp
    df["bprp_dered"] = df["bp_dered"] - df["rp_dered"]
    df["G_dered"] = df.phot_g_mean_mag.values - AG

    abs_G = mM(df.G_dered.values, df.r_est)
    df["abs_G"] = abs_G

    return df


# Calculate Absolute magntitude
def mM(m, D):
    return 5 - 5*np.log10(D) + m


def remove_nans_binaries_subgiants(df, plot=False):

    # Remove NaNs
    m2 = np.isfinite(df.abs_G.values)
    df = df.iloc[m2]

    # Remove binaries
    x = df.bp_dered - df.rp_dered
    y = df.abs_G

    AT = np.vstack((x**6, x**5, x**4, x**3, x**2, x, np.ones_like(x)))
    ATA = np.dot(AT, AT.T)
    w = np.linalg.solve(ATA, np.dot(AT, y))

    minb, maxb, extra = 0, 2.2, .27
    xs = np.linspace(minb, maxb, 1000)
    subcut = 4.

    m = (minb < x) * (x < maxb)
    m &= (y < np.polyval(w, x) - extra) + (subcut > y)
    flag = np.zeros(len(df))
    flag[~m] = np.ones(len(flag[~m]))
    df["flag"] = flag

    test = df.iloc[df.flag.values == 1]

    if plot:
        plt.plot(df.bp_dered - df.rp_dered, df.abs_G, ".", alpha=.1)
        plt.plot(test.bp_dered - test.rp_dered, test.abs_G, ".", alpha=.1)
        plt.ylim(10, 1)
        plt.savefig("test")
    return df


def add_phot_teff(df):
    # Calculate photometric Teff
    teffs = bprp_to_teff(df.bp_dered - df.rp_dered)
    df["color_teffs"] = teffs
    return df


def add_gyro_ages(df, plot=False):
    print("Calculating gyro ages")
    logages = []
    for i, p in enumerate(df.Prot.values):
        logages.append(age_model(np.log10(p), df.phot_bp_mean_mag.values[i] -
                                df.phot_rp_mean_mag.values[i]))
    df["log_age"] = np.array(logages)
    df["age"] = (10**np.array(logages))*1e-9

    if plot:
        plt.figure(figsize=(16, 9), dpi=200)
        singles = df.flag.values == 1
        plt.scatter(df.bprp_dered.values[singles], df.abs_G.values[singles],
                    c=df.age.values[singles], vmin=0, vmax=5, s=50, alpha=.2,
                    cmap="viridis", rasterized=True, edgecolor="none")
        plt.xlabel("$\mathrm{G_{BP}-G_{RP}~[dex]}$")
        plt.ylabel("$\mathrm{G~[dex]}$")
        plt.colorbar(label="$\mathrm{Gyrochronal~age~[Gyr]}$")
        plt.ylim(11, 5.5)
        plt.xlim(.8, 2.7);
        plt.savefig("age_gradient")
    return df


if __name__ == "__main__":
    print("Load data")
    # df = pd.read_pickle("../data/Mc_Gar_Sant")
    df = load_and_merge_data()
    print(np.shape(df))
    print(len(df), "stars")

    # S/N cuts
    df = sn_cuts(df)

    # Get dust and redenning.
    df = deredden(df)

    # Flag subgiants and binaries.
    df = remove_nans_binaries_subgiants(df)

    # Calculate photometric temperatures.
    df = add_phot_teff(df)

    # Calculate gyro ages
    df = add_gyro_ages(df)

    print(len(df), "stars")

    print("Saving file")
    df.to_csv("../data/mc_san_gaia_lam.csv")
