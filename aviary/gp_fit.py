import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo

from tqdm import trange
import pickle


def log_period_model(x2, x2_min, log_period_break_m, log_period_break_b):
    """
    2nd-order polynomial describing relationship between period and
    age. period = m*logage + b
    """
    return log_period_break_m * (x2 - x2_min) + log_period_break_b


# Mean model
def gyro_mean_model(x1, x2, log_period_break_m, log_period_break_b,
                    teff_break, slope_low, slope_high, smooth):
    """
    Predict the period at given values of color and age.
    Use a smoothed broken power law for the period - color relation.
    Use get_log_period_break for the period - age relation.
    """
    delta = x1 - teff_break
    brk = log_period_model(x2)  # Get the period at this age
    slope = slope_low / (1 + tt.exp(smooth * delta)) \
        + slope_high / (1 + tt.exp(-smooth * delta))
    return slope * delta + brk


def fit_gp(x, age, prot, prot_err, filename):
    """
    Fit a GP gyro model to data via optimization. Saves a pickle of the model.

    Args:
        x (array): The Gaia bp-rp colors or Teffs of stars.
        age (array): The (linear) ages of stars in Gyr.
        prot (array): The (linear) rotation periods of stars in days.
        prot_err (array): The absolute uncertainties on rotation periods in
            days.
        filename (str): The name of the pickle file to save the model in.
    """

    # Format data for GP fit.
    inds = np.argsort(x)
    x1 = np.array(x[inds])
    x2 = np.log(np.array(age[inds]))
    y = np.log(np.array(prot[inds]))
    y_err = prot_err[inds]/prot[inds]

    # Create variables used in the fitting process.
    mu1 = np.mean(x1)
    sd1 = np.std(x1)
    mu2 = np.mean(x2)
    sd2 = np.std(x2)

    x2_min = np.min(x2)

    xp1 = np.linspace(x1.min() - .2, x1.max() + .2, 1000)
    xp2 = np.linspace(x2.min(), x2.max(), 1000)
    xg1 = np.linspace(x1.min(), x1.max(), 5)
    xg2 = np.linspace(x2.min(), x2.max(), 10)
    xg2 = np.array([np.log(.12), np.log(.67), np.log(1), np.log(1.6),
                    np.log(2.7), np.log(4), np.log(4.56), np.log(8),
                    np.log(10), np.log(14)])

    # The PyMC3 model
    with pm.Model() as model:
        # x1 is color, x2 is age, y is period

        # Parameters to infer.
        # the break in temperature (actually color)
        teff_break = pm.Normal("teff_break", mu=0.9, sigma=.3)
        log_period_break_m = pm.Normal("log_period_break_m", mu=0.0, sd=5)
        log_period_break_b = pm.Normal("log_period_break_b", mu=np.log(10),
                                       sd=5)  # The constant term
        # The smoothness of the break. (lower is smoother)
        log_smooth = pm.Normal("log_smooth", mu=np.log(0.01), sigma=10.0)
        smooth = tt.exp(log_smooth)
        # The slope below the break
        slope_low = pm.Normal("slope_low", mu=0.0, sd=10.0)
        # The slope above the break
        slope_high = pm.Normal("slope_high", mu=0.0, sd=10.0)
        # The log-variance of the rotation period data.
        log_s2 = pm.Normal("log_s2", mu=1.0, sd=10.0)

        def get_log_period_break(x2):
            """
            2nd-order polynomial describing relationship between period and
            age. period = m*logage + b
            """
            return log_period_break_m * (x2 - x2_min) + log_period_break_b

        # Mean model
        def get_mean_model(x1, x2):
            """
            Predict the period at given values of color and age.
            Use a smoothed broken power law for the period - color relation.
            Use get_log_period_break for the period - age relation.
            """
            delta = x1 - teff_break
            brk = get_log_period_break(x2)  # Get the period at this age
            slope = slope_low / (1 + tt.exp(smooth * delta)) \
                + slope_high / (1 + tt.exp(-smooth * delta))
            return slope * delta + brk

        mean_model = get_mean_model(x1, x2)
        pm.Deterministic("mean_model", mean_model)

        # GP parameters
        log_amp = pm.Normal("log_amp", mu=np.log(np.var(y)), sigma=10.0)
        log_ell = pm.Normal("log_ell1", mu=0.0, sigma=10., shape=2)

        def get_K(x1, x2, xp1=None, xp2=None):
            X = np.vstack(((x1 - mu1) / sd1, (x2 - mu2) / sd2))

            if xp1 is None:
                dX = (X[:, :, None] - X[:, None, :]) \
                    * tt.exp(-log_ell)[:, None, None]
                r2 = tt.sum(dX ** 2, axis=0)
            else:
                Xp = tt.stack(((xp1 - mu1) / sd1, (xp2 - mu2) / sd2))
                dX = (Xp[:, :, None] - X[:, None, :]) \
                    * tt.exp(-log_ell)[:, None, None]
                r2 = tt.sum(dX ** 2, axis=0)

            K = tt.exp(log_amp - 0.5 * r2)
            return K

        K = get_K(x1, x2)
        K = tt.inc_subtensor(K[np.diag_indices(len(y))],
                             tt.exp(log_s2) + y_err)

        alpha = tt.slinalg.solve(K, y - mean_model)
        pm.Deterministic("alpha", alpha)
        for i, x2_ref in enumerate(xg2):
            pred_model = get_mean_model(xp1, x2_ref)
            Kp = get_K(x1, x2, xp1, x2_ref + np.zeros_like(xp1))
            pred = tt.dot(Kp, alpha) + pred_model
            pm.Deterministic("pred_{0}".format(i), pred)

        # Likelihood
        pm.MvNormal("obs", mu=mean_model, cov=K, observed=y)

        x1_test = pm.Flat("x1_test", shape=(5000,))
        x2_test = pm.Flat("x2_test", shape=(5000,))
        K_test = get_K(x1, x2, x1_test, x2_test)
        y_test = pm.Deterministic("y_test", tt.dot(K_test, alpha)
                                  + get_mean_model(x1_test, x2_test))

        model.x1 = x1
        model.x2 = x2
        model.x2_min = x2_min
        model.mu1 = mu1
        model.mu2 = mu2
        model.sd1 = sd1
        model.sd2 = sd2

        map_soln = model.test_point
        map_soln = xo.optimize(map_soln, [slope_low, slope_high])
        map_soln = xo.optimize(map_soln, [log_smooth])
        map_soln = xo.optimize(map_soln, [teff_break, log_period_break_m,
                                          log_period_break_b])
        map_soln = xo.optimize(map_soln, [slope_low, slope_high, log_smooth])
        map_soln = xo.optimize(map_soln, [log_s2, log_amp, log_ell])
        map_soln = xo.optimize(map_soln, [log_ell, log_amp, log_s2,
                                          slope_high, slope_low, log_smooth,
                                          log_period_break_b,
                                          log_period_break_m, teff_break])

    with open(filename, "wb") as f:
        pickle.dump([model, map_soln], f)

    return map_soln
