import scipy
from scipy import stats
import numpy
from matplotlib import pyplot

import convert
import profiles
from macro import *


def gas_density_betamodel_wrapper(parms, x):
    """ Wrapper around profiles.gas_density_betamodel(r, rho0, beta, rc, rcut, do_cut)
            r:     radius, int or array
            rho0:  baryonic matter central denisty, float
            beta:  ratio specific kinetic energy of galaxies to gas; slope, float
            rc:    core radius (profile is constant within rc), float
            rcut:  numerical cutoff: keep local baryon fraction above unity, float
        @param parms: tuple that contains (rho0, beta, rc)
        @param x:     independent variable, here the radius r [kpc]
        @return:      NFW DM density profile rho(r), int or array """

    return profiles.gas_density_betamodel(x, parms[0], parms[1], parms[2], numpy.nan, do_cut=False)


# Define the statistical model, in this case we shall use a chi-squared distribution, assuming normality in the errors
def stat(parms, x, y, dy):
    """ Statistical model. Here: Pearson's chi^2 assuming normality in errors
        We fit the betamodel to the observed Chandra radial density profile
        @param parms: fit parameters rho0 [g/cm^3], beta, rc [kpc], tuple
        @param x:     independent variable, here the radius r [kpc]
        @param y:     dependent variable, here the gas density
        @param dy:    uncertainty in the dependent variable (here gas density)
        @return:      chi^2 given the model parameters
    """

    # ymodel = gas_density_betamodel_wrapper(parms, x)
    ymodel = profiles.gas_density_betamodel(x, parms[0], parms[1], parms[2], numpy.nan, do_cut=False)
    chisq = numpy.sum(p2(y - ymodel) / p2(dy))
    return chisq


def betamodel_to_chandra(c, verbose=False):
    """ Fit betamodel to Chandra observation. Beta is a free parameter.
        NB rcut is not a free parameter as it is induced for numerical reasons!
        @param c:  ObservedCluster
        @return:   """

    # Set initial guess and bounds for both CygA and for CygNW
    if c.name == "cygA":
        parms=[0.1, 0.67, 10]
        bounds = [(None, None), (0.0, 1.0), (None, None)]
    if c.name == "cygNW":
        bounds = [(0.002, 0.003), (0.0, 1.0), (None, None)]
        parms=[1.0, 1.0, 1.0]

    # Minimise chi^2 to obtain best-fit parameters
    result = scipy.optimize.minimize(stat, parms,
            args=(c.avg["r"], c.avg["n"], c.avg["fn"]),
            method='L-BFGS-B', bounds=bounds)

    ml_vals = result["x"]
    ml_func = result["fun"]

    # Obtain degrees of freedom and check goodness-of-fit. This is useless tho
    moddof = len(ml_vals)  # Model degrees of freedom; nr of fit parameters
    # Here we count the unmasked values. NB id(i) != id(True)
    dof = len([i for i in c.avg["n"].mask if i == False]) - moddof  # degrees of freedom
    ch = scipy.stats.chi2(dof)
    pval = 1.0 - ch.cdf(ml_func)

    # Obtain MLEs using Scipy's curve_fit which gives covariance matrix
    ml_vals, ml_covar = scipy.optimize.curve_fit(profiles.gas_density_betamodel,
            c.avg["r"], c.avg["n"], p0=ml_vals, sigma=c.avg["fn"])

    if not result["success"]:
        print "  scipy.optimize.curve_fit broke down!\n    Reason: '{0}'"\
            .format(result["message"])
        print "  No confidence intervals have been calculated."

    err = numpy.sqrt(numpy.diag(ml_covar))

    if verbose:
        print c.name
        print "Results for the 'free beta-model' model:"
        print "  Using scipy.optimize.minimize to minimize chi^2 yields:"
        print "    n_e,0       = {0:.5f}".format(ml_vals[0])
        print "    r_c         = {0:.5f}".format(ml_vals[1])
        print "    beta        = {0:.5f}".format(ml_vals[2])
        print "    chisq       = {0:.5f}".format(ml_func)
        print "    dof         = {0:.5f}".format(dof)
        print "    chisq/dof   = {0:.5f}".format(ml_func/dof)
        print "    p-value     = {0:.5f}".format(pval)
        print "  Using scipy.optimize.curve_fit to obtain confidence intervals yields:"
        print "    n_e,0       = {0:.5f} +/- {1:.5f}".format(ml_vals[0], err[0])
        print "    r_c         = {0:.5f} +/- {1:.5f}".format(ml_vals[1], err[1])
        print "    beta        = {0:.5f} +/- {1:.5f}".format(ml_vals[2], err[2])
        print

    return ml_vals, err
