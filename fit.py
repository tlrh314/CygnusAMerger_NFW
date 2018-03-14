# -*- coding: utf-8 -*-

import scipy
from scipy import stats
import numpy
from matplotlib import pyplot

import convert
import profiles
import plot
from macro import *


def gas_density_betamodel_wrapper(parms, x):
    """ Wrapper around profiles.gas_density_betamodel(r, rho0, beta, rc, rcut)
            r:     radius, int or array
            rho0:  baryonic matter central denisty, float
            beta:  ratio specific kinetic energy of galaxies to gas; slope, float
            rc:    core radius (profile is constant within rc), float
            rcut:  numerical cutoff: keep local baryon fraction above unity, float
        @param parms: tuple that contains (rho0, beta, rc)
        @param x:     independent variable, here the radius r [kpc]
        @return:      NFW DM density profile rho(r), int or array """

    return profiles.gas_density_betamodel(x, parms[0], parms[1], parms[2], None)


# Define the statistical model, in this case we shall use a chi-squared distribution, assuming normality in the errors
def stat(parms, x, y, dy):
    """ Statistical model. Here: Pearson's chi^2 assuming normality in errors
        We fit the betamodel to the observed Chandra radial density profile
        @param parms: fit parameters rho0 [g/cm^3], beta, rc [kpc], tuple
        @param x:     independent variable, here the radius r [kpc]
        @param y:     dependent variable, here the gas density
        @param dy:    uncertainty in the dependent variable (here gas density)
        @return:      chi^2 given the model parameters """

    # ymodel = gas_density_betamodel_wrapper(parms, x)
    ymodel = profiles.gas_density_betamodel(x, parms[0], parms[1], parms[2], None)
    chisq = numpy.sum(p2(y - ymodel) / p2(dy))
    return chisq


def betamodel_to_chandra(c, verbose=False, debug=True):
    """ Fit betamodel to Chandra observation. Beta is a free parameter.
        NB rcut is not a free parameter as it is induced for numerical reasons!
        @param c:  ObservedCluster
        @return:   (MLE, one sigma confidence interval), tuple """

    # Set initial guess and bounds for both CygA and for CygNW
    if c.name == "cygA":
        parms=[0.1, 0.5, 20]
        bounds = [(0.0001, 0.3), (0.0, 1.0), (0, 50)]
    if c.name == "cygNW":
        bounds = [(0.0001, 0.3), (0.0, 1.0), (150, 300)]
        parms=[0.003, 0.5, 200]

    # Minimise chi^2 to obtain best-fit parameters
    result = scipy.optimize.minimize(stat, parms,
            args=(numpy.array(c.avg["r"]), numpy.array(c.avg["n"]),
                numpy.array(c.avg["fn"])), method='L-BFGS-B', bounds=bounds)

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
            numpy.array(c.avg["r"]), numpy.array(c.avg["n"]), p0=ml_vals,
            sigma=numpy.array(c.avg["fn"]), method="trf", bounds=zip(*bounds))

    print result["x"], "(minimise chisq)"
    print ml_vals, "(curve_fit)"

    if debug:
        avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
                "elinewidth": 1, "label": "data ({0})".format(c.data) }

        # pyplot.switch_backend("Qt5Agg")
        fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
        c.plot_chandra_average(parm="rho", ax=ax, style=avg)

        # NB at this point still uncut betamodel!
        pyplot.plot(c.ana_radii, profiles.gas_density_betamodel(c.ana_radii,
            convert.ne_to_rho(result["x"][0]), result["x"][1], result["x"][2]),
            label=r"minimise $\chi_2$")
        pyplot.plot(c.ana_radii, profiles.gas_density_betamodel(c.ana_radii,
            convert.ne_to_rho(ml_vals[0]), ml_vals[1], ml_vals[2]),
            label="curvefit")

        pyplot.xlabel("Radius (kpc)")
        pyplot.ylabel("Density (gm/cm$^3$)")
        pyplot.xscale("log")
        pyplot.yscale("log")
        pyplot.xlim(1, 900)
        pyplot.ylim(1e-28, 2e-25)
        pyplot.legend(loc="lower left")
        pyplot.savefig("out/fit/DEBUG_{0}_{1}.pdf".format(c.name, c.data))
        # pyplot.show()

    if not result["success"]:
        print "  scipy.optimize.curve_fit broke down!\n    Reason: '{0}'"\
            .format(result["message"])
        print "  No confidence intervals have been calculated."
        print ml_vals, ml_covar
        import sys; sys.exit(1)

    err = numpy.sqrt(numpy.diag(ml_covar))

    if verbose:
        print c.name
        print "Results for the 'free beta-model' model:"
        print "  Using scipy.optimize.minimize to minimize chi^2 yields:"
        print "    n_e,0       = {0:.5f}".format(ml_vals[0])
        print "    beta        = {0:.5f}".format(ml_vals[1])
        print "    r_c         = {0:.5f}".format(ml_vals[2])
        print "    chisq       = {0:.5f}".format(ml_func)
        print "    dof         = {0:.5f}".format(dof)
        print "    chisq/dof   = {0:.5f}".format(ml_func/dof)
        print "    p-value     = {0:.5f}".format(pval)
        print "  Using scipy.optimize.curve_fit to obtain confidence intervals yields:"
        print "    n_e,0       = {0:.5f} +/- {1:.5f}".format(ml_vals[0], err[0])
        print "    beta        = {0:.5f} +/- {1:.5f}".format(ml_vals[1], err[1])
        print "    r_c         = {0:.4f} +/- {1:.4f}".format(ml_vals[2], err[2])
        print

    return ml_vals, err


def total_gravitating_mass(c, cNFW=None, bf=0.17, RCUT_R200_RATIO=None,
                           verbose=False, debug=False):
    """ Find total gravitating mass under assumption of fixed baryon fraction
        at the virial radius (seventeen percent) r200.
        The problem is implicit and solved by root-finding (bisection).
        @param c   : ObservedCluster
        @param cNFW: Concentration parameter. If given, cNFW is a free parameter
                     instead of using the Duffy+ 2008 relation for c(M200, z)
        @param bf  : Baryon fraction. Default 17% within R200.
        @param RCUT_R200_RATIO
                   : rcut = RCUT_R200_RATIO * r200 to cut-off betamodel and NFW
        @return    : Dictionary of best-fit halo properties. """

    # Set bestfit betamodel parameters
    ne0, rho0, beta, rc = c.ne0, c.rho0, c.beta, c.rc
    rc *= convert.kpc2cm

    # Find r200 such that rho200 / rho_crit == 200 (True by definition)
    lower = 10 * convert.kpc2cm
    upper = 4000 * convert.kpc2cm

    # bisection method
    epsilon = 0.001
    while upper/lower > 1+epsilon:
        # bisection
        r200 = (lower+upper)/2.

        # bf = 0.17  # Critical assumption: bf == 0.17 at r200 (Planelles+ 2013)

        if RCUT_R200_RATIO is None:
            rcut = None
        else:
            rcut = r200 * RCUT_R200_RATIO
        Mgas200 = profiles.gas_mass_betamodel(r200, rho0, beta, rc, rcut)
        Mdm200 = Mgas200 * (1/bf - 1)

        M200 = Mgas200 + Mdm200

        if not cNFW: cNFW = profiles.cNFW(M200)
        rs = r200 / cNFW

        rho0_dm = Mdm200 / profiles.dm_mass_nfw(r200, 1, rs)

        """ Now rho_average(r200)/rhocrit should equal 200.
                If not? Try different r200"""
        rho200_over_rhocrit = ( M200 / (4./3 * numpy.pi * p3(r200))) / c.cc.rho_crit()
        if debug:
            print "Lower                  = {0:3.1f}".format(lower * convert.cm2kpc)
            print "r200                   = {0:3.1f}".format(r200 * convert.cm2kpc)
            print "Upper                  = {0:3.1f}".format(upper * convert.cm2kpc)
            print "Ratio                  = {0:.1f}".format(rho200_over_rhocrit/200)
            print

        # bisection
        if rho200_over_rhocrit < 200:
            upper = r200
        if rho200_over_rhocrit > 200:
            lower = r200

    # r200, thus M200 found
    halo = dict()
    halo["r200"] = r200 * convert.cm2kpc
    halo["rcut"] = rcut * convert.cm2kpc if rcut is not None else None
    halo["rho200_over_rhocrit"] = rho200_over_rhocrit
    halo["bf200"] = Mgas200/(Mdm200+Mgas200)
    halo["rho0"] = rho0
    halo["ne0"] = convert.rho_to_ne(rho0)
    halo["rc"] = rc * convert.cm2kpc
    halo["beta"] = beta
    halo["Mgas200"] = Mgas200 * convert.g2msun
    halo["Mdm200"] = Mdm200 * convert.g2msun
    halo["M200"] = M200 * convert.g2msun
    halo["cNFW"] = cNFW
    halo["rs"] = rs * convert.cm2kpc
    halo["rho0_dm"] = rho0_dm
    halo["ne0_dm"] = convert.rho_to_ne(rho0_dm)

    return halo


def find_r500(c, debug=False):
    """ Find the radius r500, therefore M500 and T500
        The problem is implicit and solved by root-finding (bisection).
        @param c   : ObservedCluster
        @return    : Tuple of parameters """

    # Set bestfit betamodel parameters
    ne0, rho0, beta, rc = c.ne0, c.rho0, c.beta, c.rc
    rc *= convert.kpc2cm
    rcut_cm = c.rcut_cm if hasattr(c, "rcut_cm") and c.rcut_cm is not None else None
    rcut_nfw_cm = c.rcut_nfw_cm if hasattr(c, "rcut_nfw_cm") and c.rcut_nfw_cm is not None else None

    # Set inferred NFW parameters
    rho0_dm, rs = c.halo["rho0_dm"], c.halo["rs"]
    rs *= convert.kpc2cm

    # Find r500 such that rho500 / rho_crit == 500 (True by definition)
    lower = 10 * convert.kpc2cm
    upper = 2000 * convert.kpc2cm

    # bisection method
    epsilon = 0.001
    while upper/lower > 1+epsilon:
        # bisection
        r500 = (lower+upper)/2.

        M500 = profiles.dm_mass_nfw(r500, rho0_dm, rs, rcut_nfw_cm) + \
            profiles.gas_mass_betamodel(r500, rho0, beta, rc, rcut_cm)

        """ Now rho_average(r200)/rhocrit should equal 200.
                If not? Try different r200"""
        rho500_over_rhocrit = ( M500 / (4./3 * numpy.pi * p3(r500))) / c.cc.rho_crit()
        if debug:
            print "Lower                  = {0:3.1f}".format(lower * convert.cm2kpc)
            print "r500                   = {0:3.1f}".format(r500 * convert.cm2kpc)
            print "Upper                  = {0:3.1f}".format(upper * convert.cm2kpc)
            print "Ratio                  = {0:.1f}".format(rho500_over_rhocrit/500)
            print

        # bisection
        if rho500_over_rhocrit < 500:
            upper = r500
        if rho500_over_rhocrit > 500:
            lower = r500

    return r500 * convert.cm2kpc, M500 * convert.g2msun


def smith_centrally_decreasing_temperature(c):
    """ Smith+ (2002; 4) Centrally decreasing expression for T(r)

        Bestfit Smith: a = 7.81 keV, b = 7.44 keV, and c = 76.4 kpc
        @param c: ObservedCluster
        @return : (MLE, one sigma confidence interval), tuple """

    ml_vals, ml_covar = scipy.optimize.curve_fit(
        profiles.smith_centrally_decreasing_temperature, c.avg["r"],
        c.avg["kT"], sigma=c.avg["fkT"], p0=[7.81, 7.44, 76.4])

    return ml_vals, numpy.sqrt(numpy.diag(ml_covar))


def temperature_wrapper(c, cNFW, bf, RCUT_R200_RATIO=None):
    print "Trying cNFW = {0}, bf = {1}, RCUT_R200_RATIO = {2}".format(cNFW, bf, RCUT_R200_RATIO)
    c.infer_NFW_mass(cNFW=cNFW, bf=bf, RCUT_R200_RATIO=RCUT_R200_RATIO)

    # Use c.avg["r"] because chi^2 fitting with observed profile
    radii = c.avg["r"]; N = len(radii); temperature = numpy.zeros(N)
    print(N)
    print(radii.mask)
    print(radii)

    infinity = 1e25
    for i, r in enumerate(radii * convert.kpc2cm):
        if not r: continue  # to skip masked values
        temperature[i] = profiles.hydrostatic_temperature(
            r, infinity, c.rho_gas, c.M_tot)

    print(temperature)
    sjenk = raw_input("Press sjenk to continue")

    # We also set the inferred temperature (with c.ana_radii) for plotting
    c.set_inferred_temperature(verbose=True)
    c.fit_counter += 1
    plot.inferred_temperature(c)
    plot.donnert2014_figure1(c)
    return convert.K_to_keV(temperature)


def total_gravitating_mass_freecbf(c, do_cut=True, verbose=False):
    """ Fit 'total_gravitating_mass' to temperature /w cNFW, bf free.
        @param c  : ObservedCluster
        @param cut: also fit the cut-off radius
        @return   : (MLE, one sigma confidence interval), tuple """

    print "Fitting cNFW, bf to retrieve T_HE = Tobs"
    if c.data == "1Msec":
        if c.name == "cygA":
            p0 = [10, 0.07] if not do_cut else [10, 0.07, 0.75]
            bounds = ((0, 0), (40, 0.25)) if not do_cut else ((0, 0, 0), (40, 0.25, 2))
        if c.name == "cygNW":
            p0 = [5, 0.17] if not do_cut else [5, 0.17, 1.1]
            bounds = ((0, 0), (400, 0.25)) if not do_cut else ((0, 0, 0), (400, 0.25, 2))
    else:
        if c.name == "cygA":
            p0 = [10.12, 0.134] if not do_cut else [8, 0.10, 0.6]
            bounds = ((0, 0), (13, 0.25)) if not do_cut else ((0, 0, 0), (13, 0.25, 2))
        if c.name == "cygNW":
            p0 = [12.93, 0.139] if not do_cut else [5, 0.17, 1.1]
            bounds = ((0, 0), (13, 0.25)) if not do_cut else ((0, 0, 0), (13, 0.25, 2))

    c.fit_counter = 0
    ml_vals, ml_covar = scipy.optimize.curve_fit(lambda r, parm0, parm1, parm2=None:
        temperature_wrapper(c, parm0, parm1, parm2 if do_cut else None),
        c.avg["r"], c.avg["kT"], p0=p0, sigma=c.avg["fkT"],
        method="trf", bounds=bounds)
    c.fit_counter = None

    return ml_vals, numpy.sqrt(numpy.diag(ml_covar))


if __name__ == "__main__":
    from main import new_argument_parser
    from main import infer_toycluster_ics
    from main import set_observed_clusters
    from plot import inferred_temperature

    a, unknown = new_argument_parser().parse_known_args()
    a.do_cut = True
    print(a)
    cygA, cygNW = infer_toycluster_ics(a)
    # cygA, cygNW = set_observed_clusters(a)

    pyplot.switch_backend("Qt5Agg")
    inferred_temperature(cygA)
    inferred_temperature(cygNW)

    # total_gravitating_mass_freecbf()
