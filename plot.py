# -*- coding: utf-8 -*-

import copy

import numpy
import scipy
from scipy import ndimage
import astropy
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
import matplotlib
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from line_profiler_support import profile
from deco import concurrent, synchronized
threads=2

import colorcet

import fit
import profiles
import convert
from macro import print_progressbar, p2
from timer import Timer


# ----------------------------------------------------------------------------
# Plots for Chandra observations
# ----------------------------------------------------------------------------
def quiescent_parm(c, parm="rho"):
    """ @param c:  ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]",
                  "Yparm": "Compton-Y Parameter]" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available".format(parm)
        return

    pyplot.figure(figsize=(12, 9))
    ax = pyplot.gca()
    c.plot_chandra_average(ax, parm=parm, style=avg)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="best")
    pyplot.tight_layout()
    pyplot.show()
    # pyplot.savefig("out/{0}_quiescent_{1}.pdf".format(c.name, parm), dpi=150)


def sector_parm(c, parm="kT"):
    """ @param c:  ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 2, "label": "Average "+parm }
    merger = { "marker": "o", "ls": "", "c": "g", "ms": 4, "alpha": 1,
               "elinewidth": 2, "label": "Merger "+parm }
    hot = { "marker": "o", "ls": "", "c": "r", "ms": 4, "alpha": 0.5,
            "elinewidth": 2, "label": "Hot "+parm }
    cold =  { "marker": "o", "ls": "", "c": "purple", "ms": 4, "alpha": 0.5,
              "elinewidth": 2, "label": "Cold "+parm }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available".format(parm)
        return

    fig = pyplot.figure(figsize=(12, 9))
    ax = pyplot.gca()
    c.plot_chandra_average(ax, parm=parm, style=avg)
    c.plot_chandra_sector(ax, parm=parm, merger=True, style=merger)
    c.plot_chandra_sector(ax, parm=parm, hot=True, style=hot)
    c.plot_chandra_sector(ax, parm=parm, cold=True, style=cold)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")  # Linear better shows merger-affected radii
    pyplot.xlim(3, 2000)
    if parm == "kT":
        pyplot.ylim(2, 12)
    else:
        pyplot.yscale("log")
    pyplot.legend(loc="best", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_sector_{1}.pdf".format(c.name, parm), dpi=150)


def chandra_coolingtime(c):
    """ @param c:  ObservedCluster """
    Tcool = profiles.sarazin_coolingtime(c.avg["n"]/u.cm**3, c.avg["T"])

    pyplot.figure(figsize=(12, 9))
    pyplot.plot(c.avg["r"], Tcool)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Cooling Time [yr]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.savefig("out/{0}_cooling_time.pdf".format(c.name), dpi=150)
    pyplot.tight_layout()


def delta_kT(cygA):
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 1, "alpha": 0.3,
            "elinewidth": 1, "label": "avg"}
    merger = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 0.3,
            "elinewidth": 1, "label": "merger" }

    cygA.avg.mask = [False for i in range(len(cygA.avg.columns))]
    cygA.merger.mask = [False for i in range(len(cygA.merger.columns))]
    r_avg = cygA.avg["r"]
    r_merger = cygA.merger["r"]

    kT_avg = cygA.avg["kT"]
    kT_merger = cygA.merger["kT"]

    for k in range(1, 6):
        for s in range(1, 16):
            fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(8, 10))

            # k=3 --> cubic spline. s is some magical smoothing factor
            average_spline = scipy.interpolate.UnivariateSpline(r_avg, numpy.array(kT_avg), k=3, s=s)
            merger_spline = scipy.interpolate.UnivariateSpline(r_merger, numpy.array(kT_merger), k=3, s=s)

            cygA.plot_chandra_sector(merger=True, parm="kT", ax=ax1, style=merger)
            cygA.plot_chandra_average(parm="kT", ax=ax1, style=avg)

            ax1.plot(cygA.avg["r"], average_spline(cygA.avg["r"]))
            ax1.plot(cygA.avg["r"], merger_spline(cygA.avg["r"]))

            ax1.set_xlim(9, 1000)
            ax1.set_ylim(4, 12)
            ax1.set_xscale("log")
            ax1.set_xticks([10, 100, 1000])
            ax1.set_xticklabels([r"$10^1$", r"$10^2$", r"$10^3$"])
            ax1.set_ylabel("kT [keV]")


            delta_kT = (merger_spline(cygA.avg["r"]) - average_spline(cygA.avg["r"]))
            ax2.plot(cygA.avg["r"], delta_kT)

            ax2.set_xlim(9, 1000)
            ax2.set_ylim(0, 2.5)
            ax2.set_xscale("log")
            ax2.set_xticks([10, 100, 1000])
            ax2.set_xticklabels([r"$10^1$", r"$10^2$", r"$10^3$"])
            ax2.set_xlabel("R [kpc]")
            ax2.set_ylabel(r"$\Delta$kT [keV]")

            pyplot.savefig("out/cygA_delta_kT_k={0}_s={1:02d}.pdf".format(k, s))
            pyplot.close()


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Plots analytical models based on observation
# ----------------------------------------------------------------------------
def bestfit_betamodel(c):
    """ Plot best-fit betamodel with residuals """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "data ({0} Msec)".format("1.03" if c.data == "1Msec" else "2.2") }
    fit = { "color": "k", "lw": 4, "linestyle": "solid" }

    fig, (ax, ax_r) = pyplot.subplots(2, 2, sharex=True, figsize=(16, 12))
    gs1 = matplotlib.gridspec.GridSpec(3, 3)
    gs1.update(hspace=0)
    ax = pyplot.subplot(gs1[:-1,:])
    ax_r = pyplot.subplot(gs1[-1,:], sharex=ax)  # residuals

    # Plot Chandra observation and betamodel with mles
    pyplot.sca(ax)

    c.plot_chandra_average(ax, parm="rho", style=avg)
    c.plot_bestfit_betamodel(ax, style=fit)
    pyplot.ylabel("Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.ylim(numpy.min(c.avg["rho"])/1.5, numpy.max(c.avg["rho"])*1.3)
    pyplot.legend(loc="lower left", fontsize=22)

    # Plot residuals
    pyplot.sca(ax_r)
    c.plot_bestfit_residuals(ax_r)
    pyplot.axhline(y=0, lw=3, ls="dashed", c="k")
    pyplot.ylabel("Residuals [\%]")
    pyplot.xlabel("Radius [kpc]")
    pyplot.xscale("log")
    pyplot.xlim(2 if c.name == "cygA" else 20, 1e3 if c.name == "cygA" else 1.1e3)
    pyplot.ylim(-35, 35)

    # Fix for overlapping y-axis markers
    ax.tick_params(labelbottom="off")
    nbins = len(ax_r.get_yticklabels())
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="upper"))

    # Force axis labels to align
    ax.get_yaxis().set_label_coords(-0.07, 0.5)
    ax_r.get_yaxis().set_label_coords(-0.07, 0.5)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_bestfit_betamodel_{1}{2}.pdf".format(
        c.name, c.data, "_cut" if c.rcut_kpc is not None else ""), dpi=150)
    pyplot.sca(ax)


def inferred_nfw_profile(c):
    """ Plot the observed gas density, best-fit betamodel and the inferred
        best-fit NFW profile for the cluster
        @param c   : ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "data (2.2 Msec)" }
    fit = { "color": "k", "lw": 1, "linestyle": "dashed" }
    dm = { "color": "k", "lw": 1, "linestyle": "solid" }

    pyplot.figure(figsize=(12,9))
    ax = pyplot.gca()
    c.plot_chandra_average(ax, parm="rho", style=avg)
    c.plot_bestfit_betamodel(ax, style=fit)
    c.plot_inferred_nfw_profile(ax, style=dm)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e-32, 9e-24,
        facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=c.halo["r200"], c="k", lw=1)
    pyplot.text(c.halo["r200"]+100, 4e-24, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-32, ymax=9e-24)
    pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_inferred_nfw_cNFW={1:.3f}_bf={2:.4f}.png"
       .format(c.name, c.halo["cNFW"], c.halo["bf200"]), dpi=150)
    pyplot.close()


def inferred_mass(c):
    """ Plot the inferred mass profiles, both gas and dark matter.
        @param c: ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    gas = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "gas" }
    dm = { "color": "k", "lw": 1, "linestyle": "solid", "label": "DM" }

    pyplot.figure(figsize=(12, 9))
    ax = pyplot.gca()
    c.plot_bestfit_betamodel_mass(ax, style=gas)
    c.plot_inferred_nfw_mass(ax, style=dm)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass [Msun]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.legend(loc="best", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_hydrostatic-temperature_cNFW={1:.3f}_bf={2:.4f}.png"
       .format(c.name, c.halo["cNFW"], c.halo["bf200"]), dpi=150)
    pyplot.close()


def inferred_temperature(c):
    """ Plot the observed temperature profile and the inferred hydrostatic
        temperature for the best-fit betamodel and inferred total mass profile
        M_tot (spherically symmetric volume-integrated NFW plus ~ betamodel)
        @param c: ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 2, "label": "Average kT" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid" }

    fig, (ax0, ax1) = pyplot.subplots(1, 2, figsize=(16, 8))

    for ax in [ax0, ax1]:
        pyplot.sca(ax)
        c.plot_chandra_average(ax, parm="kT", style=avg)
        c.plot_inferred_temperature(ax, style=tot)
        ax.set_xlabel("Radius [kpc]")
        ax.set_ylabel("kT [keV]")
        ax.set_ylim(0.1, 12)

    ax0.set_xlim(1, 1000)
    ax1.set_xlim(1, 2000)
    ax1.set_xscale("log")
    pyplot.legend(loc="best", fontsize=22)
    pyplot.tight_layout()
    cut = "_rcut={0:1.3f}".format(c.halo["rcut"]) if c.halo["rcut"] is not None else ""
    pyplot.savefig("out/{0}{1}_hydrostatic-temperature{2}_cNFW={3:.3f}_bf={4:.4f}{5}.png"
       .format("fit/" if hasattr(c, "fit_counter") is not None else "", c.name,
               "_fit-{0:02d}".format(c.fit_counter) if hasattr(c, "fit_counter") else "",
               c.halo["cNFW"], c.halo["bf200"], cut), dpi=150)
    pyplot.close()


def inferred_pressure(c):
    """ Plot the observed pressure profile and the inferred hydrostatic
        pressure for the best-fit betamodel and inferred total mass profile
        M_tot (spherically symmetric volume-integrated NFW plus ~ betamodel)
        @param c: ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 2, "label": "Average P" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "tot" }

    pyplot.figure(figsize=(12, 9))
    ax = pyplot.gca()
    c.plot_chandra_average(ax, parm="P", style=avg)
    c.plot_inferred_pressure(ax, style=tot)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Pressure [erg/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.legend(loc="best", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_hydrostatic-pressure={1:.3f}_bf={2:.4f}.png"
       .format(c.name, c.halo["cNFW"], c.halo["bf200"]), dpi=150)
    pyplot.close()


def smith_hydrostatic_mass(c, debug=False):
    """ Smith+ (2002; eq. 3-5) Hydrostatic Mass from T(r), n_e(r)
        M(<r) = - k T(r) r^2/(mu m_p G) * (1/n_e * dn_e/dr + 1/T * dT/dr)

        Smith assumes:
            i) constant temperature, or
            ii) centrally decreasing temperature T(keV) = a - b exp[ - r(kpc)/c ]
                ==> 1/T dT/dr = 1/T b/c exp[-r/c]

        Method here: use best-fit betamodel, and its analytical derivative,
        for the temperature we fit a spline to the (smoothed) observed profile
        to obtain the temperature derivative from the spline

        @param c: ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "gray", "ms": 4, "alpha": 0.5,
            "elinewidth": 2 }
    ana = { "color": "k", "lw": 2, "linestyle": "solid", }

    pyplot.figure(figsize=(12, 9))
    pyplot.loglog(c.HE_radii*convert.cm2kpc, c.HE_M_below_r*convert.g2msun, **ana)
    pyplot.ylabel("Hydrostatic Mass [MSun]")
    pyplot.xlabel("Radius [kpc]")
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_hydrostaticmass.pdf".format(c.name), dpi=300)
    pyplot.close()

    if debug:
        matplotlib.rc("font", **{"size": 22})
        fig, (ax0, ax1, ax2, ax3) = pyplot.subplots(4, 1, sharex=True, figsize=(12, 18))
        gs1 = matplotlib.gridspec.GridSpec(4, 1)
        gs1.update(hspace=0)
        ax3 = pyplot.subplot(gs1[3])
        ax2 = pyplot.subplot(gs1[2], sharex=ax3)
        ax1 = pyplot.subplot(gs1[1], sharex=ax3)
        ax0 = pyplot.subplot(gs1[0], sharex=ax3)

        pyplot.sca(ax0); c.plot_chandra_average(ax0, parm="n", style=avg)
        ax0.loglog(c.HE_radii*convert.cm2kpc, c.HE_ne, **ana)
        ax0.set_ylabel("n$_e$(r) [1/cm$^3$]")
        ax1.semilogx(c.HE_radii*convert.cm2kpc, c.HE_dne_dr, **ana)
        ax1.set_ylabel("dn$_e$ / dr")
        pyplot.sca(ax2); c.plot_chandra_average(ax2, parm="T", style=avg)
        ax2.loglog(c.HE_radii*convert.cm2kpc, c.HE_T, **ana)
        ax2.set_yticks([3e7, 5e7, 7e7, 9e7])
        ax2.set_ylabel("T(r) [K]")
        ax3.semilogx(c.HE_radii*convert.cm2kpc, c.HE_dT_dr, **ana)
        ax3.set_ylabel("dT / dr")
        ax3.set_xlabel("Radius [kpc]")

        if c.name == "cygA":
            ax0.set_ylim(1e-4, 2e-1)
            ax1.set_ylim(-1e-24, 2e-26)
            ax2.set_ylim(3e7, 1e8)
            ax3.set_ylim(-0.8e-16, 3e-16)
        if c.name == "cygNW":
            ax0.set_ylim(5e-5, 4e-3)
            ax1.set_ylim(-3e-27, 1e-27)
            ax2.set_ylim(3e7, 1e8)
            ax3.set_ylim(-2e-17, 8e-19)

        for ax in [ax0, ax2]:  # show left
            ax.tick_params(labelleft=True, labelright=False)
        for ax in [ax1, ax3]:  # show right
            ax.tick_params(labelleft=False, labelright=True)
        for ax in [ax0, ax1, ax2]:  # only show xlabel on the lowest axis
            ax.tick_params(labelbottom="off")
        for ax in [ax0, ax1, ax2, ax3]:  # force ylabels to align
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0e"))
            ax.get_yaxis().set_label_coords(-0.12, 0.5)

        pyplot.xlim(-2, 1100)
        pyplot.tight_layout()
        pyplot.savefig("out/{0}_hydrostaticmass_debug.pdf".format(c.name), dpi=300)
        pyplot.close()
        matplotlib.rc("font", **{"size": 28})

    # Smith assumes a specific temperature structure and fits it to the data
    # Set smith to True to show this best-fit compares to 1 Msec Chandra
    smith = False
    if smith and c.name == "cygA":
        mle, err = fit.smith_centrally_decreasing_temperature(c)
        print "Smith (2002)", mle, err

        pyplot.figure(figsize=(12, 9))
        ax = pyplot.gca()
        c.plot_chandra_average(ax, parm="kT", style=avg)
        kT = profiles.smith_centrally_decreasing_temperature(
            c.avg["r"], mle[0], mle[1], mle[2])
        # Fit of the Smith temperature structure breaks for latest Tobs
        pyplot.plot(c.avg["r"], kT, label="my fit")

        mle = (7.81, 7.44, 76.4)  # best-fit in the Smith paper
        kT = profiles.smith_centrally_decreasing_temperature(
            c.avg["r"], mle[0], mle[1], mle[2])
        pyplot.plot(c.avg["r"], kT, label="Smith 2002")
        pyplot.xlabel("Radius [kpc]")
        pyplot.ylabel("Temperature [keV]")
        # pyplot.xscale("log")
        pyplot.legend(loc="best", fontsize=14)
        pyplot.savefig("out/{0}_smith_temperature.pdf".format(c.name), dpi=300)


# @profile
def donnert2014_figure1(c, add_sim=False, verlinde=False):
    """ Create Donnert (2014) Figure 1 for the Cygnus observation + best-fit models
        @param c     : ObservedCluster
        @param sim   : Simulation
        @param snapnr: TODO, string"""

    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1, "elinewidth": 1, "label": "data" }
    gas = { "color": "k", "lw": 1, "linestyle": "dotted", "label": "gas" }
    dm  = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "dm" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "tot" }

    fig, ((ax0, ax1), (ax2, ax3)) = pyplot.subplots(2, 2, figsize=(18, 16))

    c.plot_chandra_average(ax0, parm="rho", style=avg)
    c.plot_bestfit_betamodel(ax0, style=gas, rho=True)
    c.plot_inferred_nfw_profile(ax0, style=dm, rho=True)
    ax0.set_yscale("log")
    ax0.set_ylim(1e-30, 1e-22)

    c.plot_bestfit_betamodel_mass(ax1, style=gas)
    c.plot_inferred_nfw_mass(ax1, style=dm)
    c.plot_inferred_total_gravitating_mass(ax1, style=tot)
    c.plot_hydrostatic_mass_err(ax1, style=avg)
    # c.plot_hydrostatic_mass(ax1, style=tot)
    #ax1.loglog(radii, convert.g2msun*masstot_check, **tot)
    ax1.set_yscale("log")
    ax1.set_ylim(1e5, 1e16)

    c.plot_chandra_average(ax2, parm="kT", style=avg)
    c.plot_inferred_temperature(ax2, style=tot)
    ax2.set_ylim(-1, 10)

    c.plot_chandra_average(ax3, parm="P", style=avg)
    c.plot_inferred_pressure(ax3, style=tot)
    #ax3.loglog(radii, hydrostatic_pressure, **tot)
    ax3.set_yscale("log")
    ax3.set_ylim(1e-15, 5e-9)

    # Add Verlinde profiles
    if verlinde: c.plot_verlinde(ax1, ax2, ax3, style=tot)

    for ax, loc in zip([ax0, ax1, ax2, ax3], [3, 2, 3, 3]):
        ax.axvline(c.halo["r200"], c="k")
        # The y coordinates are axes while the x coordinates are data
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(c.halo["r200"]+150, 0.98, r"$r_{200}$", ha="left", va="top",
                fontsize=22, transform=trans)
        ax.axvline(c.halo["r500"], c="k")
        ax.text(c.halo["r500"]-150, 0.98, r"$r_{500}$", ha="right", va="top",
                fontsize=22, transform=trans)
        ax.set_xlabel("Radius [kpc]")
        ax.set_xscale("log")
        ax.set_xlim(0, 5000)
        if not add_sim: ax.legend(fontsize=18, loc=loc)
    ax0.set_ylabel("Density [g/cm$^3$]")
    ax1.set_ylabel("Mass [MSun]")
    ax2.set_ylabel("Temperature [keV]")
    ax3.set_ylabel("Pressure [erg/cm$^3$]")

    if add_sim:
        return fig.number
    else:
        fig.tight_layout()
        fig.savefig("out/{0}{1}_donnert2014figure1{2}_cNFW={3:.3f}_bf={4:.4f}{5}{6}.pdf"
            .format("fit/" if hasattr(c, "fit_counter") else "", c.name,
                    "_fit-{0:02d}".format(c.fit_counter) if hasattr(c, "fit_counter") else "",
                    c.halo["cNFW"], c.halo["bf200"], "_cut" if c.rcut_kpc is not None else "",
                    "_withVerlinde" if verlinde else ""), dpi=300)
        pyplot.close(fig)


def add_sim_to_donnert2014_figure1(fignum, halo, savedir, snapnr=None, binned=False):
    """ - lower stepsize to include all particles when binning
        - increase nbins to increase resolution of sampled particles
        - for publication-ready figure increase dpi to cranck up resolution
        - for runtime ~21 seconds for 5e7 particles """
    fig = pyplot.figure(fignum)
    ax0, ax1, ax2, ax3 = fig.get_axes()

    if hasattr(halo, "time"):
        fig.suptitle("T = {0:04.2f} Gyr".format(halo.time))

    # binned option gives rather ugly plots...
    # Donnert 2017 randomly selects 5k particles: looks sexy
    if binned:
        nbins = 128
        radii = numpy.power(10, numpy.linspace(numpy.log10(5), numpy.log10(5e3), nbins))

        # Expensive to sort, but less expensive than numpy.intersect1d(
        # numpy.where(r < halo.gas["r"]), numpy.where(halo.gas["r"] < r+dr))
        print "Sorting gas on radius"
        halo.gas.sort("r")  # most expensive step, ~11s for 5e7 particles (53% runtime)
        bin_edge = numpy.zeros(nbins+1, dtype=numpy.int)

        i, bin_number, stepsize = 0, 0, 1  # neglible runtime
        desired_radius = radii[bin_number]
        for r in halo.gas["r"][::stepsize]:
            if r > desired_radius:
                bin_edge[bin_number] = i
                bin_number += 1
                if bin_number == nbins:
                    break
                desired_radius = radii[bin_number]
            i += stepsize

        density = numpy.zeros(nbins)
        mass = numpy.zeros(nbins)
        temperature_min = numpy.zeros(nbins)
        temperature_max = numpy.zeros(nbins)
        temperature_mean = numpy.zeros(nbins)
        temperature_median = numpy.zeros(nbins)
        temperature_std = numpy.zeros(nbins)
        pressure = numpy.zeros(nbins)

        for i in xrange(nbins):
            density[i] = numpy.median(halo.gas["rho"][bin_edge[i]:bin_edge[i]+1])
            mass[i] = numpy.median(halo.gas["mass"][bin_edge[i]:bin_edge[i]+1])
            temperature_min[i] = numpy.min(halo.gas["kT"][bin_edge[i]:bin_edge[i]+1])
            temperature_max[i] = numpy.max(halo.gas["kT"][bin_edge[i]:bin_edge[i]+1])
            temperature_mean[i] = numpy.mean(halo.gas["kT"][bin_edge[i]:bin_edge[i]+1])
            temperature_median[i] = numpy.median(halo.gas["kT"][bin_edge[i]:bin_edge[i]+1])
            temperature_std[i] = numpy.std(halo.gas["kT"][bin_edge[i]:bin_edge[i]+1])
            pressure[i] = numpy.median(halo.gas["P"][bin_edge[i]:bin_edge[i]+1])

        gas = { "linestyle": "solid", "color": "green", "linewidth": "2" }
        dm = { "linestyle": "solid", "color": "green", "linewidth": "2" }

        # Do not plot noisy inner bins
        idx = numpy.where(halo.dm_radii > 5)
        dm_radii = halo.dm_radii[idx]
        dm_mass = halo.M_dm_below_r[idx]
        dm_density = halo.rho_dm_below_r[idx]

        ax0.plot(radii, density, **gas)
        ax0.plot(dm_radii, dm_density, **dm)

        ax1.plot(radii, mass, **gas)
        ax1.plot(dm_radii, dm_mass, **dm)

        ax2.plot(radii, temperature_min, **gas)
        ax2.plot(radii, temperature_max, **gas)
        # ax2.plot(radii, temperature_mean, **gas)
        ax2.plot(radii, temperature_median, **gas)

        ax3.plot(radii, pressure, **gas)

    else:
        gas = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 1,
                "markeredgecolor": "none",  "label": "simulation"}
        dm = { "c": "g", "lw": 2, "drawstyle": "steps-post", "label": ""}

        mask = numpy.random.randint(0, len(halo.gas["r"]), size=10000)
        mask2 = numpy.where(halo.gas["r"] < 100)
        mask = numpy.union1d(numpy.unique(mask), mask2[0])
        if halo.name == "cygNW":
            mask2 = numpy.where(halo.gas["r"] > 2e3)
            mask = numpy.setdiff1d(mask, mask2[0])


        ax0.plot(halo.gas["r"][mask], halo.gas["rho"][mask], rasterized=True, **gas)
        ax0.plot(halo.dm_radii, halo.rho_dm_below_r, rasterized=True, **dm)

        ax1.plot(halo.gas["r"][mask], halo.gas["mass"][mask], rasterized=True, **gas)
        ax1.plot(halo.dm_radii, halo.M_dm_below_r, rasterized=True, **dm)
        # TODO: sampled DM profile misses, rho and mass

        ax2.plot(halo.gas["r"][mask], halo.gas["kT"][mask], rasterized=True, **gas)

        ax3.plot(halo.gas["r"][mask], halo.gas["P"][mask], rasterized=True, **gas)

    inner = numpy.where(halo.gas["r"] < 50)
    hsml = 2*numpy.median(halo.gas["hsml"][inner])
    # hist, edges = numpy.histogram(halo.gas["hsml"], bins=1000)
    # hsml = edges[numpy.argmax(hist)]

    for ax, loc in zip(fig.axes, [3, 2, 3, 3]):
        # The y coordinates are axes while the x coordinates are data
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(numpy.arange(2000, 1e4, 0.01), 0, 1,
            facecolor="grey", edgecolor="grey", alpha=0.2,
            transform=trans)
        ax.axvline(x=hsml, c="g", ls=":")
        ax.text(hsml+6, 0.05, r"$2 h_{sml}$", ha="left", color="g",
            transform=trans, fontsize=22)
        handles, labels = ax.get_legend_handles_labels()
        handles[-2].set_linestyle("-")
        ax.legend(handles, labels, fontsize=18, loc=loc)

    # ~2s, 10% runtime
    # fig.tight_layout(rect=[0, 0.00, 1, 0.98])  # rect b/c suptitle/tight_layout bug
    fig.set_tight_layout(True)
    fig.savefig(savedir+"{0}_donnert2014figure1{1}.pdf"
        .format(halo.name, "_"+snapnr if snapnr else ""))
    # ~5.5s, 25% runtime
    fig.savefig(savedir+"{0}_donnert2014figure1{1}.png"
        .format(halo.name, "_"+snapnr if snapnr else ""), dpi=600)
    pyplot.close(fig)
    return halo


def twocluster_parms(cygA, cygNW, sim=None, verlinde=False):
    """ Create Donnert (2014) Figure 1 for the Cygnus observation + best-fit models
        @param c  : ObservedCluster
        @param sim: Simulation """

    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1, "elinewidth": 2 }
    gas = { "color": "k", "lw": 1, "linestyle": "dotted", "label": "gas" }
    dm  = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "dm" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "tot" }

    print "Running plot.twocluster_parms()"
    print "Creating figure"

    matplotlib.rc("font", **{"size": 18})
    fig, axes = pyplot.subplots(4, 2, figsize=(9, 16))
    # ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7))
    gs1 = matplotlib.gridspec.GridSpec(4, 2)
    gs1.update(hspace=0, wspace=0)
    ax6 = pyplot.subplot(gs1[6])
    ax4 = pyplot.subplot(gs1[4])
    ax2 = pyplot.subplot(gs1[2])
    ax0 = pyplot.subplot(gs1[0])

    ax7 = pyplot.subplot(gs1[7], sharey=ax6)
    ax5 = pyplot.subplot(gs1[5], sharey=ax4)
    ax3 = pyplot.subplot(gs1[3], sharey=ax2)
    ax1 = pyplot.subplot(gs1[1], sharey=ax0)

    print "Figure created"

    # Also do some plotting
    pyplot.sca(ax0)
    print "Plotting cygA density"
    cygA.plot_chandra_average(ax0, parm="rho", style=avg)
    cygA.plot_bestfit_betamodel(ax0, style=gas, rho=True)
    cygA.plot_inferred_nfw_profile(ax0, style=dm, rho=True)
    pyplot.sca(ax1)
    print "Plotting cygNW density"
    cygNW.plot_chandra_average(ax1, parm="rho", style=avg)
    cygNW.plot_bestfit_betamodel(ax1, style=gas, rho=True)
    cygNW.plot_inferred_nfw_profile(ax1, style=dm, rho=True)

    pyplot.sca(ax2)
    print "Plotting cygA mass"
    cygA.plot_bestfit_betamodel_mass(ax2, style=gas)
    cygA.plot_inferred_nfw_mass(ax2, style=dm)
    cygA.plot_inferred_total_gravitating_mass(ax2, style=tot)
    cygA.plot_hydrostatic_mass(ax2, style=tot)
    pyplot.sca(ax3)
    print "Plotting cygNW mass"
    cygNW.plot_bestfit_betamodel_mass(ax3, style=gas)
    cygNW.plot_inferred_nfw_mass(ax3, style=dm)
    cygNW.plot_inferred_total_gravitating_mass(ax3, style=tot)
    cygNW.plot_hydrostatic_mass(ax3, style=tot)

    pyplot.sca(ax4)
    print "Plotting cygA temperature"
    cygA.plot_chandra_average(ax4, parm="kT", style=avg)
    cygA.plot_inferred_temperature(ax4, style=tot)
    pyplot.sca(ax5)
    print "Plotting cygNW temperature"
    cygNW.plot_chandra_average(ax5, parm="kT", style=avg)
    cygNW.plot_inferred_temperature(ax5, style=tot)

    pyplot.sca(ax6)
    print "Plotting cygA pressure"
    cygA.plot_chandra_average(ax6, parm="P", style=avg)
    cygA.plot_inferred_pressure(ax6, style=tot)
    pyplot.sca(ax7)
    print "Plotting cygNW pressure"
    cygNW.plot_chandra_average(ax7, parm="P", style=avg)
    cygNW.plot_inferred_pressure(ax7, style=tot)

    # Add Verlinde profiles
    if verlinde: cygA.plot_verlinde(ax2, ax4, ax6, style=tot)
    if verlinde: cygNW.plot_verlinde(ax3, ax5, ax7, style=tot)

    print "Done plotting"

    print "Setting axes labels etc"
    for ax in fig.axes:
        ax.set_xscale("log")
        ax.set_xlim(0, 5000)
        if ax == ax4 or ax == ax5: continue
        ax.set_yscale("log")

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(labelbottom="off")
    for ax in [ax1, ax3, ax5, ax7]:
        ax.tick_params(labelleft="off")
    for ax in [ax3, ax7]:
        ax.tick_params(labelright="on")
    for ax in [ax2, ax6]:
        ax.tick_params(labelleft="off")

    ax0.set_ylabel("Density [g/cm$^3$]")
    ax0.set_ylim(1e-30, 5e-22)
    ax2.set_ylabel("Mass [MSun]")
    ax2.set_ylim(1e5, 1e16)
    ax4.set_ylabel("Temperature [keV]")
    ax4.set_ylim(-1, 10)
    ax6.set_ylabel("Pressure [erg/cm$^3$]")
    ax6.set_ylim(1e-15, 5e-9)

    for ax in [ax0, ax2, ax4, ax6]:
        ax.get_yaxis().set_label_coords(-0.2, 0.5)

    ax6.set_xlabel("Radius [kpc]")
    ax7.set_xlabel("Radius [kpc]")

    print "Done setting axes, now saving..."

    pyplot.tight_layout()
    pyplot.savefig("out/twocluster{0}{1}.png"
        .format("_cut" if cygA.rcut_kpc is not None else "",
                "_withVerlinde" if verlinde else ""), dpi=300)

    print "... done saving"
    pyplot.close()
    matplotlib.rc("font", **{"size": 28})


def dark_component(c):
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 1, "fillstyle": "full" }
    gas = { "color": "k", "lw": 1, "linestyle": "dotted", "label": "gas" }
    dm  = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "dm" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "tot" }


    fig, ax1 = pyplot.subplots(1, 1, figsize=(12, 9))

    c.plot_bestfit_betamodel_mass(ax1, style=gas)
    c.plot_inferred_nfw_mass(ax1, style=dm)
    c.plot_inferred_total_gravitating_mass(ax1, style=tot)
    c.plot_hydrostatic_mass(ax1, style=tot)
    #ax1.loglog(radii, convert.g2msun*masstot_check, **tot)

    c.avg.mask = [False for i in range(len(c.avg.columns))]
    r = c.avg["r"]
    rho = convert.density_cgs_to_msunkpc(c.avg["rho"])
    frho = convert.density_cgs_to_msunkpc(c.avg["frho"])

    spline = scipy.interpolate.UnivariateSpline(r, 4*numpy.pi*p2(r)*rho, k=5, s=10)
    fspline = scipy.interpolate.UnivariateSpline(r, 4*numpy.pi*p2(r)*frho, k=5, s=10)

    N = len(c.avg)
    mass, fmass = numpy.zeros(N), numpy.zeros(N)
    for i in range(N):
        mass[i] = spline.integral(0, r[i])
        fmass[i] = fspline.integral(0, r[i])
    pyplot.errorbar(r, mass, yerr=fmass, **avg)
    pyplot.fill_between(r, mass-fmass, mass+fmass, color="b", alpha=0.2)
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(4e-1, 5500)
    pyplot.ylim(1e5, 5e16)
    pyplot.xlabel("R [kpc]")
    pyplot.ylabel("M($<$R) [M$_\odot$]")
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Plots numerical haloes sampled with Toycluster
# ----------------------------------------------------------------------------
def toycluster_profiles(obs, sim, halo="000"):
    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas = { "marker": "o", "ls": "", "c": "g",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    dm = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas_a = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "" }
    dm_a = { "color": "k", "lw": 1, "linestyle": "solid", "label": "" }

    pyplot.figure(figsize=(12,9))
    ax = pyplot.gca()

    obs.plot_chandra_average(ax, parm="rho", style=avg)
    rho_gas = convert.toycluster_units_to_cgs(sim.toy.profiles[halo]["rho_gas"])
    rho_dm = convert.toycluster_units_to_cgs(sim.toy.profiles[halo]["rho_dm"])
    pyplot.plot(sim.toy.profiles[halo]["r"], rho_gas, **gas_a)
    pyplot.plot(sim.toy.profiles[halo]["r"], rho_dm, **dm_a)
    # obs.plot_bestfit_betamodel(style=fit)
    # obs.plot_inferred_nfw_profile(style=dm)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e-32, 9e-24,
        facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    pyplot.text(obs.halo["r200"]+100, 4e-24, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-32, ymax=9e-24)
    pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"toycluster_density_{0}.png".format(obs.name))
    pyplot.close()


def toyclustercheck(obs, sim, halo="000"):
    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 1,
            "markeredgecolor": "none",  "label": ""}
    dm = { "marker": "o", "ls": "", "c": "b", "ms": 2, "alpha": 1,
            "markeredgecolor": "none", "label": ""}
    dashed = { "color": "k", "lw": 1, "linestyle": "dashed" }
    dotted = { "color": "k", "lw": 1, "linestyle": "dotted" }
    solid = { "color": "k", "lw": 1, "linestyle": "solid" }

    radii = numpy.arange(1, 1e4, 1)

    pyplot.figure(figsize=(12,9))
    ax = pyplot.gca()
    pyplot.plot(sim.toy.gas["r"], sim.toy.gas["rho"], **gas)
    pyplot.plot(sim.toy.dm_radii, sim.toy.rho_dm_below_r, **dm)
    obs.plot_chandra_average(ax, parm="rho", style=avg)
    obs.plot_bestfit_betamodel(ax, style=dashed)  # cut (TODO: change obs.rcut_kpc?)
    # obs.plot_bestfit_betamodel(ax, style=dotted)  # uncut
    obs.plot_inferred_nfw_profile(ax, style=dotted)
    # rho_dm_cut = profiles.dm_density_nfw(radii, obs.halo["rho0_dm"],
    #     obs.halo["rs"], sim.toy.r_sample)
    # pyplot.plot(radii, rho_dm_cut, **solid)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e-32, 9e-24,
        facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    pyplot.text(obs.halo["r200"]+100, 4e-24, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-32, ymax=9e-24)
    pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"{0}_sampled_rho.png".format(obs.name), dpi=150)
    pyplot.close()


    pyplot.figure(figsize=(12,9))
    pyplot.plot(sim.toy.gas["r"], sim.toy.gas["mass"], **gas)
    pyplot.plot(sim.toy.dm_radii, sim.toy.M_dm_below_r, **dm)
    mdm = profiles.dm_mass_nfw(radii,
        convert.density_cgs_to_msunkpc(obs.halo["rho0_dm"]), obs.halo["rs"])
    # mdm_cut = [profiles.dm_mass_nfw_cut(r,
    #     convert.density_cgs_to_msunkpc(obs.halo["rho0_dm"]), obs.halo["rs"], sim.toy.r_sample)
    #     for r in radii]
    mgas = profiles.gas_mass_betamodel(radii,
        convert.density_cgs_to_msunkpc(obs.rho0), obs.beta, obs.rc)
    # mgas_cut = [profiles.gas_mass_betamodel_cut(r,
    #     convert.density_cgs_to_msunkpc(obs.rho0), obs.beta, obs.rc, obs.halo["r200"])
    #     for r in radii]

    pyplot.plot(radii, mdm, **dotted)
    # pyplot.plot(radii, mdm_cut, **solid)
    pyplot.plot(radii, mgas, **dotted)
    # pyplot.plot(radii, mgas_cut, **dashed)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e7, 1e15,
       facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    pyplot.text(obs.halo["r200"]+100, 3e14, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass ($<$r) [MSun]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e7, ymax=1e15)
    # pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"{0}_sampled_mass.png".format(obs.name), dpi=150)
    pyplot.close()


def toyclustercheck_T(obs, sim, halo="000"):
    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 1,
            "markeredgecolor": "none",  "label": "sampled kT"}

    pyplot.figure(figsize=(12,9))
    pyplot.errorbar(obs.avg["r"], obs.avg["kT"], xerr=obs.avg["fr"]/2,
                    yerr=[obs.avg["fkT"], obs.avg["fkT"]], **avg)
    pyplot.plot(sim.toy.gas["r"], sim.toy.gas["kT"], **gas)
    pyplot.plot(sim.toy.profiles[halo]["r"],
        convert.K_to_keV(convert.gadget_u_to_t(sim.toy.profiles[halo]["u_gas"])),
        label="u\_gas")
    pyplot.plot(sim.toy.profiles[halo]["r"],
        convert.K_to_keV(convert.gadget_u_to_t(sim.toy.profiles[halo]["u_ana"])),
        label="u\_ana")

    # pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e7, 1e15,
    #   facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    # pyplot.text(obs.halo["r200"]+100, 3e14, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Temperature [keV]")
    pyplot.yscale("log")
    pyplot.xscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-1, ymax=1e2)
    pyplot.legend(loc="upper left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"{0}_sampled_temperature.png".format(obs.name), dpi=150)
    pyplot.close()


# ----------------------------------------------------------------------------
# Plots of P-Gadget3 simulation snapshots
# ----------------------------------------------------------------------------
@concurrent(processes=threads)
def plot_singlecluster_stability(obs, sim, snapnr, path_to_snaphot):
    sim = copy.deepcopy(sim)
    print snapnr, id(obs), id(sim), path_to_snaphot
    sim.set_gadget_snap_single(snapnr, path_to_snaphot)
    halo = getattr(sim, "snap{0}".format(snapnr), None)
    if halo is not None:
        halo.name = obs.name
        fignum = donnert2014_figure1(obs, add_sim=True, verlinde=False)
        add_sim_to_donnert2014_figure1(fignum, halo, sim.outdir, snapnr="{0:03d}".format(snapnr))
    else:
        print "ERROR"

    del(sim)


@synchronized
def singlecluster_stability(sim, obs, verbose=True):
    if verbose: print "Running plot_singlecluster_stability"

    sim.set_gadget_paths(verbose=verbose)
    for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots):
        snapnr = int(path_to_snaphot[-3:])  # glob ==> unordered list
        plot_singlecluster_stability(obs, sim, snapnr, path_to_snaphot)


@concurrent(processes=threads)
def plot_twocluster_stability(cygA, cygNW, sim, snapnr, path_to_snaphot):
    sim = copy.deepcopy(sim)
    print snapnr, id(cygA), id(cygNW), id(sim), path_to_snaphot
    sim.set_gadget_snap_double(snapnr, path_to_snaphot)

    cygAsim = getattr(sim, "cygA{0}".format(snapnr), None)
    if cygAsim is not None:
        cygAsim.name = "cygA"
        fignum = donnert2014_figure1(cygA, add_sim=True, verlinde=False)
        add_sim_to_donnert2014_figure1(fignum, cygAsim, sim.outdir, snapnr="{0:03d}".format(snapnr))
    else:
       print "ERROR"

    cygNWsim = getattr(sim, "cygNW{0}".format(snapnr), None)
    if cygNWsim is not None:
        cygNWsim.name = "cygNW"
        fignum = donnert2014_figure1(cygNW, add_sim=True, verlinde=False)
        add_sim_to_donnert2014_figure1(fignum, cygNWsim, sim.outdir, snapnr="{0:03d}".format(snapnr))
    else:
       print "ERROR"

    del(sim)

@synchronized
def twocluster_stability(sim, cygA, cygNW, verbose=True):
    if verbose: print "Running plot_singlecluster_stability"

    sim.set_gadget_paths(verbose=verbose)
    for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots[0:1]):
        snapnr = int(path_to_snaphot[-3:])
        plot_twocluster_stability(cygA, cygNW, sim, snapnr, path_to_snaphot)


# ----------------------------------------------------------------------------
# Plots 2D projection cubes saved by P-Smac2
# ----------------------------------------------------------------------------
def psmac_xrays_with_dmrho_peakfind(sim, snapnr, xsum, ysum, xpeaks, ypeaks, distance, EA2=""):
    """ We find the peaks of the haloes by summing the dark matter density
        in the line-of-sight integrated P-Smac2 output. """

    fig = pyplot.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 3)
    axx = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    axy = fig.add_subplot(gs.new_subplotspec((1, 2), rowspan=2))
    axd = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2, rowspan=2),
                          sharex=axx, sharey=axy)
    axt = fig.add_subplot(gs.new_subplotspec((0, 2)))
    axt.axis("off")
    gs.update(wspace=0, hspace=0, top=0.94, left=0.15)
    axx.text(0.5, 1.01, "Summed (smoothed) Dark Matter Density X", ha="center",
             va="bottom", transform=axx.transAxes, fontsize=16)
    axx.plot(range(sim.xlen), xsum)
    axx.plot(xpeaks, xsum[(xpeaks+0.5).astype(int)], "ro", markersize=10)
    axx.set_ylim(0, 1.3)

    axy.text(1.01, 0.5, "Summed (smoothed) Dark Matter Density Y", ha="left",
             va="center", transform=axy.transAxes, fontsize=16, rotation=-90)
    axy.plot(ysum, range(sim.ylen))
    axy.plot(ysum[(ypeaks+0.5).astype(int)], ypeaks, "ro", markersize=10)
    axy.set_xlim(0, 1.3)

    axx.set_xlim(500, 1500)  # 800, 1400
    axy.set_ylim(sim.xlen/2-300,sim.xlen/2+300)
    axd.set_axis_bgcolor("k")
    axd.set_xlabel("x [pixel]")
    axd.set_ylabel("y [pixel]")
    for ax in [axx, axy, axt]:
        for tl in ax.get_xticklabels() + ax.get_yticklabels()\
                + ax.get_xticklines()  + ax.get_yticklines():
            tl.set_visible(False)
    pyplot.sca(axd)
    dt = getattr(sim, "dt", 0)  # assuming Toycluster/P-smac only, no Gadget-2
    axt.text(0.02, 1, "T = {0:2.2f} [Gyr]".format(snapnr*dt),
             ha="left", va="top")
    axt.text(0.02, 0.8, "R = {0:5.2f} [kpc]".format(distance),
             ha="left", va="top")

    if not numpy.isnan(distance):
        axd.plot(xpeaks[0], ypeaks[0], "ro", markersize=10)
        axd.plot(xpeaks[1], ypeaks[1], "ro", markersize=10)
    xrays = getattr(sim.psmac, "xray{0}".format(EA2), None)
    if xrays is None:
        print "error sim.psmac has no attr xray{0}".format(EA2)
        return
    axd.imshow(numpy.log10(xrays[snapnr].clip(min=2e-8, max=0.02)),
               origin="lower", cmap="spectral")
    axd.text(0.5, 0.95, "X-ray Surface Brightness", ha="center", va="top",
            color="white", transform=axd.transAxes)
    pyplot.savefig(sim.outdir+"xray_peakfind_{0:02d}_{1:03d}.png".format(int(EA2), snapnr))
    pyplot.close()


def simulated_quiescent_parm(c, sim, snapnr, parm="kT"):
    """ @param c     :  ObservedCluster instance
        @param sim   :  Simulation instance
        @param snapnr:  Number of snapshot to plot
        @param parm  :  Parameter to plot: ['kT', 'n', 'rho', 'P'] """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]" }
    parmmapping = { "kT": "tspec", }
                   #"n": "N/A",
                   # "rho": "rhogas"}
                   #"P": "N/A" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available in observation".format(parm)
        return
    if not parmmapping.get(parm, None):
        print "ERRROR: parm '{0}' is not available in simulation".format(parm)
        return

    r, val, fval = sim.create_quiescent_profile(snapnr, parm=parmmapping[parm])

    pyplot.figure(figsize=(12, 9))
    ax = pyplot.gca()
    c.plot_chandra_average(ax, parm=parm, style=avg)
    pyplot.errorbar(r, val, yerr=[fval, fval])
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="best")
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"{0}_{1}_{2:03d}.png".format(parm, c.name, snapnr))
    pyplot.close()


def twocluster_quiescent_parm(cygA, cygNW, sim, snapnr, parm="kT"):
    """ @param cygA  : ObservedCluster
        @param cygNW : ObservedCluster
        @param sim   : Simulation instance
        @param snapnr: Number of snapshot to plot
        @param parm  : Parameter to plot: ['kT', 'n', 'rho', 'P'] """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]" }
    parmmapping = { "kT": "tspec", }
                   #"n": "N/A",
                   # "rho": "rhogas"}
                   #"P": "N/A" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available in observation".format(parm)
        return
    if not parmmapping.get(parm, None):
        print "ERRROR: parm '{0}' is not available in simulation".format(parm)
        return

    r, results = sim.create_quiescent_profile(snapnr, parm=parmmapping[parm])
    for name in ["cygA", "cygNW"]:
        val, fval = results[name]

        avg["c"] = "g" if c.name == "cygA" else "b"
        pyplot.figure(figsize=(12, 9))
        ax = pyplot.gca()
        c.plot_chandra_average(ax, parm=parm, style=avg)
        pyplot.errorbar(r, val, yerr=[fval, fval])
        pyplot.xlabel("Radius [kpc]")
        pyplot.ylabel(parmnames[parm])
        pyplot.xscale("log")
        pyplot.yscale("log")
        pyplot.xlim(1, 2000)
        # pyplot.ylim(-0.02, 0.2)
        pyplot.legend(loc="best")
        pyplot.tight_layout()
        pyplot.savefig(sim.outdir+"{0}_{1}_{2:03d}.png".format(parm, c.name, snapnr))
        pyplot.close()


#@concurrent(processes=threads)
def plot_smac_temperature(i, snap, xlen, ylen, pixelscale, dt, outdir):
    print i

    pyplot.style.use(["dark_background"])

    fig, ax = pyplot.subplots(figsize=(16, 16))
    # https://stackoverflow.com/questions/32462881
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    im = ax.imshow(convert.K_to_keV(snap), origin="lower", cmap="afmhot",
                   vmin=2.5, vmax=15)
    fig.colorbar(im, cax=cax, orientation="vertical")

    fig.suptitle("T = {0:04.2f} Gyr".format(i*10*dt), color="white", size=26, y=0.9)

    ax.set_xlim(0, xlen)
    ax.set_ylim(0, ylen)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])

    scale = xlen*pixelscale
    scale = "[{0:.1f} Mpc]\^2".format(float(scale)/1000)
    pad = 16
    ax.text(2*pad, pad, scale, color="white",  size=18,
            horizontalalignment="left", verticalalignment="bottom")

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir+"tspec_{0:03d}.png".format(i), dpi=300)
    pyplot.close(fig)


#@synchronized
def make_temperature_video(sim):
    sim.set_gadget_paths()
    sim.read_smac()
    sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
    sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)

    for i, snap in enumerate(sim.psmac.tspec0):
        plot_smac_temperature(i, snap, sim.xlen, sim.ylen, sim.pixelscale, sim.dt, sim.outdir)


def gen_bestfit_2d_tspec():
    sim = Simulation(base="/media/SURFlisa", timestamp="20170115T0906", name="both",
                     set_data=False)
    sim.set_gadget_paths()
    sim.read_smac()
    sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
    sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)

    for i, EA2, xmin, ymin in zip([92, 91, 89, 84, 70, 16], [0, 15, 30, 45, 60, 75],
                                 [720, 723, 730, 741, 757, 778], [720, 724, 729, 746, 765, 789]):
        EA0, EA1, EA2 = 90, 51, EA2
        pyplot.style.use(["dark_background"])

        fig, ax = pyplot.subplots(figsize=(16, 16))
        # https://stackoverflow.com/questions/32462881
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0)
        im = ax.imshow(convert.K_to_keV(getattr(sim.psmac,
            "tspec{0}best".format(EA2))[0]), origin="lower", cmap="afmhot",
            vmin=2.5, vmax=15)
        fig.colorbar(im, cax=cax, orientation="vertical")

        fig.suptitle("T = {0:04.2f} Gyr".format(i*dt), color="white", size=26, y=0.9)

        ax.set_xlim(xmin, xmin+480)
        ax.set_ylim(ymin, ymin+480)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([])

        scale = (1200 - 720)*sim.pixelscale
        scale = "[{0:.1f} Mpc]$^2$".format(float(scale)/1000)
        pad = 3.75
        ax.text(xmin + 2*pad, ymin+pad, scale, color="white",  size=18,
                horizontalalignment="left", verticalalignment="bottom")

        angles = r"\begin{tabular}{p{1.25cm}ll}"
        angles += r" EA0 & = & {0:03d} \\".format(EA0)
        angles += " EA1 & = & {0:03d} \\\\".format(EA1)
        angles += " EA2 & = & {0:03d} \\\\".format(EA2)
        angles += (" \end{tabular}")

        ax.text(xmin+480-2*pad, ymin+pad, angles, color="white",  size=18,
                horizontalalignment="right", verticalalignment="bottom")

        ax.set_aspect("equal")
        fig.tight_layout()

        fig.savefig(sim.outdir+"tspec_best_{0:02d}_withcolorbar.png".format(EA2), dpi=300)


def Lx_chandra_vs_sim():
    # Open observation lss [counts/s/arcsec^2]
    obs = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    mosaic = fits.open(obs)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    mosaic_smooth = scipy.ndimage.filters.gaussian_filter(mosaic[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic[0].data.max()
    maxcounts_obs_index = mosaic[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    # Open Temperature
    lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
    lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"
    mosaic_kT = fits.open(lss_kT)


    ############################################
    # Convert Chandra countrate to cgs
    # PIMMS predicts a flux ( 0.500- 7.000keV) of 1.528E-11 ergs/cm/cm/s
    # pimms =  1.528E-11
    # data = mosaic[0].data  # counts / second
    # data *= pimms  # erg/cm/cm/s
    # goal  # [erg/cm^2/s/Hz]

    # cnts2erg_per_cm2 = 1.528e-11   # WebPIMMS
    # erg2erg_per_cm2 = 1./4/pi/ (1e3*cc.DL_Mpc*convert.kpc2cm)
    ############################################

    fig = pyplot.figure()

    # Display the smoothed lss Chandra observation
    # fig, ((ax0, ax1),(ax2, ax3)) = pyplot.subplots(2, 2, figsize=(12, 9))
    # gs1 = matplotlib.gridspec.GridSpec(2, 2)
    # gs1.update(wspace=0)
    # ax0 = pyplot.subplot(gs1[0])
    # ax1 = pyplot.subplot(gs1[1])
    # ax2 = pyplot.subplot(gs1[2])
    # ax3 = pyplot.subplot(gs1[3])

    # distance_factor = 4*numpy.pi*(cc.DL_Mpc*1000*convert.kpc2cm)**2
    # Lx = mosaic_smooth * convert.keV_to_erg(mosaic_kT[0].data) * distance_factor
    # im = ax0.imshow(Lx,
    #                 vmin=7.0e-10*convert.keV_to_erg(5.9)*distance_factor,
    #                 vmax=1.0e-6*convert.keV_to_erg(5.9)*distance_factor,
    #                 norm=matplotlib.colors.LogNorm(),
    #                 origin="lower", cmap="spectral")

    # # Indicate the CygA centroid
    # ax0.axhline(ycenter_obs, c="w")
    # ax0.axvline(xcenter_obs, c="w")

    # # Add a colorbar
    # im.axes.set_xticks([], [])
    # im.axes.set_yticks([], [])
    # divider = make_axes_locatable(ax0)
    # cax = divider.append_axes("bottom", size="5%", pad=0)
    # ticks = matplotlib.ticker.LogLocator(subs=range(10))
    # pyplot.colorbar(im, cax=cax, ticks=ticks, orientation="horizontal")

    mosaic_kT_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_kT[0].data, 15)
    Lx = mosaic_smooth * convert.keV_to_erg(mosaic_kT_smooth)
    ax = pyplot.gca()
    im = ax.imshow(Lx, vmin=7.0e-10*convert.keV_to_erg(5.9),
                   vmax=1.0e-6*convert.keV_to_erg(5.9),
                   norm=matplotlib.colors.LogNorm(),
                   origin="lower", cmap="spectral")

    # im = ax.imshow(mosaic_kT_smooth, origin="lower", cmap="afmhot",
    #    vmin=3.5, vmax=12)


    scale = "[{0:.1f} x {1:.1f} kpc]".format(xlen_obs_kpc, ylen_obs_kpc)
    pad = 32
    ax.text(xlen_obs_pix-pad, pad, scale, color="white",  size=16,
            horizontalalignment="right", verticalalignment="bottom")

    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.xaxis.label.set_color("white")
    ax.tick_params(axis="both", colors="white", length=4, width=0.5)

    im.axes.set_xticks(numpy.linspace(0, 2048, 17, dtype=int))
    im.axes.set_xticklabels([])
    im.axes.set_yticks(numpy.linspace(0, 2048, 17, dtype=int))
    im.axes.set_yticklabels([])

    ticks = matplotlib.ticker.LogLocator(subs=range(10))
    cax = pyplot.colorbar(im, ax=ax, shrink=0.45, pad=0.03, ticks=ticks,
        aspect=12, orientation="horizontal")
    cax.ax.tick_params(axis="x", length=1.5, width=0.5, which="major")
    cax.ax.tick_params(axis="x", length=4, width=0.5, which="minor")
    cax.ax.xaxis.set_ticks_position("both")

    # cax.ax.tick_params(axis="x", length=4, width=0.5, which="major")
    # cax.ax.tick_params(axis="x", length=1.5, width=4, which="minor")

    xlabel = r"Temperature $\left[ {\rm keV} \right]$"
    xlabel = r"X-ray Flux $\left[ \frac{\rm erg}{\rm cm^2 \, s} \right]$"
    cax.ax.set_xlabel(xlabel)

    pyplot.tight_layout()
    pyplot.savefig("out/lss_flux_15.pdf", dpi=600)
    # pyplot.savefig("out/lss_kT_15.pdf", dpi=600)


    return

    # Display temperature
    im = ax2.imshow(mosaic_kT_smooth, origin="lower", cmap="afmhot",
        vmin=2.5, vmax=15)
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="5%", pad=0)
    pyplot.colorbar(im, cax=cax, orientation="horizontal")


    # Open simulation Lx Smac Cube [erg/cm^2/s/Hz]
    import parse
    sim = "/Users/timohalbesma/Desktop/20170115T0905_xray_45_best.fits.fz"
    header, data = parse.psmac2_fitsfile(sim)

    # Find the centroid of "CygA" to align simulation and observation later on
    maxcounts_sim = data.max()
    maxcounts_sim_index = data.argmax()  # of flattened array
    ylen_sim_pix, xlen_sim_pix = data[0].shape
    ylen_sim_kpc = xlen_sim_kpc = float(header["XYSize"])
    xcenter_sim = maxcounts_sim_index % xlen_sim_pix
    ycenter_sim = maxcounts_sim_index / xlen_sim_pix
    pix2kpc_sim = float(header["XYSize"])/int(header["XYPix"])

    print "Smac Cube"
    print "  Shape ({0},   {1})   pixels".format(xlen_sim_pix, ylen_sim_pix)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_sim_kpc, ylen_sim_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_sim, ycenter_sim, maxcounts_sim)

    # Cut relevant part from the simulation
    desired_xlen_sim_kpc = xlen_obs_kpc
    desired_ylen_sim_kpc = ylen_obs_kpc
    desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
    desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
    xoffset = int((xcenter_sim * pix2kpc_sim - xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
    yoffset = int((ycenter_sim * pix2kpc_sim - ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

    equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                         xoffset: xoffset+desired_xlen_sim_pix]

    # Convolve with 2D Gaussian, radius converted to kpc in simulation
    # from a 9 pixel radius in the Chandra observation
    smooth_obs_kpc = 9 * pix2kpc_obs * arcsec2kpc
    smooth_sim_kpc = smooth_obs_kpc / pix2kpc_sim
    smaccube_smooth = scipy.ndimage.filters.gaussian_filter(equal_boxsize_kpc_smaccube, smooth_sim_kpc)

    # central value of CygNW observation / central value of simulated CygNW in 0th snapshot
    # magic = 5.82e-9 / 1.77e-5   # for 20170115T0905_xray_45_best.fits
    # magic = 5.82e-9 / 2.788e-5   # for 20170115T0905_xray_0_best.fits
    # smaccube_smooth *= magic

    # Display the cut-out, zoomed-in, correctly smoothed Smac Cube
    im = ax1.imshow(smaccube_smooth, vmin=7.0e-10, vmax=1.0e-6,
                    norm=matplotlib.colors.LogNorm(),
                    origin="lower", cmap="spectral",
                    extent=[0, xlen_obs_pix, 0, ylen_obs_pix])

    # Note that the extent morphs the simulation to observed range!
    ax1.axhline(ycenter_obs, c="w")
    ax1.axvline(xcenter_obs, c="w")

    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("bottom", size="5%", pad=0)

    ticks = matplotlib.ticker.LogLocator(subs=range(10))
    pyplot.colorbar(im, cax=cax, ticks=ticks, orientation="horizontal")

    # Create residuals plot
    zoomx = float(ylen_obs_pix) / smaccube_smooth.shape[0]
    zoomy = float(xlen_obs_pix) / smaccube_smooth.shape[1]
    shape_matched = scipy.ndimage.zoom(smaccube_smooth,  [zoomx, zoomy], order=3)

    im = ax3.imshow(100*(mosaic_smooth-shape_matched)/shape_matched,
                    vmin=-25, vmax=200, origin="lower", cmap="cubehelix",
                    extent=[0, xlen_obs_pix, 0, ylen_obs_pix])

    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("bottom", size="5%", pad=0)

    pyplot.colorbar(im, cax=cax, orientation="horizontal")

    pyplot.suptitle("X-ray Surface Brightness")
    ax0.set_title("Chandra Observation [1.03 MSec]")
    ax1.set_title("Simulation, inclination = 45 deg")
    ax2.set_title("")
    ax3.set_title("Residuals [percentage]")
    # pyplot.switch_backend("Agg")
    # pyplot.savefig("out/Lx_45_with_residuals.pdf", dpi=300)

    # print "Total luminosity = ",
    # print data[0].sum() * (float(header["XYSize"])*convert.kpc2cm)**2 / int(header["XYPix"])**2


def build_hank():
    # Open observation lss [counts/s/arcsec^2]
    obs = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    mosaic = fits.open(obs)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    mosaic_smooth = scipy.ndimage.filters.gaussian_filter(mosaic[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic[0].data.max()
    maxcounts_obs_index = mosaic[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    fig = pyplot.figure(figsize=(6.0625, 9.5625))
    axCLx = pyplot.subplot2grid((9,6), (0,0), colspan=3, rowspan=3)
    axCkT = pyplot.subplot2grid((9,6), (0,3), colspan=3, rowspan=3)
    axes = []
    for y in range(6):
        for x in range(6):
            ax = pyplot.subplot2grid((9,6), (3+y,x))
            ax.set_xticks([], []); ax.set_yticks([], [])
            axes.append(ax)

    # Display Lx
    im = axCLx.imshow(mosaic_smooth, vmin=7.0e-10, vmax=1.0e-6,
                    norm=matplotlib.colors.LogNorm(),
                    origin="lower", cmap="spectral")

    # Add a colorbar
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])
    # divider = make_axes_locatable(axCLx)
    # cax = divider.append_axes("right", size="5%", pad=0)
    # ticks = matplotlib.ticker.LogLocator(subs=range(10))
    # pyplot.colorbar(im, cax=cax, ticks=ticks, orientation="vertical")

    # Display temperature
    lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
    lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"
    mosaic = fits.open(lss_kT)

    im = axCkT.imshow(mosaic[0].data, origin="lower", cmap="afmhot",
        vmin=2.5, vmax=15)
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])
    # divider = make_axes_locatable(axCkT)
    # cax = divider.append_axes("right", size="5%", pad=0)
    # pyplot.colorbar(im, cax=cax, orientation="vertical")

    from simulation import Simulation
    sim50 = Simulation(base="/media/SURFlisa", timestamp="20170115T0905", name="both",
                     set_data=False)
    sim75 = Simulation(base="/media/SURFlisa", timestamp="20170115T0906", name="both",
                     set_data=False)
    sim25 = Simulation(base="/media/SURFlisa", timestamp="20170115T0907", name="both",
                     set_data=False)

    for Xe_i, sim in enumerate([sim25, sim50, sim75]):
        # sim.read_ics()
        # sim.set_gadget_paths()
        sim.read_smac(verbose=True)
        sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
        sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)

        for EA2_i, inclination in enumerate([0, 15, 30, 45, 60, 75]):
            data = getattr(sim.psmac, "xray{0}best".format(inclination))
            header = getattr(sim.psmac, "xray{0}best_header".format(inclination))

            maxcounts_sim = data.max()
            maxcounts_sim_index = data.argmax()  # of flattened array
            ylen_sim_pix, xlen_sim_pix = data[0].shape
            ylen_sim_kpc = xlen_sim_kpc = float(header["XYSize"])
            xcenter_sim = maxcounts_sim_index % xlen_sim_pix
            ycenter_sim = maxcounts_sim_index / xlen_sim_pix
            pix2kpc_sim = float(header["XYSize"])/int(header["XYPix"])

            print "Smac Cube, i =", inclination
            print "  Shape ({0},   {1})   pixels".format(xlen_sim_pix, ylen_sim_pix)
            print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_sim_kpc, ylen_sim_kpc)
            print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_sim, ycenter_sim, maxcounts_sim)

            # Cut relevant part from the simulation
            desired_xlen_sim_kpc = xlen_obs_kpc
            desired_ylen_sim_kpc = ylen_obs_kpc
            desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
            desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
            xoffset = int((xcenter_sim * pix2kpc_sim -
                xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
            yoffset = int((ycenter_sim * pix2kpc_sim -
                ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

            equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                                 xoffset: xoffset+desired_xlen_sim_pix]

            # Convolve with 2D Gaussian, radius converted to kpc in simulation
            # from a 9 pixel radius in the Chandra observation
            smooth_obs_kpc = 9 * pix2kpc_obs * arcsec2kpc
            smooth_sim_kpc = smooth_obs_kpc / pix2kpc_sim
            smaccube_smooth = scipy.ndimage.filters.gaussian_filter(
                equal_boxsize_kpc_smaccube, smooth_sim_kpc)

            # central value of CygNW observation / central value of simulated CygNW in 0th snapshot
            magic = 5.82e-9 / 1.77e-5   # for 20170115T0905_xray_45_best.fits
            # magic = 5.82e-9 / 2.788e-5   # for 20170115T0905_xray_0_best.fits
            smaccube_smooth *= magic

            # Display the cut-out, zoomed-in, correctly smoothed Smac Cube
            im = axes[6*EA2_i+Xe_i].imshow(smaccube_smooth, vmin=7.0e-10, vmax=1.0e-6,
                norm=matplotlib.colors.LogNorm(), origin="lower", cmap="spectral",
                extent=[0, xlen_obs_pix, 0, ylen_obs_pix])

            data = getattr(sim.psmac, "tspec{0}best".format(inclination))
            equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                                 xoffset: xoffset+desired_xlen_sim_pix]
            tspec = convert.K_to_keV(equal_boxsize_kpc_smaccube)
            im = axes[3+6*EA2_i+Xe_i].imshow(tspec, vmin=2.5, vmax=15,
                origin="lower", cmap="afmhot", extent=[0, xlen_obs_pix, 0, ylen_obs_pix])

    axCLx.text(0.5, 0.98, "\\textbf{\emph{Chandra} X-ray Surface Brightness}",
               fontsize=12, color="white", ha="center", va="top", transform=axCLx.transAxes)
    axCLx.text(0.02, 0.17, "ACIS Mosaic\n0.5-7.0 keV\n1.02 Msec total exposure", color="white",
               fontsize=12, ha="left", va="top", transform=axCLx.transAxes)
    axCkT.text(0.5, 0.98, "\\textbf{Temperature [keV]}",
               fontsize=12, color="white", ha="center", va="top", transform=axCkT.transAxes)

    for xE, ax in zip([0.25, 0.50, 0.75, 0.25, 0.50, 0.75], axes[0:6]):
        ax.text(0.02, 0.95, "$X_E=\,${0:.2f}".format(xE), color="white", fontsize=12,
                ha="left", va="top", transform=ax.transAxes)

    for EA2, ax in zip([0, 15, 30, 45, 60, 75], axes[::6]):
        ax.text(0.02, 0.02, "$i=\,${0:02d}".format(EA2),
                color="white", fontsize=12, ha="left", va="bottom", transform=ax.transAxes)

        # axes[n].text(0.5, 0.5, str(n), transform=axes[n].transAxes)
    pyplot.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.01)
    pyplot.savefig("out/matrix.png", pdi=6000)


def build_hank2():
    # Open observation lss [counts/s/arcsec^2]
    lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
    lss_Lx = lss+".dir/Frame1/cygnus_lss_fill_flux.fits"
    lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"

    mosaic_Lx = fits.open(lss_Lx)
    mosaic_kT = fits.open(lss_kT)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    Lx_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_Lx[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic_Lx[0].data.max()
    maxcounts_obs_index = mosaic_Lx[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic_Lx[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic_Lx[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic_Lx[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic_Lx[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    fig = pyplot.figure(figsize=(12, 15))
    axes = []
    for y in range(6):
        for x in range(6):
            ax = pyplot.subplot2grid((7, 6), (y, x))
            ax.set_xticks([], []); ax.set_yticks([], [])
            axes.append(ax)

    cax_left = fig.add_axes([0.05, 0.11, 0.4, 0.03])
    cax_left.set_xticks([], []); cax_left.set_yticks([], [])
    cax_right = fig.add_axes([0.55, 0.11, 0.4, 0.03])
    cax_right.set_xticks([], []); cax_right.set_yticks([], [])

    # pyplot.show()
    # return

    from simulation import Simulation
    sim50 = Simulation(base="/media/SURFlisa", timestamp="20170115T0905", name="both",
                     set_data=False)
    sim75 = Simulation(base="/media/SURFlisa", timestamp="20170115T0906", name="both",
                     set_data=False)
    sim25 = Simulation(base="/media/SURFlisa", timestamp="20170115T0907", name="both",
                     set_data=False)

    for Xe_i, sim in enumerate([sim25, sim50, sim75]):
        # sim.read_ics()
        # sim.set_gadget_paths()
        sim.read_smac(verbose=True)
        sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
        sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)

        for EA2_i, inclination in enumerate([0, 15, 30, 45, 60, 75]):
            data = getattr(sim.psmac, "xray{0}best".format(inclination))
            header = getattr(sim.psmac, "xray{0}best_header".format(inclination))

            maxcounts_sim = data.max()
            maxcounts_sim_index = data.argmax()  # of flattened array
            ylen_sim_pix, xlen_sim_pix = data[0].shape
            ylen_sim_kpc = xlen_sim_kpc = float(header["XYSize"])
            xcenter_sim = maxcounts_sim_index % xlen_sim_pix
            ycenter_sim = maxcounts_sim_index / xlen_sim_pix
            pix2kpc_sim = float(header["XYSize"])/int(header["XYPix"])

            print "Smac Cube, i =", inclination
            print "  Shape ({0},   {1})   pixels".format(xlen_sim_pix, ylen_sim_pix)
            print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_sim_kpc, ylen_sim_kpc)
            print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_sim, ycenter_sim, maxcounts_sim)

            # Cut relevant part from the simulation
            desired_xlen_sim_kpc = xlen_obs_kpc
            desired_ylen_sim_kpc = ylen_obs_kpc
            desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
            desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
            xoffset = int((xcenter_sim * pix2kpc_sim -
                xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
            yoffset = int((ycenter_sim * pix2kpc_sim -
                ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

            equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                                 xoffset: xoffset+desired_xlen_sim_pix]

            # Convolve with 2D Gaussian, radius converted to kpc in simulation
            # from a 9 pixel radius in the Chandra observation
            smooth_obs_kpc = 9 * pix2kpc_obs * arcsec2kpc
            smooth_sim_kpc = smooth_obs_kpc / pix2kpc_sim
            smaccube_smooth = scipy.ndimage.filters.gaussian_filter(
                equal_boxsize_kpc_smaccube, smooth_sim_kpc)

            # central value of CygNW observation / central value of simulated CygNW in 0th snapshot
            magic = 5.82e-9 / 1.77e-5   # for 20170115T0905_xray_45_best.fits
            # magic = 5.82e-9 / 2.788e-5   # for 20170115T0905_xray_0_best.fits
            smaccube_smooth *= magic

            # Display the cut-out, zoomed-in, correctly smoothed Smac Cube
            Lx = axes[6*EA2_i+Xe_i].imshow(smaccube_smooth, vmin=5.0e-10, vmax=1.0e-7,
                norm=matplotlib.colors.LogNorm(), origin="lower", cmap=colorcet.cm["linear_bmw_5_95_c86"],
                extent=[0, xlen_obs_pix, 0, ylen_obs_pix])

            data = getattr(sim.psmac, "tspec{0}best".format(inclination))
            equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                                 xoffset: xoffset+desired_xlen_sim_pix]
            tspec = convert.K_to_keV(equal_boxsize_kpc_smaccube)
            kT = axes[3+6*EA2_i+Xe_i].imshow(tspec, vmin=3.5, vmax=12,
                origin="lower", cmap=colorcet.cm["linear_kryw_5_100_c67"], extent=[0, xlen_obs_pix, 0, ylen_obs_pix])

            if Xe_i is 0 and EA2_i is 0:
                # Colorbar for X-ray Surface Brightness
                cax = pyplot.colorbar(Lx, cax=cax_left, orientation="horizontal")
                cax.ax.xaxis.set_ticks_position("both")
                cax.ax.tick_params(axis="both", which="major", length=6, width=1, labelsize=16, direction="in")
                cax.ax.set_xlabel(r"X-ray Surface Brightness $\left[\frac{\mathrm{counts}}{\mathrm{cm}^2 \, \mathrm{s}} \right]$", fontsize=18)
                cax.set_ticks([1e-9, 1e-8, 1e-7])
                cax.set_ticklabels(["$10^{-9}$", "$10^{-8}$", "$10^{-7}$"])
                minorticks = Lx.norm(numpy.hstack([numpy.arange(5, 10, 1)/1e10,
                    numpy.arange(2, 10, 1)/1e9, numpy.arange(2, 10, 1)/1e8]))
                cax.ax.tick_params(which="minor", length=3, width=1, direction="in")
                cax.ax.xaxis.set_ticks(minorticks, minor=True)

                # Colorbar for Spectroscopic Temperature
                cax = pyplot.colorbar(kT, cax=cax_right, orientation="horizontal")
                cax_right.xaxis.set_ticks_position("both")
                cax_right.tick_params(axis="both", length=6, width=1, labelsize=16, direction="in")
                cax_right.set_xlabel(r"Temperature [keV]", fontsize=18)
                cax_right.tick_params(which="minor", length=3, width=1, direction="in")

    for xE, ax in zip([0.25, 0.50, 0.75, 0.25, 0.50, 0.75], axes[0:6]):
        ax.text(0.02, 0.95, "$X_E={0:.2f}$".format(xE), color="white", fontsize=22,
                ha="left", va="top", transform=ax.transAxes)

    for EA2, ax in zip([0, 15, 30, 45, 60, 75], axes[::6]):
        ax.text(0.02, 0.02, "$i={0:02d}$".format(EA2),
                color="white", fontsize=22, ha="left", va="bottom", transform=ax.transAxes)

        # axes[n].text(0.5, 0.5, str(n), transform=axes[n].transAxes)
    pyplot.subplots_adjust(left=0., bottom=0.02, right=1., top=1., wspace=0., hspace=0.01)
    pyplot.savefig("out/matrix.png", pdi=6000)
    pyplot.savefig("out/matrix.pdf", pdi=6000)
    pyplot.close(fig)


def cygA_total_Lx():
    # Open observation lss [counts/s/arcsec^2]
    obs = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    mosaic = fits.open(obs)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    mosaic_smooth = scipy.ndimage.filters.gaussian_filter(mosaic[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic[0].data.max()
    maxcounts_obs_index = mosaic[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)
    print

    # Open Temperature
    lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
    lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"
    mosaic_kT = fits.open(lss_kT)
    mosaic_kT_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_kT[0].data, 15)

    # Display the smoothed lss Chandra observation
    pyplot.figure()

    distance_factor = 4*numpy.pi*(cc.DL_Mpc*1000*convert.kpc2cm)**2
    Lx = mosaic_smooth * convert.keV_to_erg(mosaic_kT_smooth) * distance_factor
    # TODO: is pix2arcsec_obs**2 really needed?
    # Lx *= p2(pix2arcsec_obs)

    im = pyplot.imshow(Lx, origin="lower", cmap="spectral",
                       norm=matplotlib.colors.LogNorm(),
                       vmin=7.0e-10*convert.keV_to_erg(5.9)*distance_factor,
                       vmax=1.0e-6*convert.keV_to_erg(5.9)*distance_factor)

    y,x = numpy.ogrid[-ycenter_obs:ylen_obs_pix-ycenter_obs,
                      -xcenter_obs:xlen_obs_pix-xcenter_obs]

    mask = (p2(x) + p2(y) <= p2(700))
    mask_no_nucleus = (p2(x) + p2(y) <= p2(700)) & (p2(x) + p2(y) >= p2(150))
    mask_inner = (p2(x) + p2(y) <= p2(150)) & (p2(x) + p2(y) >= p2(150-6))
    mask_outer = (p2(x) + p2(y) <= p2(700)) & (p2(x) + p2(y) >= p2(700-6))
    y, x = numpy.where(mask_inner)
    pyplot.scatter(x, y, s=1, c="r", edgecolor="face", alpha=1)
    y, x = numpy.where(mask_outer)
    pyplot.scatter(x, y, s=1, c="r", edgecolor="face", alpha=1)

    print "Lx CygA  = ", sum(Lx[mask])
    print "Lx CygA  = ", sum(Lx[mask_no_nucleus]), "w/o nucleus"
    print "Lx CygA  = ", sum(Lx[mask])*p2(pix2arcsec_obs), "?"
    print "Lx CygA  = ", sum(Lx[mask_no_nucleus])*p2(pix2arcsec_obs), "?", "w/o nucleus"

    xcenter_NW, ycenter_NW = 1547, 1657
    y,x = numpy.ogrid[-ycenter_NW:ylen_obs_pix-ycenter_NW,
                      -xcenter_NW:xlen_obs_pix-xcenter_NW]
    mask = (p2(x) + p2(y) <= p2(700))
    mask_outer = (p2(x) + p2(y) <= p2(700)) & (p2(x) + p2(y) >= p2(700-6))
    print "Lx CygNW = ", sum(Lx[mask])
    print "Lx CygNW = ", sum(Lx[mask])*p2(pix2arcsec_obs), "?"

    y, x = numpy.where(mask_outer)
    pyplot.scatter(x, y, s=1, c="r", edgecolor="face", alpha=1)

    pyplot.xlim(0, xlen_obs_pix)
    pyplot.ylim(0, ylen_obs_pix)
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])
    divider = make_axes_locatable(pyplot.gca())
    cax = divider.append_axes("bottom", size="5%", pad=0)
    pyplot.colorbar(im, cax=cax, orientation="horizontal")

    pyplot.savefig("out/total_Lx_regions.pdf", pdi=600)


def build_1d_matrix():
    from simulation import Simulation
    sim50 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0905", name="both",
                     set_data=False)
    sim75 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0906", name="both",
                     set_data=False)
    sim25 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0907", name="both",
                     set_data=False)

    avg = { "marker": "o", "ls": "", "c": "b", "ms": 1, "alpha": 0.3,
            "elinewidth": 1, "label": "avg"}
    merger = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 0.3,
            "elinewidth": 1, "label": "merger" }

    import main
    a = main.new_argument_parser().parse_args()
    a.do_cut = False; a.clustername = "both"
    cygA, cygNW = main.set_observed_clusters(a)


    # pyplot.switch_backend("Agg")
    fig = pyplot.figure(figsize=(22,12))
    axes = list()
    for y in range(6):
        for x in range(6):
            ax = pyplot.subplot2grid((6,6), (y,x))
            axes.append(ax)

    from panda import create_panda
    for Xe_i, sim in enumerate([sim25, sim50, sim75]):
        # sim.read_ics()
        # sim.set_gadget_paths()
        sim.read_smac(verbose=True)
        sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
        sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)

        rmax = 900/sim.pixelscale
        radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(rmax), 42))
        dr = radii[1:] - radii[:-1]
        radii = radii[:-1]
        N = len(radii)

        for EA2_i, inclination in enumerate([0, 15, 30, 45, 60, 75]):
            quiescent_temperature = numpy.zeros(N)
            quiescent_temperature_std = numpy.zeros(N)
            merger_temperature = numpy.zeros(N)
            merger_temperature_std = numpy.zeros(N)

            data = getattr(sim.psmac, "tspec{0}best765".format(inclination))[0]
            header = getattr(sim.psmac, "tspec{0}best765_header".format(inclination))

            # snapnr = int(header["Input_File"].strip("'")[-3:])
            # snapnr=0 because tspec{0}best is not cube
            cA, cNW, distance = sim.find_cluster_centroids_psmac_dmrho(
                snapnr=0, EA2=inclination)

            offset = 0
            for (xc, yc), name in zip([cA, cNW], ["cygA", "cygNW"]):
                pyplot.figure(figsize=(12, 12))
                for i, r in enumerate(radii):
                    print_progressbar(i, N)
                    angle1 = 96 if name == "cygA" else 276
                    angle2 = 6 if name == "cygA" else 186
                    quiescent_mask = create_panda(sim.xlen, sim.ylen, xc, yc,
                                                  r, angle1, angle2)
                    quiescent_temperature[i] = convert.K_to_keV(
                        numpy.median(data[quiescent_mask]))
                    quiescent_temperature_std[i] = convert.K_to_keV(
                        numpy.std(data[quiescent_mask]))

                    angle1 = 6 if name == "cygA" else 186
                    angle2 = 96 if name == "cygA" else 276
                    merger_mask = create_panda(sim.xlen, sim.ylen, xc, yc,
                                                  r, angle1, angle2)
                    merger_temperature[i] = convert.K_to_keV(
                        numpy.median(data[merger_mask]))
                    merger_temperature_std[i] = convert.K_to_keV(
                        numpy.std(data[merger_mask]))

                    y, x = numpy.where(quiescent_mask)
                    pyplot.scatter(x, y, s=1, c="r", edgecolor="face", alpha=1)
                    y, x = numpy.where(merger_mask)
                    pyplot.scatter(x, y, s=1, c="k", edgecolor="face", alpha=1)
                im = pyplot.imshow(convert.K_to_keV(data), origin="lower",
                                   cmap="afmhot",  vmin=2.5, vmax=15)
                pyplot.gca().set_aspect("equal")
                pyplot.xlim(cA[0]-100, cA[0]+300)
                pyplot.ylim(cA[1]-100, cA[1]+300)
                pyplot.xticks([], [])
                pyplot.yticks([], [])
                divider = make_axes_locatable(pyplot.gca())
                cax = divider.append_axes("bottom", size="5%", pad=0)
                pyplot.colorbar(im, cax=cax, orientation="horizontal")
                pyplot.tight_layout()

                pyplot.savefig("out/{0}_XE={1:.2f}_i={2:02d}.png".format(name, (Xe_i+1)*0.25, inclination))
                pyplot.close()

                axes[offset+6*EA2_i+Xe_i].errorbar(radii*sim.pixelscale,
                    quiescent_temperature, [quiescent_temperature_std, quiescent_temperature_std],
                    c="b")
                axes[offset+6*EA2_i+Xe_i].errorbar(radii*sim.pixelscale,
                    merger_temperature, [merger_temperature_std, merger_temperature_std],
                    c="g")

                if name == "cygA":
                    cygA.plot_chandra_average(axes[offset+6*EA2_i+Xe_i],
                        parm="kT", style=avg)
                    cygA.plot_chandra_sector(axes[offset+6*EA2_i+Xe_i],
                        parm="kT", merger=True, style=merger)
                elif name == "cygNW":
                    cygNW.plot_chandra_average(axes[offset+6*EA2_i+Xe_i],
                        parm="kT", style=avg)
                pyplot.xscale("log")
                pyplot.yscale("log")
                pyplot.xlim(1, 2000)

                offset = 3
            # break
        # break

    i=0
    for xE, ax in zip([0.25, 0.50, 0.75, 0.25, 0.50, 0.75], axes[0:6]):
        i+=1
        ax.text(0.02, 0.95, "$X_E=\,${0:.2f}".format(xE)+("\nCygA" if i<4 else "\ncygNW"),
                color="black", fontsize=12, ha="left", va="top", transform=ax.transAxes)

    for EA2, ax in zip([0, 15, 30, 45, 60, 75], axes[::6]):
        ax.text(0.02, 0.02, "$i=\,${0:02d}".format(EA2),
                color="black", fontsize=12, ha="left", va="bottom", transform=ax.transAxes)

    for ax in axes:
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_ylim(2.5, 15)
        ax.set_xlim(1, 900)
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    pyplot.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)
    pyplot.savefig("out/1d_matrix_lin.png", pdi=6000)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(2.5, 15)
        ax.set_xlim(1, 900)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    pyplot.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)
    pyplot.savefig("out/1d_matrix_log.png", pdi=6000)

    for Xe_i, sim in enumerate([sim25, sim50, sim75]):
        sim.read_smac(verbose=True)
        sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
        sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)
        sim.dt = 0.01

        for EA2_i, inclination in enumerate([0, 15, 30, 45, 60, 75]):
            cA, cNW, distance = sim.find_cluster_centroids_psmac_dmrho(
                snapnr=0, EA2=inclination)
            header = getattr(sim.psmac, "tspec{0}best765_header".format(inclination))
            snapnr = int(header["Input_File"].strip("'")[-3:])

            offset = 0
            for (xc, yc), name in zip([cA, cNW], ["cygA", "cygNW"]):
                ax = axes[offset+6*EA2_i+Xe_i]
                ax.cla()
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                info = r"\begin{tabular}{lll}"
                info += r" ID & = & {0} \\".format(sim.timestamp)
                info += " $X_E$ & = & {0:.2f} \\\\".format((Xe_i+1)*0.25)
                info += " snapnr & = & {0:03d} \\\\".format(snapnr)
                info += " time & = & {0:04.2f} Gyr \\\\".format(snapnr*sim.dt)
                info += " distance & = & {0:03.2f} kpc \\\\".format(distance)
                info += (" \end{tabular}")

                ax.text(0.5, 0.5, info,
                        transform=ax.transAxes, va="center", ha="center",
                        fontsize=12)
                offset = 3
                # ax.texts[-1].remove()
    i=0
    for xE, ax in zip([0.25, 0.50, 0.75, 0.25, 0.50, 0.75], axes[0:6]):
        i+=1
        ax.text(0.02, 0.95, "$X_E=\,${0:.2f}".format(xE)+("\nCygA" if i<4 else "\ncygNW"),
                color="black", fontsize=12, ha="left", va="top", transform=ax.transAxes)

    for EA2, ax in zip([0, 15, 30, 45, 60, 75], axes[::6]):
        ax.text(0.02, 0.02, "$i=\,${0:02d}".format(EA2),
                color="black", fontsize=12, ha="left", va="bottom", transform=ax.transAxes)

    pyplot.savefig("out/1d_matrix_info.png", pdi=6000)



def hankieeeee():
    # Space for ylabel
    pyplot.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.0, right=0.1, bottom=0, top=1, wspace=0)
    axy = pyplot.subplot(gs[0])
    axy.set_xticks([], []); axy.set_yticks([], [])
    for k in axy.spines.keys():
        axy.spines[k].set_visible(False)
    axy.text(0.5, 0.55, "Temperature [keV]",
             ha="center", va="center", rotation=90)


    # Space for xlabel
    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.1, right=1, bottom=0, top=0.1, wspace=0)
    axx = pyplot.subplot(gs[0])
    axx.set_xticks([], []); axx.set_yticks([], [])
    for k in axx.spines.keys():
        axx.spines[k].set_visible(False)
    axx.text(0.5, 0.5, "Radius [kpc]", ha="center", va="center")

    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.1, right=1, bottom=0.1001, top=1, wspace=0)
    ax = pyplot.subplot(gs[0])
    ax.set_xticks([], []); ax.set_yticks([], [])

    for i in range(3):
        for j in range(6):
            gs1 = gridspec.GridSpec(1,2)
            gs1.update(left=0.11+(1./3)*i*0.9, right=0.105+(1./3)*(i+1)*0.89,
                       bottom=0.11+(1./6)*j*0.9, top=0.105+(1./6)*(j+1)*0.89,
                       wspace=0.0)

            ax = pyplot.subplot(gs1[-1, :-1])
            ax.set_xticks([], []); ax.set_yticks([], [])
            ax = pyplot.subplot(gs1[-1, -1])
            ax.set_xticks([], []); ax.set_yticks([], [])



if __name__ == "__main__":
    pyplot.rcParams.update( { "text.usetex": True, "font.size": 18 } )
    # matplotlib.rcParams['text.latex.preamble']=[r"\boldmath"]
    build_1d_matrix()
    pyplot.rcParams.update( { "text.usetex": True, "font.size": 28 } )

