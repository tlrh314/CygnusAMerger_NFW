# -*- coding: utf-8 -*-

import copy

import numpy
import astropy.units as u
import astropy.constants as const
import matplotlib
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from line_profiler_support import profile
from deco import concurrent, synchronized
threads=16

import fit
import profiles
import convert


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
                  "P": "Pressure [erg/cm$^3$]" }
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
    pyplot.savefig("out/{0}_quiescent_{1}.pdf".format(c.name, parm), dpi=150)


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
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Plots analytical models based on observation
# ----------------------------------------------------------------------------
def bestfit_betamodel(c):
    """ Plot best-fit betamodel with residuals """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
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
    pyplot.ylabel("Density [1/cm$^3$]")
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
    pyplot.savefig("out/{0}_bestfit_betamodel.pdf".format(c.name), dpi=150)
    pyplot.sca(ax)


def inferred_nfw_profile(c):
    """ Plot the observed gas density, best-fit betamodel and the inferred
        best-fit NFW profile for the cluster
        @param c   : ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
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

    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1, "elinewidth": 2 }
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
    c.plot_hydrostatic_mass(ax1, style=tot)
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

    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_xlabel("Radius [kpc]")
        ax.set_xscale("log")
        ax.set_xlim(0, 5000)
        ax.legend(fontsize=12)
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


def add_sim_to_donnert2014_figure1(fignum, halo, savedir, snapnr=None, binned=True):
        fig = pyplot.figure(fignum)
        ax0, ax1, ax2, ax3 = fig.get_axes()

        if hasattr(halo, "time"):
            fig.suptitle("T = {0:04.2f} Gyr".format(halo.time))

        if binned:
            nbins = 100
            radii = numpy.power(10, numpy.linspace(numpy.log10(10), numpy.log10(5e3), nbins+1))
            print radii[-1]
            dr = radii[1:] - radii[:-1]
            radii = radii[:-1]

            density = numpy.zeros(nbins)
            mass = numpy.zeros(nbins)
            temperature_min = numpy.zeros(nbins)
            temperature_max = numpy.zeros(nbins)
            temperature_mean = numpy.zeros(nbins)
            temperature_median = numpy.zeros(nbins)
            temperature_std = numpy.zeros(nbins)
            pressure = numpy.zeros(nbins)
            halo.gas.sort("r")
            for i, (r, dr) in enumerate(zip(radii, dr)):
                upper = numpy.where(halo.gas["r"] > r)
                lower = numpy.where(halo.gas["r"] < r+dr)
                in_bin = numpy.intersect1d(upper, lower)
                if in_bin.size:
                    density[i] = numpy.median(halo.gas["rho"][in_bin])
                    mass[i] = numpy.median(halo.gas["mass"][in_bin])
                    temperature_min[i] = numpy.min(halo.gas["kT"][in_bin])
                    temperature_max[i] = numpy.max(halo.gas["kT"][in_bin])
                    temperature_mean[i] = numpy.mean(halo.gas["kT"][in_bin])
                    temperature_median[i] = numpy.median(halo.gas["kT"][in_bin])
                    temperature_std[i] = numpy.std(halo.gas["kT"][in_bin])
                    pressure[i] = numpy.median(halo.gas["P"][in_bin])

            gas = { "linestyle": "solid", "color": "green" }
            dm = { "linestyle": "solid", "color": "green" }

            ax0.plot(radii, density, **gas)
            ax0.plot(halo.dm_radii, halo.rho_dm_below_r, **dm)

            ax1.plot(radii, mass, **gas)
            ax1.plot(halo.dm_radii, halo.M_dm_below_r, **dm)

            ax2.plot(radii, temperature_min, **gas)
            ax2.plot(radii, temperature_max, **gas)
            # ax2.plot(radii, temperature_mean, **gas)
            ax2.plot(radii, temperature_median, **gas)

            ax3.plot(radii, pressure, **gas)


        else:
            gas = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 1,
                    "markeredgecolor": "none",  "label": ""}
            dm = { "marker": "o", "ls": "", "c": "g", "ms": 2, "alpha": 1,
                    "markeredgecolor": "none", "label": ""}
            ax0.plot(halo.gas["r"], halo.gas["rho"], **gas)
            ax0.plot(halo.dm_radii, halo.rho_dm_below_r, **dm)

            ax1.plot(halo.gas["r"], halo.gas["mass"], **gas)
            ax1.plot(halo.dm_radii, halo.M_dm_below_r, **dm)
            # TODO: sampled DM profile misses, rho and mass

            ax2.plot(halo.gas["r"], halo.gas["kT"], **gas)

            ax3.plot(halo.gas["r"], halo.gas["P"], **gas)

        inner = numpy.where(halo.gas["r"] < 100)
        hsml = 2*numpy.median(halo.gas["hsml"][inner])
        for ax in fig.axes:
            # The y coordinates are axes while the x coordinates are data
            trans = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.fill_between(numpy.arange(2000, 1e4, 0.01), 0, 1,
                facecolor="grey", edgecolor="grey", alpha=0.2,
                transform=trans)
            ax.axvline(x=hsml, c="g", ls=":")
            ax.text(hsml, 0.05, r"$2 h_{sml}$", ha="left", color="g",
                transform=trans, fontsize=22)

        fig.tight_layout(rect=[0, 0.00, 1, 0.98])  # rect b/c suptitle/tight_layout bug
        fig.savefig(savedir+"{0}_donnert2014figure1{1}.png"
            .format(halo.name, "_"+snapnr if snapnr else ""), dpi=300)
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
    for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots):
        snapnr = int(path_to_snaphot[-3:])
        plot_twocluster_stability(cygA, cygNW, sim, snapnr, path_to_snaphot)


# ----------------------------------------------------------------------------
# Plots 2D projection cubes saved by P-Smac2
# ----------------------------------------------------------------------------
def psmac_xrays_with_dmrho_peakfind(sim, snapnr, xsum, ysum, xpeaks, ypeaks, distance):
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
        axd.plot(xpeaks[1], ypeaks[0], "ro", markersize=10)
    axd.imshow(numpy.log10(sim.psmac.xray[snapnr].clip(min=2e-8, max=0.02)),
               origin="lower", cmap="spectral")
    axd.text(0.5, 0.95, "X-ray Surface Brightness", ha="center", va="top",
            color="white", transform=axd.transAxes)
    pyplot.savefig(sim.outdir+"xray_peakfind_{0:03d}.png".format(snapnr))
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
