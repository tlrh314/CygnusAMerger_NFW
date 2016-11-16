import numpy
import astropy.units as u
import astropy.constants as const
import matplotlib
from matplotlib import pyplot

import profiles
import convert


# ----------------------------------------------------------------------------
# Plots for Chandra observations
# ----------------------------------------------------------------------------
def chandra_temperature(c):
    """ @param c:  ObservedCluster """
    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(alpha=1)
    c.plot_chandra_sector(alpha=1)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("kT [keV]")
    pyplot.xscale("log")  # Linear better shows merger-affected radii
    pyplot.xlim(3, 2000)
    pyplot.ylim(2, 12)
    pyplot.legend(loc="upper left")


def chandra_parm(c, parm="rho"):
    """ @param c:  ObservedCluster """
    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(alpha=1, parm=parm)
    # cygA.plot_chandra_sector(alpha=1, parm=parm)
    r = numpy.arange(0, 2e3, 1)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="upper left")


def chandra_coolingtime(c):
    """ @param c:  ObservedCluster """
    Tcool = profiles.sarazin_coolingtime(c.avg["n"]/u.cm**3, c.avg["kT"]*u.keV/const.k_B.to(u.keV/u.K))

    pyplot.figure(figsize=(12, 9))
    pyplot.plot(c.avg["r"], Tcool.value)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Cooling Time [yr]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.tight_layout()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Plots analytical models based on observation
# ----------------------------------------------------------------------------
def bestfit_betamodel(c):
    """ Plot best-fit betamodel with residuals """
    fig, (ax, ax_r) = pyplot.subplots(2, 2, sharex=True, figsize=(16, 12))
    gs1 = matplotlib.gridspec.GridSpec(3, 3)
    gs1.update(hspace=0)
    ax = pyplot.subplot(gs1[:-1,:])
    ax_r = pyplot.subplot(gs1[-1,:], sharex=ax)  # residuals

    # Plot Chandra observation and betamodel with mles
    pyplot.sca(ax)
    c.plot_chandra_average(alpha=1, parm="n")
    c.plot_bestfit_betamodel()
    pyplot.ylabel("Density [1/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.ylim(numpy.min(c.avg["n"])/1.5, numpy.max(c.avg["n"])*1.3)
    pyplot.legend(loc="lower left", prop={"size": 30})

    # Plot residuals
    pyplot.sca(ax_r)
    c.plot_bestfit_residuals()
    pyplot.axhline(y=0, lw=3, ls="dashed", c="k")
    pyplot.ylabel("Residuals [\%]")
    pyplot.xlabel("Radius [kpc]")
    pyplot.xscale("log")
    pyplot.xlim(2 if c.name == "cygA" else 20, 1e3 if c.name == "cygA" else 1.1e3)
    pyplot.ylim(-35, 35)

    # Fix for overlapping y-axis markers
    from matplotlib.ticker import MaxNLocator
    ax.tick_params(labelbottom="off")
    nbins = len(ax_r.get_yticklabels())
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="upper"))

    # Force axis labels to align
    ax.get_yaxis().set_label_coords(-0.07, 0.5)
    ax_r.get_yaxis().set_label_coords(-0.07, 0.5)
    pyplot.tight_layout()
    pyplot.savefig("out/bestfit_betamodel_{0}.pdf".format(c.name), dpi=150)
# ----------------------------------------------------------------------------
