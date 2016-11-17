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
    c.plot_chandra_average(parm=parm, style=avg)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="best")
    pyplot.tight_layout()
    pyplot.savefig("out/quiescent_{0}_{1}.pdf".format(parm, c.name), dpi=150)


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

    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(parm=parm, style=avg)
    c.plot_chandra_sector(parm=parm, merger=True, style=merger)
    c.plot_chandra_sector(parm=parm, hot=True, style=hot)
    c.plot_chandra_sector(parm=parm, cold=True, style=cold)
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
    pyplot.savefig("out/sector_{0}_{1}.pdf".format(parm, c.name), dpi=150)


def chandra_coolingtime(c):
    """ @param c:  ObservedCluster """
    Tcool = profiles.sarazin_coolingtime(c.avg["n"]/u.cm**3,
            c.avg["kT"]*u.keV/const.k_B.to(u.keV/u.K))

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

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }

    fig, (ax, ax_r) = pyplot.subplots(2, 2, sharex=True, figsize=(16, 12))
    gs1 = matplotlib.gridspec.GridSpec(3, 3)
    gs1.update(hspace=0)
    ax = pyplot.subplot(gs1[:-1,:])
    ax_r = pyplot.subplot(gs1[-1,:], sharex=ax)  # residuals

    # Plot Chandra observation and betamodel with mles
    pyplot.sca(ax)

    c.plot_chandra_average(parm="n", style=avg)
    c.plot_bestfit_betamodel()
    pyplot.ylabel("Density [1/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.ylim(numpy.min(c.avg["n"])/1.5, numpy.max(c.avg["n"])*1.3)
    pyplot.legend(loc="lower left", fontsize=22)

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
