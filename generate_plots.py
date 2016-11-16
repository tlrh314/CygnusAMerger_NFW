import numpy
from matplotlib import pyplot

import profiles
import convert


# ----------------------------------------------------------------------------
# Plots for Chandra observations
# ----------------------------------------------------------------------------
def plot_chandra_temperature(c):
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


def plot_chandra_parm(c, parm="rho"):
    """ @param c:  ObservedCluster """
    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(alpha=1, parm=parm)
    # cygA.plot_chandra_sector(alpha=1, parm=parm)
    r = numpy.arange(0, 2e3, 1)
    # first order using best-fit of previous dataset
    pyplot.plot(r, profiles.gas_density_betamodel(r, convert.ne_to_rho(0.09162), 0.538, 26.02, 1347))
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm^3]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="upper left")
# ----------------------------------------------------------------------------
