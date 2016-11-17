import numpy
from matplotlib import pyplot

from cluster import ObservedCluster
import fit
import plot

from plotsettings import PlotSettings
style = PlotSettings()


if __name__ == "__main__":
    cygA = ObservedCluster("cygA")
    cygNW = ObservedCluster("cygNW")

    # plot.quiescent_parm(cygA, "n")  # n, rho, kT, P; also for cygNW
    # plot.sector_parm(cygA, parm="kT")  # n, rho, kT, P; only for cygA
    # plot.chandra_coolingtime(cygA)
    # plot.bestfit_betamodel(cygA)
    # plot.bestfit_betamodel(cygNW)
    plot.inferred_nfw_profile(cygA)
    plot.inferred_nfw_profile(cygNW)
    # pyplot.show()
