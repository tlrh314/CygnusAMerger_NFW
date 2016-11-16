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

    # plot.chandra_temperature(cygA)
    # plot.chandra_parm(cygA, "n")
    # plot.chandra_coolingtime(cygA)
    plot.bestfit_betamodel(cygA)
    plot.bestfit_betamodel(cygNW)
    # pyplot.show()
