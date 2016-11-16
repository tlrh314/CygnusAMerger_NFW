import numpy
from matplotlib import pyplot

import profiles
from cluster import ObservedCluster
from generate_plots import plot_chandra_temperature
from generate_plots import plot_chandra_parm


if __name__ == "__main__":
    cygA = ObservedCluster("cygA")
    cygNW = ObservedCluster("cygNW")

    plot_chandra_parm(cygA, "rho")
    pyplot.show()
