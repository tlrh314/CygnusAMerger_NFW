import os
import argparse
import numpy
from matplotlib import pyplot

from cluster import ObservedCluster
import fit
import plot
from parse import write_toycluster_parameterfile
from simulation import Simulation

from plotsettings import PlotSettings
style = PlotSettings()


def new_argument_parser():
    parser = argparse.ArgumentParser(
        description="Simulation Pipeline")
    parser.add_argument("-t", "--timestamp", dest="timestamp",
            help="string of the Simulation ID", default="20161124T0148")
    parser.add_argument("-b", "--basedir", dest="basedir",
            help="path to the base directory", default="/usr/local/mscproj")
    parser.add_argument("-c", "--clustername", dest="clustername",
            help="name of the subcluster", default=None, choices=["cygA", "cygNW"])
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("-t", "--timestamp", dest="timestamp", nargs=1,
    #    help="string of the Simulation ID")

    return parser


if __name__ == "__main__":
    arguments = new_argument_parser().parse_args()

    # cygA = ObservedCluster("cygA", verbose=True)
    # cygNW = ObservedCluster("cygNW", verbose=False)
    # plot.quiescent_parm(cygA, "n")  # n, rho, kT, P; also for cygNW
    # plot.sector_parm(cygA, parm="kT")  # n, rho, kT, P; only for cygA
    # plot.chandra_coolingtime(cygA)
    # plot.bestfit_betamodel(cygA)
    # plot.bestfit_betamodel(cygNW)
    # plot.inferred_nfw_profile(cygA)
    # plot.inferred_nfw_profile(cygNW)

    generate_toycluster_parameterfiles = False
    if generate_toycluster_parameterfiles:
        ic_cygA= { "description": "(free) betamodel+NFW. Cygnus A single halo.",
                   "Mtotal": cygA.halo["M200"]/1e10, "Mass_Ratio": 0,
                   "beta_0": cygA.beta, "beta_1": 0, "name_0": cygA.name, "name_1": "N/A",
                   "c_nfw_0": cygA.halo["cNFW"], "rc_0": cygA.rc,
                   "c_nfw_1": 0, "rc_1": 0, "filename": "ic_cygA_free.par" }
        ic_cygNW = { "description": "(free) betamodel+NFW. Cygnus NW single halo.",
                     "Mtotal": cygNW.halo["M200"]/1e10, "Mass_Ratio": 0,
                     "beta_0": cygNW.beta, "beta_1": 0, "name_0": cygNW.name, "name_1": "N/A",
                     "c_nfw_0": cygNW.halo["cNFW"], "rc_0": cygNW.rc,
                     "c_nfw_1": 0, "rc_1": 0, "filename": "ic_cygNW_free.par" }
        ic_both = { "description": "(free) betamodel+NFW. Cygnus A and Cygnus NW haloes.",
                   "Mtotal": (cygA.halo["M200"]+cygNW.halo["M200"])/1e10,
                   "Mass_Ratio": cygNW.halo["M200"]/cygA.halo["M200"],
                   "beta_0": cygA.beta, "beta_1": cygNW.beta,
                   "name_0": cygA.name, "name_1": cygNW.name,
                   "c_nfw_0": cygA.halo["cNFW"], "rc_0": cygA.rc,
                   "c_nfw_1": cygNW.halo["cNFW"], "rc_1": cygNW.rc,
                   "filename": "ic_both_free.par"}
        write_toycluster_parameterfile(ic_cygA)
        write_toycluster_parameterfile(ic_cygNW)
        write_toycluster_parameterfile(ic_both)

    obs = ObservedCluster(arguments.clustername, verbose=True)
    sim = Simulation(arguments.basedir, arguments.timestamp, arguments.clustername)
    for i in range(sim.nsnaps):
        # sim.find_cluster_centroids_psmac_dmrho(i)
        plot.simulated_quiescent_parm(obs, sim, i, parm="kT")
    os.chdir(sim.outdir)
    # os.system('ffmpeg -y -r 8 -i "xray_peakfind_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "xray-dmdensity.mp4"')
    os.system('ffmpeg -y -r 8 -i "kT_cygA_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "kT_cygA.mp4"')
    os.system('ffmpeg -y -r 8 -i "tspec_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "tspec.mp4"')


    # plot.toyclustercheck(cygA, toyA)
    # plot.toyclustercheck_T(cygA, toyA)
    # for k,v in toyA.header.iteritems(): print "{0:<17}: {1}".format(k, v)
    # toyA.gas, toyA.dm
    # plot.toycluster_profiles(cygA, toyA)
    # toyNW = Toycluster("cygNW")

    pyplot.show()
