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

# import warnings
# warnings.simplefilter('error', UserWarning)


def show_observations(arguments):
    cygA = ObservedCluster("cygA", verbose=arguments.verbose)
    cygNW = ObservedCluster("cygNW", verbose=arguments.verbose)

    for parm in ["n", "rho", "kT", "P"]:  # CygA and CygNW
        plot.quiescent_parm(cygA, parm)
        plot.quiescent_parm(cygNW, parm)
    for parm in ["n", "rho", "kT", "P"]:  # CygA only
        plot.sector_parm(cygA, parm)

    plot.chandra_coolingtime(cygA)
    plot.chandra_coolingtime(cygNW)

    plot.bestfit_betamodel(cygA)
    plot.bestfit_betamodel(cygNW)

    plot.inferred_nfw_profile(cygA)
    plot.inferred_nfw_profile(cygNW)

    plot.inferred_temperature(cygA)
    plot.inferred_temperature(cygNW)


def infer_hydrostatic_mass(arguments):
    """ Smith+ (2002; eq. 3) Hydrostatic mass from observed temperature and
        number density"""
    cygA = ObservedCluster("cygA", verbose=arguments.verbose, debug=arguments.debug)
    cygNW = ObservedCluster("cygNW", verbose=arguments.verbose, debug=arguments.debug)

    plot.smith_hydrostatic_mass(cygA, debug=arguments.debug)
    plot.smith_hydrostatic_mass(cygNW, debug=arguments.debug)


def write_ics(arguments):
    cygA = ObservedCluster("cygA", verbose=arguments.verbose)
    cygNW = ObservedCluster("cygNW", verbose=arguments.verbose)
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


def check_toycluster_rho_and_temperature(arguments, bla=False):
    if not arguments.clustername:
        print "Please specify clustername. Returning."
        return

    if bla:  # TODO: proper name #NoInspiration
        if arguments.clustername == "cygA" and bla:
            cNFW = 12.181
            bf = 0.0740
        if arguments.clustername == "cygNW" and bla:
            cNFW=5.13
            bf=0.055
    else:
        cNFW = None  # will use Duffy+ 2008
        bf = 0.17    # Planelles+ 2013

    obs = ObservedCluster(arguments.clustername, cNFW=cNFW, bf=bf, verbose=arguments.verbose)
    sim = Simulation(arguments.basedir, arguments.timestamp, arguments.clustername)
    plot.toycluster_profiles(obs, sim)
    plot.toyclustercheck(obs, sim)
    plot.toyclustercheck_T(obs, sim)


def plot_smac_snapshots(arguments):
    if not arguments.clustername:
        print "Please specify clustername. Returning."
        return
    obs = ObservedCluster(arguments.clustername, verbose=arguments.verbose)
    sim = Simulation(arguments.basedir, arguments.timestamp, arguments.clustername)
    for i in range(sim.nsnaps):
        # sim.find_cluster_centroids_psmac_dmrho(i)
        plot.simulated_quiescent_parm(obs, sim, i, parm="kT")
    os.chdir(sim.outdir)
    # os.system('ffmpeg -y -r 8 -i "xray_peakfind_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "xray-dmdensity.mp4"')
    os.system('ffmpeg -y -r 8 -i "kT_cygA_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "kT_cygA.mp4"')
    os.system('ffmpeg -y -r 8 -i "tspec_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "tspec.mp4"')


def test_cnfw(arguments):
    for cNFW in range(1, 25, 2):
        cygA = ObservedCluster("cygA", cNFW=cNFW, verbose=arguments.verbose)
        plot.inferred_nfw_profile(cygA)
        plot.inferred_temperature(cygA)


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Simulation Pipeline Parser")
    args.add_argument("-t", "--timestamp", dest="timestamp",
        help="String of the Simulation ID", default="20161124T0148")
    args.add_argument("-b", "--basedir", dest="basedir",
        help="Path to the base directory", default="/usr/local/mscproj")
    args.add_argument("-c", "--clustername", dest="clustername",
        help="Name of the subcluster", default=None, choices=["cygA", "cygNW"])
    args.add_argument("-v", "--verbose", dest="verbose", action="store_true",
        help="Toggle verbose. Verbose is True by default", default=True)
    args.add_argument("-d", "--debug", dest="debug", action="store_true",
        help="Toggle debug. Debug is False by default", default=False)
    # group = args.add_mutually_exclusive_group(required=True)
    # group.add_argument("-t", "--timestamp", dest="timestamp", nargs=1,
    #    help="string of the Simulation ID")

    return args


if __name__ == "__main__":
    arguments = new_argument_parser().parse_args()
    # show_observations(arguments)
    infer_hydrostatic_mass(arguments)
    # write_ics(arguments)
    # plot_smac_snapshots(arguments)

    # For fun, fit the concentration parameter and baryon fraction
    # fit.total_gravitating_mass_freecbf(ObservedCluster("cygNW"), verbose=True)
    # cygA = ObservedCluster("cygA", cNFW=12.181, bf=0.0740, **arguments)
    # cygNW = ObservedCluster("cygNW", cNFW=5.13, bf=0.055, **arguments)

    # cygA = ObservedCluster("cygA", **arguments)
    # cygNW = ObservedCluster("cygNW", **arguments
    # plot.inferred_nfw_profile(cygA)
    # plot.inferred_mass(cygA)
    # plot.inferred_temperature(cygA)
    # plot.inferred_pressure(cygA)

    # check_toycluster_rho_and_temperature(arguments, bla=False)

    # pyplot.show()
