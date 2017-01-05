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


def show_observations(cygA, cygNW):
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


def write_ics(cygA, cygNW):
    ic_cygA= { "description": "(free) betamodel+NFW. Cygnus A single halo.",
               "Mtotal": cygA.halo["M200"]/1e10, "Mass_Ratio": 0,
               "beta_0": cygA.beta, "beta_1": 0,
               "bf_0": cygA.halo["bf200"], "bf_1": 0,
               "name_0": cygA.name, "name_1": "N/A",
               "c_nfw_0": cygA.halo["cNFW"], "rc_0": cygA.rc,
               "c_nfw_1": 0, "rc_1": 0, "filename": "ic_cygA_free.par" }
    ic_cygNW = { "description": "(free) betamodel+NFW. Cygnus NW single halo.",
                 "Mtotal": cygNW.halo["M200"]/1e10, "Mass_Ratio": 0,
                 "beta_0": cygNW.beta, "beta_1": 0,
                 "bf_0": cygNW.halo["bf200"], "bf_1": 0,
                 "name_0": cygNW.name, "name_1": "N/A",
                 "c_nfw_0": cygNW.halo["cNFW"], "rc_0": cygNW.rc,
                 "c_nfw_1": 0, "rc_1": 0, "filename": "ic_cygNW_free.par" }
    ic_both = { "description": "(free) betamodel+NFW. Cygnus A and Cygnus NW haloes.",
               "Mtotal": (cygA.halo["M200"]+cygNW.halo["M200"])/1e10,
               "Mass_Ratio": cygNW.halo["M200"]/cygA.halo["M200"],
               "beta_0": cygA.beta, "beta_1": cygNW.beta,
               "bf_0": cygA.halo["bf200"], "bf_1": cygNW.halo["bf200"],
               "name_0": cygA.name, "name_1": cygNW.name,
               "c_nfw_0": cygA.halo["cNFW"], "rc_0": cygA.rc,
               "c_nfw_1": cygNW.halo["cNFW"], "rc_1": cygNW.rc,
               "filename": "ic_both_free.par"}
    write_toycluster_parameterfile(ic_cygA)
    write_toycluster_parameterfile(ic_cygNW)
    write_toycluster_parameterfile(ic_both)


def check_toycluster_rho_and_temperature(a, match_Tobs=True):
    if not a.clustername and (a.clustername != "cygA" or a.clustername != "cygNW"):
        print "Please specify clustername. Also for single cluster. Returning."
        return

    if match_Tobs:
        if a.clustername == "cygA" and match_Tobs:
            cNFW = 12.1616022474
            bf = 0.075207094556
            RCUT_R200_RATIO = None  # in Makefile_Toycluster
        if a.clustername == "cygNW" and match_Tobs:
            cNFW = 4.84194426883
            bf = 0.0535343411893
            RCUT_R200_RATIO = None
    else:
        cNFW = None  # will use Duffy+ 2008
        bf = 0.17    # Planelles+ 2013

    sim = Simulation(a.basedir, a.timestamp, a.clustername)
    if hasattr(sim.toy, "RCUT_R200_RATIO"):
        RCUT_R200_RATIO = None # sim.toy.RCUT_R200_RATIO

    obs = ObservedCluster(a.basedir, a.clustername, cNFW=cNFW, bf=bf,
        RCUT_R200_RATIO=RCUT_R200_RATIO, verbose=a.verbose)

    r500, M500 = fit.find_r500(obs)
    print "r500    =", r500
    print "M500tot =", M500

    # plot.donnert2014_figure1(obs, sim, verlinde=False)
    # plot.toycluster_profiles(obs, sim)
    # plot.toyclustercheck(obs, sim)
    # plot.toyclustercheck_T(obs, sim)

    sim.plot_singlecluster_stability(obs)


def check_twocluster_ics(a):
    cygA = ObservedCluster(a.basedir, "cygA", cNFW=12.40, bf=0.07653)
    cygNW = ObservedCluster(a.basedir, "cygNW", cNFW=5.17, bf=0.05498)
    sim = Simulation(a.basedir, a.timestamp, name="both")
    plot.twocluster_quiescent_parm(cygA, cygNW, sim, 0, parm="kT")


def plot_smac_snapshots(a):
    if a.clustername:
        print "Running for single cluster", a.clustername
        obs = ObservedCluster(a.basedir, a.clustername, verbose=a.verbose)
        sim = Simulation(a.basedir, a.timestamp, a.clustername)
        for i in range(sim.nsnaps):
            # sim.find_cluster_centroids_psmac_dmrho(i)
            plot.simulated_quiescent_parm(obs, sim, i, parm="kT")
        os.chdir(sim.outdir)
        # os.system('ffmpeg -y -r 8 -i "xray_peakfind_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "xray-dmdensity.mp4"')
        os.system('ffmpeg -y -r 8 -i "kT_cygA_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "kT_cygA.mp4"')
        os.system('ffmpeg -y -r 8 -i "tspec_%3d.png" -profile:v high444 -level 4.1 -c:v libx264 -preset slow -crf 25 -s "2000:2000" -an "tspec.mp4"')
    else:
        sim = Simulation(a.basedir, a.timestamp)


def test_cnfw(a):
    for cNFW in range(1, 25, 2):
        cygA = ObservedCluster(a.basedir, "cygA", cNFW=cNFW, verbose=a.verbose)
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
        help="Name of the subcluster", default=None, choices=["cygA", "cygNW", "both"])
    args.add_argument("-v", "--verbose", dest="verbose", action="store_true",
        help="Toggle verbose. Verbose is True by default", default=True)
    args.add_argument("-d", "--debug", dest="debug", action="store_true",
        help="Toggle debug. Debug is False by default", default=False)
    # group = args.add_mutually_exclusive_group(required=True)
    # group.add_argument("-t", "--timestamp", dest="timestamp", nargs=1,
    #    help="string of the Simulation ID")

    return args


if __name__ == "__main__":
    a = new_argument_parser().parse_args()
    # sim = Simulation(a.basedir, a.timestamp, a.clustername)

    check_toycluster_rho_and_temperature(a, match_Tobs=True)

    # Fit the concentration parameter and baryon fraction
    # mle, cis = fit.total_gravitating_mass_freecbf(
    #     ObservedCluster(a.basedir, "cygA", verbose=False))
    # cygA = ObservedCluster(a.basedir, "cygA", cNFW=mle[0], bf=mle[1],
    #                        RCUT_R200_RATIO=None, verbose=a.verbose)
    # cygA = ObservedCluster(a.basedir, "cygA",
    #     # cNFW=10.7444613578, bf=0.066148612882, RCUT_R200_RATIO=0.722808251743,
    #     cNFW=12.1616022474, bf=0.075207094556, RCUT_R200_RATIO=None,
    #     verbose=a.verbose, debug=a.debug)

    # mle, cis = fit.total_gravitating_mass_freecbf(
    #     ObservedCluster(a.basedir, "cygNW", verbose=False))
    # cygNW = ObservedCluster(a.basedir, "cygNW", cNFW=mle[0], bf=mle[1],
    #                         RCUT_R200_RATIO=None, verbose=a.verbose)
    # cygNW = ObservedCluster(a.basedir, "cygNW",
    #     # cNFW=3.36312049278, bf=0.0410345976722, RCUT_R200_RATIO=0.884433321242,
    #     cNFW=4.84194426883, bf=0.0535343411893, RCUT_R200_RATIO=None,
    #     verbose=a.verbose, debug=a.debug)

    # show_observations(cygA, cygNW)
    # plot.inferred_nfw_profile(cygA)
    # plot.inferred_mass(cygA)
    # plot.inferred_temperature(cygA)
    # plot.inferred_pressure(cygA)
    # plot.inferred_nfw_profile(cygNW)
    # plot.inferred_mass(cygNW)
    # plot.inferred_temperature(cygNW)
    # plot.inferred_pressure(cygNW)
    # plot.smith_hydrostatic_mass(cygA, debug=a.debug)
    # plot.smith_hydrostatic_mass(cygNW, debug=a.debug)
    # plot.donnert2014_figure1(cygA, verlinde=False)
    # plot.donnert2014_figure1(cygNW, verlinde=False)

    # write_ics(cygA, cygNW)

    # plot_smac_snapshots(a)  # TODO: have correct cNFW, bf

    # sim = check_twocluster_ics(a)
