# -*- coding: utf-8 -*-

import os
import copy
import argparse
import numpy
from matplotlib import pyplot

from cluster import ObservedCluster
import fit
import plot
from parse import write_toycluster_parameterfile
from simulation import Simulation
from line_profiler_support import profile

from plotsettings import PlotSettings
style = PlotSettings()

# import warnings
# warnings.simplefilter('error', UserWarning)

threads=16
from conc import concurrent
from conc import synchronized


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

    plot.inferred_mass(cygA)
    plot.inferred_mass(cygNW)

    plot.smith_hydrostatic_mass(cygA, debug=True)
    plot.smith_hydrostatic_mass(cygNW, debug=True)

    plot.inferred_temperature(cygA)
    plot.inferred_temperature(cygNW)

    plot.inferred_pressure(cygA)
    plot.inferred_pressure(cygNW)

    plot.donnert2014_figure1(cygA, verlinde=False)
    plot.donnert2014_figure1(cygNW, verlinde=False)

    plot.donnert2014_figure1(cygA, verlinde=True)
    plot.donnert2014_figure1(cygNW, verlinde=True)


def write_ics(cygA, cygNW):
    cutA = cygA.rcut_kpc is not None
    cutNW = cygNW.rcut_kpc is not None
    if cutA is not cutNW: print "WARNING: use cut-off in both clusters, or none"
    ic_cygA= { "description": "(free) betamodel+NFW. Cygnus A single halo.",
               "Mtotal": cygA.halo["M200"]/1e10, "Mass_Ratio": 0,
               "beta_0": cygA.beta, "beta_1": 0,
               "bf_0": cygA.halo["bf200"], "bf_1": 0,
               "rcut_r200_ratio_0": cygA.RCUT_R200_RATIO, "rcut_r200_ratio_1": 0,
               "name_0": cygA.name, "name_1": "N/A",
               "c_nfw_0": cygA.halo["cNFW"], "rc_0": cygA.rc,
               "c_nfw_1": 0, "rc_1": 0, "filename":
               "ic_cygA_free{0}.par".format("_cut" if cutA else "") }
    ic_cygNW = { "description": "(free) betamodel+NFW. Cygnus NW single halo.",
                 "Mtotal": cygNW.halo["M200"]/1e10, "Mass_Ratio": 0,
                 "beta_0": cygNW.beta, "beta_1": 0,
                 "bf_0": cygNW.halo["bf200"], "bf_1": 0,
                 "rcut_r200_ratio_0": cygNW.RCUT_R200_RATIO, "rcut_r200_ratio_1": 0,
                 "name_0": cygNW.name, "name_1": "N/A",
                 "c_nfw_0": cygNW.halo["cNFW"], "rc_0": cygNW.rc,
                 "c_nfw_1": 0, "rc_1": 0, "filename":
                 "ic_cygNW_free{0}.par".format("_cut" if cutNW else "") }
    ic_both = { "description": "(free) betamodel+NFW. Cygnus A and Cygnus NW haloes.",
               "Mtotal": (cygA.halo["M200"]+cygNW.halo["M200"])/1e10,
               "Mass_Ratio": cygNW.halo["M200"]/cygA.halo["M200"],
               "beta_0": cygA.beta, "beta_1": cygNW.beta,
               "bf_0": cygA.halo["bf200"], "bf_1": cygNW.halo["bf200"],
               "rcut_r200_ratio_0": cygA.RCUT_R200_RATIO,
               "rcut_r200_ratio_1": cygNW.RCUT_R200_RATIO,
               "name_0": cygA.name, "name_1": cygNW.name,
               "c_nfw_0": cygA.halo["cNFW"], "rc_0": cygA.rc,
               "c_nfw_1": cygNW.halo["cNFW"], "rc_1": cygNW.rc,
               "filename": "ic_both_free{0}.par".format("_cut" if cutA else "")}
    write_toycluster_parameterfile(ic_cygA)
    write_toycluster_parameterfile(ic_cygNW)
    write_toycluster_parameterfile(ic_both)


def infer_toycluster_ics(a):
    # Fit the concentration parameter and baryon fraction
    mle, cis = fit.total_gravitating_mass_freecbf(
        ObservedCluster(a.basedir, "cygA", verbose=True), do_cut=a.do_cut)
    cygA = ObservedCluster(a.basedir, "cygA", cNFW=mle[0], bf=mle[1],
                           RCUT_R200_RATIO=mle[2] if a.do_cut else None, verbose=a.verbose)

    mle, cis = fit.total_gravitating_mass_freecbf(
        ObservedCluster(a.basedir, "cygNW", verbose=False), do_cut=a.do_cut)
    cygNW = ObservedCluster(a.basedir, "cygNW", cNFW=mle[0], bf=mle[1],
                            RCUT_R200_RATIO=mle[2] if a.do_cut else None, verbose=a.verbose)

    write_ics(cygA, cygNW)

    return cygA, cygNW


@profile
def set_observed_cluster(a):
    if a.clustername == "cygA":
        if a.do_cut:
            cNFW = 10.8084913766
            bf = 0.0448823494125
            RCUT_R200_RATIO = 0.749826451566
        else:
            cNFW = 12.1616022474
            bf = 0.075207094556
            RCUT_R200_RATIO = None

    if a.clustername == "cygNW":
        if a.do_cut:
            cNFW = 3.69286089273
            bf = 0.0357979867269
            RCUT_R200_RATIO = 0.994860631699
        else:
            cNFW = 4.84194426883
            bf = 0.0535343411893
            RCUT_R200_RATIO = None

    obs = ObservedCluster(a.basedir, a.clustername, cNFW=cNFW, bf=bf,
        RCUT_R200_RATIO=RCUT_R200_RATIO, verbose=a.verbose)

    return obs

@profile
def set_observed_clusters(a):
    a.clustername = "cygA"
    cygA = set_observed_cluster(a)
    a.clustername = "cygNW"
    cygNW = set_observed_cluster(a)
    a.clustername = "both"
    return cygA, cygNW


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


@concurrent(processes=threads)
def plot_singlecluster_stability(obs, sim, snapnr, path_to_snaphot):
    sim = copy.deepcopy(sim)
    print snapnr, id(obs), id(sim), path_to_snaphot
    sim.set_gadget_snap_single(snapnr, path_to_snaphot)
    halo = getattr(sim, "snap{0}".format(snapnr), None)
    if halo is not None:
        fignum = plot.donnert2014_figure1(obs, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, halo, sim.outdir, snapnr="{0:03d}".format(snapnr))
    else:
        print "ERROR"

    del(sim)


@synchronized
def singlecluster_stability(sim, obs, verbose=True):
    if verbose: print "Running plot_singlecluster_stability"

    sim.set_gadget_paths(verbose=a.verbose)
    for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots):
        plot_singlecluster_stability(obs, sim, snapnr, path_to_snaphot)


@concurrent(processes=threads)
def plot_twocluster_stability(cygA, cygNW, sim, snapnr, path_to_snaphot):
    sim = copy.deepcopy(sim)
    print snapnr, id(cygA), id(cygNW), id(sim), path_to_snaphot
    sim.set_gadget_snap_double(snapnr, path_to_snaphot)

    cygAsim = getattr(sim, "cygA{0}".format(snapnr), None)
    if cygAsim is not None:
        fignum = plot.donnert2014_figure1(cygA, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, cygAsim, sim.outdir, snapnr="{0:03d}".format(snapnr))
    else:
       print "ERROR"

    cygNWsim = getattr(sim, "cygNW{0}".format(snapnr), None)
    if cygNWsim is not None:
        fignum = plot.donnert2014_figure1(cygNW, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, cygNWsim, sim.outdir, snapnr="{0:03d}".format(snapnr))
    else:
       print "ERROR"

    del(sim)

@synchronized
def twocluster_stability(sim, cygA, cygNW, verbose=True):
    if verbose: print "Running plot_singlecluster_stability"

    sim.set_gadget_paths(verbose=a.verbose)
    for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots):
        plot_twocluster_stability(cygA, cygNW, sim, snapnr, path_to_snaphot)


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Simulation Pipeline Parser")
    args.add_argument("-t", "--timestamp", dest="timestamp",
        help="String of the Simulation ID", default="both_debug")
    args.add_argument("-b", "--basedir", dest="basedir",
        help="Path to the base directory", default="/usr/local/mscproj")
    args.add_argument("-c", "--clustername", dest="clustername",
        help="Name of the subcluster", default=None, choices=["cygA", "cygNW", "both"])
    args.add_argument("--cut", dest="do_cut", action="store_true",
        help="Show analytical profiles with cut-off", default=False)
    args.add_argument("-v", "--verbose", dest="verbose", action="store_true",
        help="Toggle verbose. Verbose is True by default", default=True)
    args.add_argument("-d", "--debug", dest="debug", action="store_true",
        help="Toggle debug. Debug is False by default", default=False)
    args.add_argument("-e", "--embed", dest="embed", action="store_true",
        help="Toggle iPython embedding. Embed is False by default", default=False)
    # group = args.add_mutually_exclusive_group(required=True)
    # group.add_argument("-t", "--timestamp", dest="timestamp", nargs=1,
    #    help="string of the Simulation ID")

    return args


if __name__ == "__main__":
    a = new_argument_parser().parse_args()
    if a.embed: header = ""

    sim = Simulation(base=a.basedir, name=a.clustername, timestamp=a.timestamp, set_data=False)
    if a.embed: header += "Simulation instance in `sim'\n"


    # cygA, cygNW = infer_toycluster_ics(a)
    if a.clustername == "both":
        cygA, cygNW = set_observed_clusters(a)
        twocluster_stability(sim, cygA, cygNW, verbose=a.verbose)
        if a.embed: header += "ObservedCluster instances in `cygA' and `cygNW'\n"

    if a.clustername == "cygA" or a.clustername == "cygNW":
        obs = set_observed_cluster(a)
        singlecluster_stability(sim, obs, verbose=a.verbose)
        if a.embed: header += "ObservedCluster instance in `obs'\n"


    # fig = plot.donnert2014_figure1(obs)
    # plot.addsim sim and snapnr

    # plot.donnert2014_figure1(obs, verlinde=False)

    # plot_smac_snapshots(a)

    if a.embed:
        import IPython
        IPython.embed(banner1="", header=header)
