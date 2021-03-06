# -*- coding: utf-8 -*-

import os
import copy
import argparse
import numpy
import scipy
import peakutils
import matplotlib
from matplotlib import pyplot
from line_profiler_support import profile

import fit
import plot
import parse
from cluster import ObservedCluster
from simulation import Simulation

from plotsettings import PlotSettings
style = PlotSettings()

# import warnings
# warnings.simplefilter('error', UserWarning)

from deco import concurrent, synchronized
threads=2


def show_observations(cygA, cygNW):
    for parm in ["n", "rho", "kT", "P"]:  # CygA and CygNW
        plot.quiescent_parm(cygA, parm)
        plot.quiescent_parm(cygNW, parm)
    for parm in ["n", "rho", "kT", "P"]:  # CygA only
        if cygA.data == "2Msec": continue
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

    # plot.inferred_temperature(cygA)
    # plot.inferred_temperature(cygNW)

    # plot.inferred_pressure(cygA)
    # plot.inferred_pressure(cygNW)

    # plot.donnert2014_figure1(cygA, verlinde=False)
    # plot.donnert2014_figure1(cygNW, verlinde=False)

    # plot.donnert2014_figure1(cygA, verlinde=True)
    # plot.donnert2014_figure1(cygNW, verlinde=True)


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
    parse.write_toycluster_parameterfile(ic_cygA)
    parse.write_toycluster_parameterfile(ic_cygNW)
    parse.write_toycluster_parameterfile(ic_both)


def infer_toycluster_ics(a):
    # Fit the concentration parameter and baryon fraction
    mle, cis = fit.total_gravitating_mass_freecbf(
        ObservedCluster(a.basedir, "cygA", verbose=True, data=a.data),
        do_cut=a.do_cut)
    cygA = ObservedCluster(a.basedir, "cygA", cNFW=mle[0], bf=mle[1],
                           RCUT_R200_RATIO=mle[2] if a.do_cut else None,
                           verbose=a.verbose, data=a.data)

    mle, cis = fit.total_gravitating_mass_freecbf(
        ObservedCluster(a.basedir, "cygNW", verbose=False, data=a.data),
        do_cut=a.do_cut)
    cygNW = ObservedCluster(a.basedir, "cygNW", cNFW=mle[0], bf=mle[1],
                            RCUT_R200_RATIO=mle[2] if a.do_cut else None,
                            verbose=a.verbose, data=a.data)

    write_ics(cygA, cygNW)

    return cygA, cygNW


@profile
def set_observed_cluster(a, data_only=False):
    if a.data == "2Msec":
        if a.clustername == "cygA":
            if a.do_cut:
                cNFW = 7.1801
                bf = 0.0701
                RCUT_R200_RATIO = 854.7/1830.6
            else:  # not sure how much I trust these results...
                cNFW = 13.0000
                bf = 0.2271
                RCUT_R200_RATIO = None

        if a.clustername == "cygNW":
            if a.do_cut:
                cNFW = 2.8680
                bf = 0.0735
                RCUT_R200_RATIO = 986.1/1606.6
            else:  # not sure how much I trust these results...
                cNFW = 5.0780
                bf = 0.1879
                RCUT_R200_RATIO = None

    else:
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
        RCUT_R200_RATIO=RCUT_R200_RATIO, verbose=a.verbose, data=a.data,
        data_only=data_only)

    return obs


@profile
def set_observed_clusters(a, data_only=False):
    a.clustername = "cygA"
    cygA = set_observed_cluster(a, data_only=data_only)
    a.clustername = "cygNW"
    cygNW = set_observed_cluster(a, data_only=data_only)
    a.clustername = "both"
    return cygA, cygNW


def check_twocluster_ics(a):
    cygA = ObservedCluster(a.basedir, "cygA", cNFW=12.40, bf=0.07653, data=a.data)
    cygNW = ObservedCluster(a.basedir, "cygNW", cNFW=5.17, bf=0.05498, data=a.data)
    sim = Simulation(a.basedir, a.timestamp, name="both")
    plot.twocluster_quiescent_parm(cygA, cygNW, sim, 0, parm="kT")


def plot_smac_snapshots(a):
    if a.clustername:
        print "Running for single cluster", a.clustername
        obs = ObservedCluster(a.basedir, a.clustername,
                              verbose=a.verbose, data=a.data)
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
        cygA = ObservedCluster(a.basedir, "cygA", cNFW=cNFW,
                               verbose=a.verbose, data=a.data)
        plot.inferred_nfw_profile(cygA)
        plot.inferred_temperature(cygA)


def find_dm_peak(header, dm, expected, dim=0):
    if dim != 0 and dim != 1 and dim != 2:
        print "ERROR: please use integer '0', '1', or '2' as dimension in find_dm_peak"
        return None
    nbins = int(numpy.sqrt(header["ndm"]))
    hist, edges = numpy.histogram(dm[:,dim], bins=nbins, normed=True)
    edges = (edges[:-1] + edges[1:])/2

    # savgol = scipy.signal.savgol_filter(hist, 21, 5)
    hist_smooth = scipy.ndimage.filters.gaussian_filter1d(hist, 5)
    spline = scipy.interpolate.splrep(edges, hist_smooth)
    xval = numpy.arange(0, header["boxSize"], 0.1)
    hist_splev = scipy.interpolate.splev(xval, spline, der=0)
    peaks = peakutils.indexes(hist_splev)

    # pyplot.figure()
    # pyplot.plot(edges, hist)
    # pyplot.plot(xval, hist_splev)
    # pyplot.ylim(0, 1.1*numpy.max(hist))
    # pyplot.xlabel({ 0: "x", 1: "y", 2: "z"}.get(dim))
    # pyplot.ylabel("Normed Counts")
    # for peak in xval[peaks]: pyplot.axvline(peak)
    # pyplot.tight_layout()
    # pyplot.show()
    # pyplot.savefig(sim.outdir+"dm_peak_{0}".format(dim)+snapnr+".png", dpi=300)
    # pyplot.close()

    if len(peaks) != expected:
        print "ERROR: more than one {0}peak found".format(dim)
        return None

    return xval[peaks]


def find_dm_centroid(header, dm, single=False, verbose=True):
    if single:
        exp_x = 1
        exp_y = 1
        exp_z = 1
    else:  # two clusters w/o rotation (at same line)
        exp_x = 2
        exp_y = 1
        exp_z = 1
    xpeaks = find_dm_peak(header, dm, exp_x, 0)
    ypeaks = find_dm_peak(header, dm, exp_y, 1)
    zpeaks = find_dm_peak(header, dm, exp_z, 2)

    if type(xpeaks) != numpy.ndarray or type(ypeaks) != numpy.ndarray \
            or type(zpeaks) != numpy.ndarray : return None

    halo0 = xpeaks[0], ypeaks[0], zpeaks[0]
    halo1 = xpeaks[1 if exp_x == 2 else 0], ypeaks[1 if exp_y == 2 else 0], zpeaks[0]

    distance = numpy.sqrt((halo0[0] - halo1[0])**2 + (halo0[1] - halo1[1])**2 +
                          (halo0[2] - halo1[2])**2)
    if single: halo1 = None
    if verbose:
        print "    Success: found {0} xpeaks, {1} ypeak, and {2} zpeak!"\
            .format(exp_x, exp_y, exp_z)
        print "      halo0:  (x, y, z) = {0}".format(halo0)
        print "      halo1:  (x, y, z) = {0}".format(halo1)
        print "      distance          = {0:.2f} kpc\n".format(distance)
    return distance


#@concurrent(processes=threads)
def compute_distance(sim, snapnr, path_to_snaphot, verbose=False):
    if verbose:
        print "Checking", snapnr
    # sim.set_gadget_snap_double(snapnr, path_to_snaphot, verbose=verbose)
    # cygAsim = getattr(sim, "cygA{0}".format(snapnr), None)
    # cygNWsim = getattr(sim, "cygNW{0}".format(snapnr), None)

    header = parse.eat_f77(path_to_snaphot, "HEAD", verbose=verbose)
    pos = parse.eat_f77(path_to_snaphot, "POS", verbose=verbose)

    pos = pos.reshape((header["ntot"], 3))
    gas = pos[0:header["ngas"]]
    dm = pos[header["ngas"]:header["ntot"]]

    distance = find_dm_centroid(header, dm, verbose=verbose)
    return distance

    # median based has offset in finding com, and slight offset in centroids
    com = numpy.median(dm[:,0]), numpy.median(dm[:,1]), numpy.median(dm[:,2])
    left = numpy.where(dm[:,0] < com[0])
    right = numpy.where(dm[:,0] > com[0])

    halo = dm[left]
    depth = 7.5
    zslice = numpy.where( (halo[:,2] > com[2] - depth) & (halo[:,2] < com[2] + depth) )
    pyplot.figure()
    pyplot.scatter(halo[:,0][zslice], halo[:,1][zslice], c="r", lw=0, s=1)
    c0 = numpy.median(halo[:,0]), numpy.median(halo[:,1]), numpy.median(halo[:,2])
    pyplot.axvline(c0[0], c="r")
    pyplot.axhline(c0[1], c="r")

    halo = dm[right]
    zslice = numpy.where( (halo[:,2] > com[2] - depth) & (halo[:,2] < com[2] + depth) )
    pyplot.scatter(halo[:,0][zslice], halo[:,1][zslice], c="g", lw=0, s=1)
    c1 = numpy.median(halo[:,0]), numpy.median(halo[:,1]), numpy.median(halo[:,2])
    pyplot.axvline(c1[0], c="g")
    pyplot.axhline(c1[1], c="g")
    pyplot.gca().set_aspect("equal")
    pyplot.xlim(0, header["boxSize"])
    pyplot.ylim(0, header["boxSize"])
    pyplot.xlabel("x [kpc]")
    pyplot.ylabel("y [kpc]")

    pyplot.savefig(sim.outdir+"peakfind_{0:03d}".format(snapnr))

    distance = numpy.sqrt((c0[0] - c1[0])**2 + (c0[1] - c1[1])**2 + (c0[2] - c1[2])**2)

    return distance


# @synchronized
def find_and_plot_700_kpc(sim, verbose=False):
    if verbose: print "Running find_and_plot_700_kpc"

    sim.set_gadget_paths(verbose=verbose)
    # sim = copy.deepcopy(sim)

    distances = numpy.zeros(len(sim.gadget.snapshots))

    for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots):
        snapnr = int(path_to_snaphot[-3:])
        distances[snapnr] = compute_distance(sim, snapnr, path_to_snaphot, verbose=verbose).get()

    print distances


def compare_one_and_two_megaseconds(data_only=False):
    a.clusters = "both"
    a.data = "1Msec"

    cygA_1Msec, cygNW_1Msec = set_observed_clusters(a, data_only=data_only)

    a.data = "2Msec"
    cygA_2Msec, cygNW_2Msec = set_observed_clusters(a, data_only=data_only)

    clusters = {
        "cygA": [cygA_1Msec, cygA_2Msec],
        "cygNW": [cygNW_1Msec, cygNW_2Msec]
    }

    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1, "elinewidth": 1, "label": "data" }
    gas = { "color": "k", "lw": 1, "linestyle": "dotted", "label": "gas" }
    dm  = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "dm" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "tot" }

    for clustername in ["cygA", "cygNW"]:
        fig, ((ax0, ax1), (ax2, ax3)) = pyplot.subplots(2, 2, figsize=(18, 16))
        for color, c in zip(["g", "b"], clusters[clustername]):
            avg["color"] = color
            gas["color"] = color
            dm["color"] = color
            tot["color"] = color

            c.plot_chandra_average(ax0, parm="rho", style=avg)
            c.plot_bestfit_betamodel(ax0, style=gas, rho=True)
            if not data_only:
                c.plot_inferred_nfw_profile(ax0, style=dm, rho=True)

            c.plot_bestfit_betamodel_mass(ax1, style=gas)
            if not data_only:
                c.plot_inferred_nfw_mass(ax1, style=dm)
                c.plot_inferred_total_gravitating_mass(ax1, style=tot)
            c.plot_hydrostatic_mass_err(ax1, style=avg)

            c.plot_chandra_average(ax2, parm="kT", style=avg)
            if not data_only:
                c.plot_inferred_temperature(ax2, style=tot)

            c.plot_chandra_average(ax3, parm="P", style=avg)
            if not data_only:
                c.plot_inferred_pressure(ax3, style=tot)

        ax0.set_yscale("log")
        ax0.set_ylim(1e-30, 1e-22)
        ax1.set_yscale("log")
        ax1.set_ylim(1e5, 1e16)
        ax2.set_ylim(-1, 10)
        ax3.set_yscale("log")
        ax3.set_ylim(1e-15, 5e-9)

        for ax, loc in zip([ax0, ax1, ax2, ax3], [3, 2, 3, 3]):
            ax.set_xlabel("Radius [kpc]")
            ax.set_xscale("log")
            ax.set_xlim(0, 5000)
            ax.legend(fontsize=18, loc=loc)
            if data_only: continue
            ax.axvline(c.halo["r200"], c="k")
            # The y coordinates are axes while the x coordinates are data
            trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(c.halo["r200"]+150, 0.98, r"$r_{200}$", ha="left", va="top",
                    fontsize=22, transform=trans)
            ax.axvline(c.halo["r500"], c="k")
            ax.text(c.halo["r500"]-150, 0.98, r"$r_{500}$", ha="right", va="top",
                    fontsize=22, transform=trans)
        ax0.set_ylabel("Density [g/cm$^3$]")
        ax1.set_ylabel("Mass [MSun]")
        ax2.set_ylabel("Temperature [keV]")
        ax3.set_ylabel("Pressure [erg/cm$^3$]")

        fig.tight_layout()
        fig.savefig("out/1vs2Msec_{0}{1}.pdf".format(clustername,
            "_cut" if c.rcut_kpc is not None else ""))
        pyplot.close(fig)


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Simulation Pipeline Parser")
    args.add_argument("-t", "--timestamp", dest="timestamp",
        help="String of the Simulation ID", default="both_debug")
    args.add_argument("-b", "--basedir", dest="basedir",
        help="Path to the base directory", default="/usr/local/mscproj")
    args.add_argument("-c", "--clustername", dest="clustername",
        help="Name of the subcluster", default=None, choices=["cygA", "cygNW", "both"])
    args.add_argument("--chandra", dest="chandra", action="store_true",
        help="Generate observational plots", default=False)
    args.add_argument("--cut", dest="do_cut", action="store_true",
        help="Show analytical profiles with cut-off", default=False)
    args.add_argument("--gen1D", dest="gen1D", action="store_true",
        help="Generate 1D radial profiles plots for all snapshots", default=False)
    args.add_argument("--checkIC", dest="check_ics", action="store_true",
        help="Generate 1D radial profiles plots for ICs", default=False)
    args.add_argument("--infer", dest="infer", action="store_true",
        help="Infer DM model from ObservedCluster", default=False)
    args.add_argument("--compare", dest="compare", action="store_true",
        help="Compare 1Msec Chandra data with 2Msec Chandra data", default=False)
    args.add_argument("--best700", dest="find_700", action="store_true",
        help="Find bestfit 700 kpc snapshot", default=False)
    args.add_argument("--data", dest="data", default="2Msec",
        help="Exposuretime of Chandra observation", choices=["1Msec", "2Msec"])
    args.add_argument("-v", "--verbose", dest="verbose", action="store_true",
        help="Toggle verbose. Verbose is True by default", default=True)
    args.add_argument("-w", "--wise2018", dest="wise2018", action="store_true",
        help="Toggle Wise2018 to fit Vikhlinin2006 equations to data", default=False)
    args.add_argument("-d", "--debug", dest="debug", action="store_true",
        help="Toggle debug. Debug is False by default", default=False)
    args.add_argument("-e", "--embed", dest="embed", action="store_true",
        help="Toggle iPython embedding. Embed is False by default", default=False)
    # group = args.add_mutually_exclusive_group(required=True)
    # group.add_argument("-t", "--timestamp", dest="timestamp", nargs=1,
    #    help="string of the Simulation ID")

    return args


if __name__ == "__main__":
    a, unknown = new_argument_parser().parse_known_args()
    if a.embed: header = ""

    for k, v in vars(a).items():
        print("{0:<12} = {1}".format(k, v))
    print("")

    if a.compare:
        # python main.py --compare
        # python main.py --compare --cut
        compare_one_and_two_megaseconds()
        import sys; sys.exit(0)

    if a.infer:
        # python main.py --infer --data "2Msec"
        # python main.py --infer --data "2Msec" --cut
        cygA, cygNW = infer_toycluster_ics(a)
        import sys; sys.exit(0)

    # python main.py --chandra --data 1Msec -c "both"
    # python main.py --chandra --data 1Msec -c "both" --cut
    # python main.py --chandra --data 2Msec -c "both"
    # python main.py --chandra --data 2Msec -c "both" --cut
    if a.chandra:
        if a.clustername == "cygA":
            cygA = set_observed_cluster(a)
            plot.bestfit_betamodel(cygA)
        if a.clustername == "cygNW":
            cygNW = set_observed_cluster(a)
            plot.bestfit_betamodel(cygNW)
        if a.clustername == "both":
            cygA, cygNW = set_observed_clusters(a)

            plot.bestfit_betamodel(cygA)
            plot.bestfit_betamodel(cygNW)
            show_observations(cygA, cygNW)

        import sys; sys.exit(0)

    if a.wise2018:
        if a.clustername == "cygA":
            cygA = set_observed_cluster(a)
        if a.clustername == "cygNW":
            cygNW = set_observed_cluster(a)
        if a.clustername == "both":
            cygA, cygNW = set_observed_clusters(a)
            fit.wise2018_temperature_and_density(cygA)
            fit.wise2018_temperature_and_density(cygNW)
            plot.wise2018_temperature_and_density(cygA)
            plot.wise2018_temperature_and_density(cygNW)

        import sys; sys.exit(0)


    sim = Simulation(base=a.basedir, name=a.clustername, timestamp=a.timestamp, set_data=False)

    # plot.make_temperature_video(sim)
    # import sys; sys.exit(0)

    if a.embed: header += "Simulation instance in `sim'\n"

    if a.find_700:
        find_and_plot_700_kpc(sim)
        import sys; sys.exit(0)


    if a.clustername == "both":
        cygA, cygNW = set_observed_clusters(a)
        # Remove unpickleable items in ObservedCluster.__dict__
        if a.gen1D:
            del(cygA.HE_T)
            del(cygNW.HE_T)
            del(cygA.HE_dT_dr)
            del(cygNW.HE_dT_dr)
            del(cygA.T_spline)
            del(cygNW.T_spline)
            plot.twocluster_stability(sim, cygA, cygNW, verbose=a.verbose)
        if a.embed: header += "ObservedCluster instances in `cygA' and `cygNW'\n"

    if a.clustername == "cygA" or a.clustername == "cygNW":
        obs = set_observed_cluster(a)
        if a.gen1D:
            del(obs.HE_T)
            del(obs.HE_dT_dr)
            del(obs.T_spline)
            plot.singlecluster_stability(sim, obs, verbose=a.verbose)
        if a.embed: header += "ObservedCluster instance in `obs'\n"

    if a.check_ics:
        sim.read_ics(verbose=True)

        sim.toy.halo0.name = "cygA"
        fignum = plot.donnert2014_figure1(cygA, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, sim.toy.halo0, sim.outdir)

        sim.toy.halo1.name = "cygNW"
        fignum = plot.donnert2014_figure1(cygNW, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, sim.toy.halo1, sim.outdir)

    # fig = plot.donnert2014_figure1(obs)
    # plot.addsim sim and snapnr

    # plot.donnert2014_figure1(obs, verlinde=False)

    # plot_smac_snapshots(a)

    if a.embed:
        import IPython
        IPython.embed(banner1="", header=header)
