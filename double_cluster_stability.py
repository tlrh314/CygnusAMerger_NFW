import os
import argparse
import numpy
import scipy
import peakutils
import matplotlib
from matplotlib import pyplot

import main
import convert
import fit
import plot
import parse
from cluster import ObservedCluster
from simulation import Simulation
from macro import *
from line_profiler_support import profile

from plotsettings import PlotSettings
style = PlotSettings()


@profile
def loop_snaps(cygA, cygNW, sim):
    if not hasattr(sim.gadget, "snapshots"): return

    for path_to_snaphot in sim.gadget.snapshots:
        print path_to_snaphot
        sim.current_snapnr = path_to_snaphot.split("_")[-1]

        sim.toy.header, sim.toy.gas, sim.toy.dm = parse.toycluster_icfile(path_to_snaphot)
        sim.toy.gas["rho"] = convert.toycluster_units_to_cgs(sim.toy.gas["rho"])
        genplots(cygA, cygNW, sim, i=int(sim.current_snapnr))

        # sim.toy.gas["kT"] = convert.K_to_keV(convert.gadget_u_to_t(sim.toy.gas["u"]))

        # sim.toy.set_gas_mass()
        # sim.toy.set_gas_pressure()
        # sim.toy.M_dm_tot = sim.toy.header["ndm"] * sim.toy.header["massarr"][1] * 1e10
        # sim.toy.M_gas_tot = sim.toy.header["ngas"] * sim.toy.header["massarr"][0] * 1e10
        # sim.toy.set_dm_mass()
        # sim.toy.set_dm_density()
        # plot.donnert2014_figure1(obs, sim, verlinde=False)
        # print


@profile
def genplots(cygA, cygNW, sim, i=None):
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 1, "alpha": 1, "elinewidth": 1,
            "capsize": 0}
    gas = { "marker": "o", "ls": "", "c": "g", "ms": 2, "alpha": 1,
            "markeredgecolor": "none",  "label": ""}
    dm  = { "marker": "o", "ls": "", "c": "k", "ms": 2, "alpha": 1,
            "markeredgecolor": "none",  "label": ""}

    snapnr = "_{0:03d}".format(i) if i is not None else ""

    if i is None:
        # Compute and show analytical profiles versus data: kT
        pyplot.figure()

        pyplot.semilogx(sim.toy.profiles["000"]["r"],
            convert.K_to_keV(convert.gadget_u_to_t(sim.toy.profiles["000"]["u_gas"])),
            label="u\_gas")
        pyplot.semilogx(sim.toy.profiles["000"]["r"],
            convert.K_to_keV(convert.gadget_u_to_t(sim.toy.profiles["000"]["u_ana"])),
            label="u\_ana")
        cygA.plot_chandra_average(parm="kT", style=avg)

        pyplot.semilogx(sim.toy.profiles["001"]["r"],
            convert.K_to_keV(convert.gadget_u_to_t(sim.toy.profiles["001"]["u_gas"])),
            label="u\_gas")
        pyplot.semilogx(sim.toy.profiles["001"]["r"],
            convert.K_to_keV(convert.gadget_u_to_t(sim.toy.profiles["001"]["u_ana"])),
            label="u\_ana")
        cygNW.plot_chandra_average(parm="kT", style=avg)

        pyplot.xlim(1, 5000)
        pyplot.ylim(-1, 10)
        pyplot.xlabel("Radius [kpc]")
        pyplot.ylabel("Temperature [keV]")
        pyplot.legend(fontsize=12)
        pyplot.tight_layout()
        pyplot.savefig(sim.outdir+"both_kT_ana"+snapnr+".png", dpi=300)
        pyplot.close()


        # Compute and show analytical profiles versus data: rho
        pyplot.figure()

        pyplot.loglog(sim.toy.profiles["000"]["r"],
            convert.toycluster_units_to_cgs(sim.toy.profiles["000"]["rho_gas"]))
        cygA.plot_chandra_average(parm="rho", style=avg)

        pyplot.loglog(sim.toy.profiles["001"]["r"],
            convert.toycluster_units_to_cgs(sim.toy.profiles["001"]["rho_gas"]))
        cygNW.plot_chandra_average(parm="rho", style=avg)

        pyplot.xlim(1, 5000)
        pyplot.ylim(1e-30, 5e-22)
        pyplot.xlabel("Radius [kpc]")
        pyplot.ylabel("Density [g/cm$^3$]")
        pyplot.tight_layout()
        pyplot.savefig(sim.outdir+"both_rho_ana"+snapnr+".png", dpi=300)
        pyplot.close()

    @profile
    def find_peaks(dim="x"):
        boxsize = sim.toy.header["boxSize"]
        boxhalf = sim.toy.header["boxSize"]/2

        if dim != "x" and dim != "y" and dim != "z":
            print "ERROR: please use 'x', 'y', or 'z' as dimension in find_peaks"
            return None
        nbins = int(numpy.sqrt(sim.toy.header["ndm"]))
        hist, edges = numpy.histogram(sim.toy.dm[dim], bins=nbins, normed=True)
        edges = (edges[:-1] + edges[1:])/2

        # savgol = scipy.signal.savgol_filter(hist, 21, 5)
        hist_smooth = scipy.ndimage.filters.gaussian_filter1d(hist, 5)
        spline = scipy.interpolate.splrep(edges, hist_smooth)  # built-in smoothing breaks
        xval = numpy.arange(0, boxsize, 0.1)
        hist_splev = scipy.interpolate.splev(xval, spline, der=0)
        peaks = peakutils.indexes(hist_splev)

        pyplot.figure()
        pyplot.plot(edges, hist, **dm)
        pyplot.plot(xval, hist_splev)
        pyplot.ylim(0, 1.1*numpy.max(hist))
        pyplot.xlabel(dim)
        pyplot.ylabel("Normed Counts")
        for peak in xval[peaks]: pyplot.axvline(peak)
        pyplot.tight_layout()
        pyplot.savefig(sim.outdir+"dm_peak_{0}".format(dim)+snapnr+".png", dpi=300)
        pyplot.close()

        return xval[peaks]

    # TODO: not enough resolution in the histogram to find two y peaks
    # (if impactparam != 0). We could split the haloes based on x-position
    # and then look for the y peaks in sim.toy.dm["y"]
    xpeaks = find_peaks("x")
    if len(xpeaks) != 2:
        print "ERROR: cluster centroids along merger axis not found"
        return
    ypeaks = find_peaks("y")
    if len(ypeaks) != 1:
        print "ERROR: more than one ypeak found"
        return
    zpeaks = find_peaks("z")
    if len(zpeaks) != 1:
        print "ERROR: more than one zpeak found"
        return

    if sim.toy.parms["ImpactParam"] == 0.0:
        pass  # TODO: investigate if this makes profiles less puffy
        # ypeaks[0] = 0.0
        # zpeaks[0] = 0.0

    # It is important to get this right, otherwise we plot puffy profiles while
    # in fact the profiles could be very sharp...
    # Toycluster xpeaks and ypeaks is printed in the runtime output
    print xpeaks, ypeaks, zpeaks

    # Assign particles to left or right halo
    com = (xpeaks[0]+xpeaks[1])/2   # in the middle between both haloes
    zslice = numpy.where(sim.toy.gas["z"] > 0)
    zslice2 = numpy.where(sim.toy.gas["z"] < 20000)
    zslice = (numpy.intersect1d(zslice, zslice2),)
    left = numpy.where(sim.toy.gas["x"] < com)
    leftzslice = (numpy.intersect1d(left, zslice),)
    right = numpy.where(sim.toy.gas["x"] > com)
    rightzslice = (numpy.intersect1d(right, zslice),)

    # Show slice in z direction of particle-to-halo assignment
    pyplot.figure()
    pyplot.scatter(sim.toy.gas["x"][leftzslice], sim.toy.gas["y"][leftzslice], c="r", edgecolor="r", s=1)
    pyplot.scatter(sim.toy.gas["x"][rightzslice], sim.toy.gas["y"][rightzslice], c="g", edgecolor="g",  s=1)
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"halo_assignment"+snapnr+".png", dpi=300)
    pyplot.close()


    xleft = sim.toy.gas["x"][left] - xpeaks[0]
    yleft = sim.toy.gas["y"][left] - ypeaks[0]
    zleft = sim.toy.gas["z"][left] - zpeaks[0]
    rleft = numpy.sqrt(p2(xleft) + p2(yleft) + p2(zleft))
    rholeft = sim.toy.gas["rho"][left]  # Toycluster RHO is in cgs ?!
    kTleft = convert.K_to_keV(convert.gadget_u_to_t(sim.toy.gas["u"][left]))

    xright = sim.toy.gas["x"][right] - xpeaks[1]
    yright = sim.toy.gas["y"][right] - ypeaks[0]
    zright = sim.toy.gas["z"][right] - zpeaks[0]
    rright = numpy.sqrt(p2(xright) + p2(yright) + p2(zright))
    rhoright = sim.toy.gas["rho"][right]  # Toycluster RHO is in cgs ?!
    kTright = convert.K_to_keV(convert.gadget_u_to_t(sim.toy.gas["u"][right]))

    pyplot.figure()
    pyplot.loglog(rleft, rholeft, **gas)
    cygA.plot_chandra_average(parm="rho", style=avg)
    pyplot.loglog(sim.toy.profiles["000"]["r"], convert.toycluster_units_to_cgs(
        sim.toy.profiles["000"]["rho_gas"]), c="k")
    inner = numpy.where(rleft < 100)
    hsml = 2*numpy.median(sim.toy.gas["hsml"][left][inner])
    pyplot.axvline(x=hsml, c="g", ls=":")
    trans = matplotlib.transforms.blended_transform_factory(
        pyplot.gca().transData, pyplot.gca().transAxes)
    pyplot.text(hsml, 0.05, r"$2 h_{sml}$", ha="left", color="g",
                transform=trans, fontsize=22)
    pyplot.xlim(1, 5000)
    pyplot.ylim(1e-30, 5e-25)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Density [g/cm$^3$]")
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"cygA_sampled_rho"+snapnr+".png", dpi=300)
    pyplot.close()

    pyplot.figure()
    pyplot.loglog(rright, rhoright, **gas)
    cygNW.plot_chandra_average(parm="rho", style=avg)
    pyplot.loglog(sim.toy.profiles["001"]["r"], convert.toycluster_units_to_cgs(
        sim.toy.profiles["001"]["rho_gas"]), c="k")
    inner = numpy.where(rright < 100)
    hsml = 2*numpy.median(sim.toy.gas["hsml"][right][inner])
    trans = matplotlib.transforms.blended_transform_factory(
        pyplot.gca().transData, pyplot.gca().transAxes)
    pyplot.axvline(x=hsml, c="g", ls=":")
    pyplot.text(hsml, 0.05, r"$2 h_{sml}$", ha="left", color="g",
                transform=trans, fontsize=22)
    pyplot.xlim(1, 5000)
    pyplot.ylim(1e-30, 5e-25)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Density [g/cm$^3$]")
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"cygNW_sampled_kT"+snapnr+".png", dpi=300)
    pyplot.close()

    pyplot.figure()
    pyplot.semilogx(rleft, kTleft, **gas)
    cygA.plot_chandra_average(parm="kT", style=avg)
    pyplot.semilogx(sim.toy.profiles["000"]["r"], convert.K_to_keV(
        convert.gadget_u_to_t(sim.toy.profiles["000"]["u_gas"])), c="k")
    inner = numpy.where(rleft < 100)
    hsml = 2*numpy.median(sim.toy.gas["hsml"][left][inner])
    pyplot.axvline(x=hsml, c="g", ls=":")
    trans = matplotlib.transforms.blended_transform_factory(
        pyplot.gca().transData, pyplot.gca().transAxes)
    pyplot.text(hsml, 0.05, r"$2 h_{sml}$", ha="left", color="g",
                transform=trans, fontsize=22)
    pyplot.xlim(1, 5000)
    pyplot.ylim(-1, 10)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Temperature [keV]]")
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"cygA_sampled_kT"+snapnr+".png", dpi=300)
    pyplot.close()

    pyplot.figure()
    # TODO: add time counter
    # TODO: plot modulo 10
    # TODO: make parallel
    pyplot.semilogx(rright, kTright, **gas)
    cygNW.plot_chandra_average(parm="kT", style=avg)
    pyplot.semilogx(sim.toy.profiles["001"]["r"], convert.K_to_keV(
        convert.gadget_u_to_t(sim.toy.profiles["001"]["u_gas"])), c="k")
    inner = numpy.where(rright < 100)
    hsml = 2*numpy.median(sim.toy.gas["hsml"][right][inner])
    trans = matplotlib.transforms.blended_transform_factory(
        pyplot.gca().transData, pyplot.gca().transAxes)
    pyplot.axvline(x=hsml, c="g", ls=":")
    pyplot.text(hsml, 0.05, r"$2 h_{sml}$", ha="left", color="g",
                transform=trans, fontsize=22)
    pyplot.xlim(1, 5000)
    pyplot.ylim(-1, 10)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Temperature [keV]")
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"cygNW_sampled_rho"+snapnr+".png", dpi=300)
    pyplot.close()


@profile
def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Simulation Pipeline Parser")
    args.add_argument("-t", "--timestamp", dest="timestamp",
        help="String of the Simulation ID", default="20161124T0148")
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
    # group = args.add_mutually_exclusive_group(required=True)
    # group.add_argument("-t", "--timestamp", dest="timestamp", nargs=1,
    #    help="string of the Simulation ID")

    return args

@profile
def run_double_clusterstability():
    a = new_argument_parser().parse_args()

    a.clustername = "cygA"
    cygA = main.set_observed_cluster(a)
    a.clustername = "cygNW"
    cygNW = main.set_observed_cluster(a)
    a.clustername = "both"

    sim = Simulation(base=a.basedir, name="both", timestamp=a.timestamp)

    genplots(cygA, cygNW, sim)
    loop_snaps(cygA, cygNW, sim)

    # plot.twocluster_parms(cygA, cygNW)
    # plot.donnert2014_figure1(cygA)
    # plot.donnert2014_figure1(cygNW)

if __name__ == "__main__":
    run_double_clusterstability()
