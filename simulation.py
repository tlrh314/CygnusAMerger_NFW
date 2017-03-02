# -*- coding: utf-8 -*-

import os
import glob
import numpy
import scipy
from scipy import signal
import peakutils

import parse
import plot
import convert
from macro import p2, p3, print_progressbar
from cluster import Toycluster
from cluster import Cluster
from cluster import Gadget3Output
from cluster import PSmac2Output
from panda import create_panda

# ----------------------------------------------------------------------------
# Class to set simulation paths and the like
# ----------------------------------------------------------------------------
class Simulation(object):
    def __init__(self, base, name, timestamp, set_data=True, verbose=True):
        """ Set the simulation output paths
            @param base     : base path where 'runs' dir is stored, string
            @param timestamp: SimulationID; name of subfolder in 'runs', string
            @param name     : by default, simulation snapshots contain two clusters
                              if only one cluster is sampled, specify the name
                              of the subcluster, either "cygA" or "cygNW", string
            @return         : class instance holding correct paths and parsed data"""

        if name and (name != "cygA" and name != "cygNW" and name != "both"):
            print "ERROR: incorrect usage of 'SimulationOutputParser'. Use",
            print "either 'cygA' or 'cygNW' as value of parameter 'name'"
            return
        self.name = name

        self.timestamp = timestamp
        self.rundir = "{0}/runs/{1}/".format(base, self.timestamp)
        if not (os.path.isdir(self.rundir) or os.path.exists(self.rundir)):
            print "ERROR: directory '{0}' does not exist.".format(self.rundir)
            return

        self.icsdir = self.rundir+"ICs/"
        self.simdir = self.rundir+"snaps/"
        self.analysisdir = self.rundir+"analysis/"

        self.outdir = self.rundir+"out/"
        if not (os.path.isdir(self.outdir) or os.path.exists(self.outdir)):
            os.mkdir(self.outdir)
            print "Created directory {0}".format(self.outdir)

        if verbose: print self
        if set_data:
            self.read_ics(verbose)
            self.set_gadget_paths(verbose)
            self.read_smac(verbose)

    def __str__(self):
        tmp = "Simulation: {0} --> {1}\n".format(self.timestamp, self.name)
        tmp += "  rundir     : {0}\n".format(self.rundir)
        tmp += "  icsdir     : {0}\n".format(self.icsdir)
        tmp += "  simdir     : {0}\n".format(self.simdir)
        tmp += "  analysisdir: {0}\n".format(self.analysisdir)
        tmp += "  outdir     : {0}\n".format(self.outdir)
        return tmp

    def read_ics(self, verbose=False):
        if verbose: print "  Parsing Toycluster output"
        if self.name != "both":
            self.toy = Toycluster(self.icsdir, single=True, verbose=verbose)
            if verbose: print "  Succesfully loaded ICs"
            if verbose: print "  {0}".format(self.toy)
        else:
            self.toy = Toycluster(self.icsdir, single=False, verbose=verbose)
            if verbose: print "  Succesfully loaded ICs"
            if verbose: print "  {0}".format(self.toy)

        if verbose: print ""

    def set_gadget_paths(self, verbose=False):
        if verbose: print "  Parsing Gadet-3 output"
        if not (os.path.isdir(self.simdir) or os.path.exists(self.simdir)):
            print "    Directory '{0}' does not exist.".format(self.simdir)
            return

        self.gadget = Gadget3Output(self.simdir, verbose=verbose)
        self.dt = self.gadget.parms['TimeBetSnapshot']

        if verbose:
            print "  Succesfully loaded snaps"
            print "  Snapshot paths:"
            for p in self.gadget.snapshots: print " "*4+p
            print "\n  {0}".format(self.gadget)
            print

    def plot_ics(self, obs):
        if not hasattr(self, "toy"):
            print "ERROR: simulation has no Toycluster instance"
            return

        fig = plot.donnert2014_figure1(obs, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fig, self.toy,  self.outdir)
        plot.toycluster_profiles(obs, self)
        plot.toyclustercheck(obs, self)
        plot.toyclustercheck_T(obs, self)

    def plot_twocluster_stability(self, cygA, cygNW, verbose=True):
        if not hasattr(self, "gadget"):
            print "ERROR: simulation has no Gadget3Output instance"
            return

        # TODO: run parallel, but deco does not work for class methods
        if verbose: print "Running plot_twocluster_stability"
        for snapnr, path_to_snaphot in enumerate(self.gadget.snapshots):
            if verbose: print "  {0}: {1}".format(snapnr, path_to_snaphot)

    def set_gadget_snap_single(self, snapnr, path_to_snaphot, verbose=False):
        h = Cluster(None, verbose=verbose)
        h.name = "snap{0}".format(snapnr)
        h.time = snapnr*self.dt
        h.set_gadget_single_halo(snapnr, path_to_snaphot, verbose=verbose)
        setattr(self, h.name, h)

    def unset_gadget_snap_single(self, snapnr):
        # 'free' the mem b/c want to parallelise
        delattr(self, "snap{0}".format(snapnr))
        #  if i%16 == 0: gc.collect()  # TODO: test necessity
        print

    def set_gadget_snap_double(self, snapnr, path_to_snaphot, verbose=False):
        if verbose: print "    Found two clusters in box --> running find_dm_centroid"

        header, gas, dm = parse.toycluster_icfile(path_to_snaphot, verbose=verbose)
        gas["rho"] = convert.toycluster_units_to_cgs(gas["rho"])
        gas["kT"] = convert.K_to_keV(convert.gadget_u_to_t(gas["u"]))

        # First find dark matter centroids
        tmp = Cluster(header, verbose=verbose)
        tmp.set_header_properties()
        tmp.dm = dm
        tmp.parms = parse.read_toycluster_parameterfile(glob.glob(self.icsdir+"*.par")[0])

        if not tmp.find_dm_centroid(single=False, verbose=verbose):
            print "ERROR: find_dm_centroid failed!"
            return

        # Assign particles to left or right halo
        self.com = (tmp.centroid0[0] + tmp.centroid1[0])/2   # midpoint between both haloes
        left = numpy.where(gas["x"] < self.com)
        right = numpy.where(gas["x"] > self.com)

        h0 = Cluster(header, verbose=verbose)
        h0.name = "cygA{0}".format(snapnr)
        h0.time = snapnr*self.dt
        h0.set_gadget_double_halo(gas[left], dm[left], tmp.centroid0, verbose=verbose)
        setattr(self, h0.name, h0)

        h1 = Cluster(header, verbose=verbose)
        h1.name = "cygNW{0}".format(snapnr)
        h1.time = snapnr*self.dt
        h1.set_gadget_double_halo(gas[right], dm[right], tmp.centroid1, verbose=verbose)
        setattr(self, h1.name, h1)

    def read_smac(self, verbose=False):
        if verbose: print "  Parsing P-Smac2 output"
        if not (os.path.isdir(self.analysisdir) or os.path.exists(self.analysisdir)):
            print "    Directory '{0}' does not exist.".format(self.analysisdir)
            return

        self.psmac = PSmac2Output(self, verbose=verbose)
        if verbose: print "  Succesfully loaded P-Smac2 fitsfiles"
        if verbose: print "  {0}".format(self.psmac)
        # assumes xray cube is loaded
        if hasattr(self.psmac, "xray"):
            self.nsnaps, self.xlen, self.ylen = self.psmac.xray.shape
            self.pixelscale = float(self.psmac.xray_header["XYSize"])/int(self.xlen)
        else:
            print "Warning: Simulation instance attributes 'nsnaps', 'xlen', 'ylen', 'pixelscale' not set"

    def find_cluster_centroids_psmac_dmrho(self, snapnr=0, EA2="", do_plot=False):
        print "Checking snapshot {0}".format(snapnr)

        if self.name and self.name != "both":  # Only one cluster in simulation box
            expected_xpeaks = 1
            expected_ypeaks = 1
            thres, min_dist = 0.9, 20
        # elif self.toy.parms["ImpactParam"] < 0.1:  # two clusters, no impact parameter
        #     expected_xpeaks = 2
        #     expected_ypeaks = 1
        #     thres, min_dist = 0.15, 10
        else:  # two clusters, yes impact parameter, or yes rotated /w EA1
            expected_xpeaks = 2
            expected_ypeaks = 2
            thres, min_dist = 0.35, 100

        """ Peakutils sometimes finds noise (i.e. 1 pixel with a slightly higher
        density, where slightly is no more than 0.1%). To kill of these tiny noise
        fluctuations the summed dark matter density is squared, then normalised
        to the maximum, and finally smoothed with a Savitzky-Golay filter. """
        rhodm = getattr(self.psmac, "rhodm{0}best".format(EA2), None)
        if rhodm is None: return
        xsum = p2(numpy.sum(rhodm[snapnr], axis=0))
        xsum /= numpy.max(xsum)
        xsum = scipy.signal.savgol_filter(xsum, 15, 3)
        ysum = p2(numpy.sum(rhodm[snapnr], axis=1))
        ysum /= numpy.max(ysum)
        ysum = scipy.signal.savgol_filter(ysum, 15, 3)
        xpeaks = peakutils.indexes(xsum, thres=thres, min_dist=min_dist)
        ypeaks = peakutils.indexes(ysum, thres=thres, min_dist=min_dist)

        if not (len(xpeaks) == expected_xpeaks and len(ypeaks) == expected_ypeaks):
            print "ERROR: found incorrect number of peaks in dmrho"
            print "xpeaks = {0}\nypeaks = {1}".format(xpeaks, ypeaks)
            print "Snapshot number = {0}\n".format(snapnr)
            if do_plot: plot.psmac_xrays_with_dmrho_peakfind(
                self, snapnr, xsum, ysum, xpeaks, ypeaks, numpy.nan, EA2=EA2)
            return xpeaks, ypeaks, numpy.nan

        try:  # Further optimize peakfinding by interpolating
            xpeaks = peakutils.interpolate(range(0, self.xlen), xsum, ind=xpeaks)
            ypeaks = peakutils.interpolate(range(0, self.ylen), ysum, ind=ypeaks)
        except RuntimeError as err:
            if "Optimal parameters not found: Number of calls to function has reached" in str(err):
                print "WARNING: peakutils.interpolate broke, using integer values"
            else:
                raise

        if self.name and self.name != "both":  # Only one cluster in simulation box
            center = xpeaks[0], ypeaks[0]
            print "Success: found 1 xpeak, and 1 ypeak!"
            print "  {0}:  (x, y) = {1}".format(self.name, center)
            if do_plot: plot.psmac_xrays_with_dmrho_peakfind(
                self, snapnr, xsum, ysum, xpeaks, ypeaks, numpy.nan, EA2=EA2)
            return center
        else:
            #if True:  # self.toy.parms["ImpactParam"] < 0.1:  # two clusters, no impact parameter
            #    cygA = xpeaks[0], ypeaks[0]
            #    cygNW = xpeaks[1], ypeaks[0]
            #else:  # two clusters, yes impact parameter.
            cygA = xpeaks[0], ypeaks[0]
            cygNW = xpeaks[1], ypeaks[1]

            distance = numpy.sqrt(p2(cygA[0]-cygNW[0])+p2(cygA[1]-cygNW[1]))
            distance *= self.pixelscale
            print "Success: found {0} xpeaks, and {1} ypeak!"\
                .format(expected_xpeaks, expected_ypeaks)
            print "  cygA:  (x, y) = {0}".format(cygA)
            print "  cygNW: (x, y) = {0}".format(cygNW)
            print "  distance      = {0:.2f} kpc\n".format(distance)
            if do_plot: plot.psmac_xrays_with_dmrho_peakfind(
                self, snapnr, xsum, ysum, xpeaks, ypeaks, distance, EA2=EA2)
            return cygA, cygNW, distance

    def create_quiescent_profile(self, snapnr, parm="tspec", do_plot=True):
        cube = getattr(self.psmac, parm, None)
        if cube is None:
            print "ERROR: sim.psmac does not have attribute '{0}'".format(parm)
            print "       available:", self.psmac.available_smac_cubes()
            return
        parmdata = cube[snapnr]

        unitfix = { "tspec": convert.K_to_keV }
        unitfix = unitfix.get(parm, None)
        if not unitfix: print "ERROR: unitfix not given for '{0}'".format(parm)

        rmax = 900/self.pixelscale
        radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(rmax), 42))
        dr = radii[1:] - radii[:-1]
        radii = radii[:-1]
        N = len(radii)
        quiescent_temperature = numpy.zeros(N)
        quiescent_temperature_std = numpy.zeros(N)
        if do_plot:
            from matplotlib import pyplot
            fig = pyplot.figure(figsize=(12, 12))
        if self.name and self.name != "both":  # Only one cluster in simulation box
            xc, yc = self.find_cluster_centroids_psmac_dmrho(snapnr=snapnr, EA2="")
            for i, r in enumerate(radii):
                print_progressbar(i, N)
                quiescent_mask = create_panda(self.xlen, self.ylen, xc, yc, r, 45, -45)
                quiescent_temperature[i] = numpy.median(parmdata[quiescent_mask])
                quiescent_temperature_std[i] = numpy.std(parmdata[quiescent_mask])
                y, x = numpy.where(quiescent_mask)
                if do_plot: pyplot.scatter(x, y, s=1, c="r", edgecolor="face", alpha=1)
            if do_plot:
                pyplot.imshow(parmdata, origin="lower", cmap="afmhot")
                pyplot.xlabel("x [pixel]")
                pyplot.ylabel("y [pixel]")
                pyplot.gca().set_aspect("equal")
                pyplot.xlim(800, 1400)
                pyplot.ylim(self.xlen/2-300, self.xlen/2+300)
                pyplot.savefig(self.outdir+"{0}_{1:03d}.png".format(parm, snapnr))
                pyplot.close()
            return radii*self.pixelscale, unitfix(quiescent_temperature),\
                unitfix(quiescent_temperature_std)
        else:
            cygA, cygNW, distance = self.find_cluster_centroids_psmac_dmrho(snapnr=snapnr, EA2=0)
            results = dict()
            for (xc, yc), name in zip([cygA, cygNW], ["cygA", "cygNW"]):
                for i, r in enumerate(radii):
                    print_progressbar(i, N)
                    angle1 = 45 if name == "cygA" else 135
                    angle2 = -45 if name == "cygA" else 225
                    quiescent_mask = create_panda(self.xlen, self.ylen, xc, yc,
                                                  r, angle1, angle2)
                    quiescent_temperature[i] = numpy.median(parmdata[quiescent_mask])
                    quiescent_temperature_std[i] = numpy.std(parmdata[quiescent_mask])
                    y, x = numpy.where(quiescent_mask)
                    if do_plot: pyplot.scatter(x, y, s=1, c="r", edgecolor="face", alpha=1)
                if do_plot:
                    pyplot.imshow(parmdata, origin="lower", cmap="afmhot")
                    pyplot.xlabel("x [pixel]")
                    pyplot.ylabel("y [pixel]")
                    pyplot.gca().set_aspect("equal")
                    pyplot.xlim(300, 1500)
                    pyplot.ylim(self.xlen/2-300, self.xlen/2+300)
                    pyplot.savefig(self.outdir+"{0}_{1}_{2:03d}.png"
                                   .format(name, parm, snapnr))
                    pyplot.close()
                    results[name] = unitfix(quiescent_temperature),\
                        unitfix(quiescent_temperature_std)
            return radii*self.pixelscale, results
