import os
import numpy
import scipy
from scipy import signal
import peakutils

import parse
import plot
from macro import p2, p3
from cluster import Toycluster
from cluster import Gadget2Output
from cluster import PSmac2Output

# ----------------------------------------------------------------------------
# Class to set simulation paths and the like
# ----------------------------------------------------------------------------
class Simulation(object):
    def __init__(self, base, timestamp, name=None, verbose=True):
        """ Set the simulation output paths
            @param base     : base path where 'runs' dir is stored, string
            @param timestamp: SimulationID; name of subfolder in 'runs', string
            @param name     : by default, simulation snapshots contain two clusters
                              if only one cluster is sampled, specify the name
                              of the subcluster, either "cygA" or "cygNW", string
            @return         : class instance holding correct paths and parsed data"""

        if name and (name != "cygA" and name != "cygNW"):
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
            print "Created directory {0}!".format(self.outdir)

        if verbose: print self
        self.read_ics(verbose)
        self.set_gadget(verbose)
        self.read_smac(verbose)

    def __str__(self):
        tmp = "Simulation: {0}\n".format(self.timestamp)
        tmp += "  rundir     : {0}\n".format(self.rundir)
        tmp += "  icsdir     : {0}\n".format(self.icsdir)
        tmp += "  simdir     : {0}\n".format(self.simdir)
        tmp += "  analysisdir: {0}\n".format(self.analysisdir)
        tmp += "  outdir     : {0}\n".format(self.outdir)
        return tmp

    def read_ics(self, verbose=False):
        if verbose: print "  Parsing Toycluster output"
        self.toy = Toycluster(self.icsdir, verbose=verbose)
        if verbose: print "  Succesfully loaded ICs"
        if verbose: print "  {0}".format(self.toy)

    def set_gadget(self, verbose=False):
        if verbose: print "  Parsing Gadet-2 output"
        self.gadget = Gadget2Output(self.simdir, verbose=verbose)
        self.dt = self.gadget.parms['TimeBetSnapshot']
        if verbose: print "  Succesfully loaded snaps"
        if verbose: print "  {0}".format(self.gadget)

    def eat_gadget(self, snapnr, verbose=False):
        pass

    def read_smac(self, verbose=False):
        if verbose: print "  Parsing P-Smac2 output"
        self.psmac = PSmac2Output(self, verbose=verbose)
        if verbose: print "  Succesfully loaded P-Smac2 fitsfiles"
        if verbose: print "  {0}".format(self.psmac)
        # assumes xray cube is loaded
        self.nsnaps, self.xlen, self.ylen = self.psmac.xray.shape
        self.pixelscale = float(self.psmac.xray_header["XYSize"])/int(self.xlen)

    def find_cluster_centroids_psmac_dmrho(self, snapnr=0):
        print "Checking snapshot {0}".format(snapnr)

        if self.name:  # Only one cluster in simulation box
            expected_xpeaks = 1
            expected_ypeaks = 1
            thres, min_dist = 0.9, 20
        elif self.toy.parms["ImpactParam"] < 0.1:  # two clusters, no impact parameter
            expected_xpeaks = 2
            expected_ypeaks = 1
            thres, min_dist = 0.15, 10
        else:  # two clusters, yes impact parameter
            expected_xpeaks = 2
            expected_ypeaks = 2
            # thres, min_dist = 0.4, 20  # TODO: check. Want error for now

        """ Peakutils sometimes finds noise (i.e. 1 pixel with a slightly higher
        density, where slightly is no more than 0.1%). To kill of these tiny noise
        fluctuations the summed dark matter density is squared, then normalised
        to the maximum, and finally smoothed with a Savitzky-Golay filter. """
        xsum = p2(numpy.sum(self.psmac.rhodm[snapnr], axis=0))
        xsum /= numpy.max(xsum)
        xsum = scipy.signal.savgol_filter(xsum, 15, 3)
        ysum = p2(numpy.sum(self.psmac.rhodm[snapnr], axis=1))
        ysum /= numpy.max(ysum)
        ysum = scipy.signal.savgol_filter(ysum, 15, 3)
        xpeaks = peakutils.indexes(xsum, thres=thres, min_dist=min_dist)
        ypeaks = peakutils.indexes(ysum, thres=thres, min_dist=min_dist)

        if len(xpeaks) == expected_xpeaks and len(ypeaks) == expected_ypeaks:
            try:  # Further optimize peakfinding by interpolating
                xpeaks = peakutils.interpolate(range(0, self.xlen), xsum, ind=xpeaks)
                ypeaks = peakutils.interpolate(range(0, self.ylen), ysum, ind=ypeaks)
            except RuntimeError as err:
                if "Optimal parameters not found: Number of calls to function has reached" in str(err):
                    print "WARNING: peakutils.interpolate broke, using integer values"
                else:
                    raise

            if self.name:  # Only one cluster in simulation box
                center = xpeaks[0], ypeaks[0]
                print "Success: found 1 xpeak, and 1 ypeak!"
                print "  {0}:  (x, y) = {1}".format(self.name, center)
                plot.psmac_xrays_with_dmrho_peakfind(
                    self, snapnr, xsum, ysum, xpeaks, ypeaks, numpy.nan)
                return center
            else:
                if self.toy.parms["ImpactParam"] < 0.1:  # two clusters, no impact parameter
                    cygA = xpeaks[0], ypeaks[0]
                    cygNW = xpeaks[1], ypeaks[0]
                else:  # two clusters, yes impact parameter.
                    #TODO: check which ypeak belongs to which cluster
                    cygA = xpeaks[0], ypeaks[0]
                    cygNW = xpeaks[1], ypeaks[1]

                distance = numpy.sqrt(p2(cygA[0]-cygNW[0])+p2(cygA[1]-cygNW[1]))
                distance *= self.pixelscale
                print "Success: found {0} xpeaks, and {1} ypeak!"\
                    .format(expected_xpeaks, expected_ypeaks)
                print "  cygA:  (x, y) = {0}".format(cygA)
                print "  cygNW: (x, y) = {0}".format(cygNW)
                print "  distance      = {0:.2f}\n".format(distance)
                plot.psmac_xrays_with_dmrho_peakfind(
                    self, snapnr, xsum, ysum, xpeaks, ypeaks, distance)
                return cygA, cygNW, distance
        else:
            print "ERROR: found incorrect number of peaks in dmrho"
            print "xpeaks = {0}\nypeaks = {1}".format(xpeaks, ypeaks)
            print "Snapshot number = {0}\n".format(snapnr)
            plot.psmac_xrays_with_dmrho_peakfind(
                self, snapnr, xsum, ysum, xpeaks, ypeaks, numpy.nan)
