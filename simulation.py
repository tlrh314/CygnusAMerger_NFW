import os
import parse

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
    #if "pixelscale" not in dir(self):
    #    self.nsnaps, self.xlen, self.ylen = getattr(self, attr+"data").shape
    #    self.pixelscale = float(self.scale)/int(self.xlen)
