import astropy
from matplotlib import pyplot

import ioparser
import convert
from cosmology import CosmologyCalculator


# ----------------------------------------------------------------------------
# Class to hold Chandra observation
# ----------------------------------------------------------------------------
class ObservedCluster(object):
    """ Parse and store Chandra XVP (PI Wise) observation """
    def __init__(self, name):
        """ Read in the quiescent radial profiles of CygA/CygNW afer 1.03 Msec
            Chandra XVP observations (PI Wise). Data courtesy of M.N. de Vries.
            Files are copied over from Struis account martyndv.

            Available profiles: density, metallicity, pressure, temperature, Y
            Both average sector, and hot/cold/merger sectors available
        """

        self.name = name

        # Redshift of Cygnus cluster Owen+ 1997. CygNW might have different z.
        # We adopt concordance cosmology with generic cosmological parameters
        self.cc = CosmologyCalculator(z=0.0562, H0=70, WM=0.3, WV=0.7)

        self.avg = ioparser.parse_chandra_quiescent(self.name)
        self.set_radius(self.avg)
        self.set_massdensity(self.avg)
        self.avg = self.mask_bins(self.avg, first=5, last=3)  # or 2 2
        if self.name == "cygA":  # no have sectoranalysis for CygNW
            self.merger, self.hot, self.cold = ioparser.parse_chandra_sectors()
            self.set_radius(self.merger)
            self.set_radius(self.hot)
            self.set_radius(self.cold)
            self.set_massdensity(self.merger)
            self.set_massdensity(self.hot)
            self.set_massdensity(self.cold)
            # sector analysis fit broke for last two bins
            self.merger = self.mask_bins(self.merger, first=2, last=2)
            self.hot = self.mask_bins(self.hot, first=6, last=2)
            self.cold = self.mask_bins(self.cold, first=2, last=2)


    def __str__(self):
        return str(self.avg)

    def set_radius(self, t):
        # An adaptive binning routine is used for data extraction to ensure
        # SNR==100. Therefore binsizes vary, but error bars are equal.
        arcsec2kpc = self.cc.kpc_DA  # kpc
        t["radius"] = (t["Radius1"] + t["Radius2"])/2 * arcsec2kpc
        t["binsize"] = (t["Radius2"] - t["Radius1"]) * arcsec2kpc

    def set_massdensity(self, t):
        # Set mass density from number density
        t["rho"] = convert.ne_to_rho(t["n"])
        t["frho"] = convert.ne_to_rho(t["fn"])

    def mask_bins(self, t, first=0, last=1):
        """ Mask first n bins, default 0 (mask nothing)
            Mask last (n-1) bins, default 1 (mask nothing) """
        t = astropy.table.Table(t, masked=True)
        # discard first six bins: CygA dominated
        t[0:first].mask = [True for i in range(len(t.columns))]
        # discard last two bins: too low SNR
        t[-last:].mask = [True for i in range(len(t.columns))]
        return t


    def plot_chandra_average(self, parm="kT", alpha=0.2):
        """ plot of observed average profile of parm """
        pyplot.errorbar(self.avg["radius"], self.avg[parm],
                        xerr=self.avg["binsize"]/2,
                        yerr=[self.avg["f"+parm], self.avg["f"+parm]],
                        marker="o", ls="", c="b", ms=4,
                        alpha=alpha, elinewidth=2, label="Average "+parm)

    def plot_chandra_sector(self, parm="kT", merger=True, hot=True, cold=True, alpha=0.2):
        if merger:
            pyplot.errorbar(self.merger["radius"], self.merger[parm],
                            xerr=self.merger["binsize"]/2,
                            yerr=[self.merger["f"+parm], self.merger["f"+parm]],
                            marker="o", ls="", c="g", ms=4, alpha=alpha,
                            elinewidth=2, label="Merger "+parm)
        if hot:
            pyplot.errorbar(self.hot["radius"], self.hot[parm],
                            xerr=self.hot["binsize"]/2,
                            yerr=[self.hot["f"+parm], self.hot["f"+parm]],
                            marker="o", ls="", c="r", ms=6, alpha=alpha,
                            elinewidth=2, label="Hot "+parm)
        if cold:
            pyplot.errorbar(self.cold["radius"], self.cold[parm],
                            xerr=self.cold["binsize"]/2,
                            yerr=[self.cold["f"+parm], self.cold["f"+parm]],
                            marker="o", ls="", c="purple", ms=6, alpha=alpha,
                            elinewidth=2, label="Cold "+parm)
# ----------------------------------------------------------------------------
