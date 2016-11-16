import numpy
import astropy
from matplotlib import pyplot

from cosmology import CosmologyCalculator
import ioparser
import convert
import profiles
import fit


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
        if self.name == "cygA":  # no have sectoranalysis for CygNW
            self.avg = self.mask_bins(self.avg, first=5, last=3)  # or 2 2
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
        if self.name == "cygNW":
            self.avg = self.mask_bins(self.avg, first=0, last=1)

        self.set_bestfit_betamodel()


    def __str__(self):
        return str(self.avg)

    def set_radius(self, t):
        # An adaptive binning routine is used for data extraction to ensure
        # SNR==100. Therefore binsizes vary, but error bars are equal.
        arcsec2kpc = self.cc.kpc_DA  # kpc
        t["r"] = (t["Radius1"] + t["Radius2"])/2 * arcsec2kpc  # radius
        t["fr"] = (t["Radius2"] - t["Radius1"]) * arcsec2kpc   # binsize

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

    def set_bestfit_betamodel(self):
        self.mles, self.fmles = fit.betamodel_to_chandra(self, verbose=True)

    def plot_chandra_average(self, parm="kT", alpha=0.2):
        """ plot of observed average profile of parm """
        pyplot.errorbar(self.avg["r"], self.avg[parm], xerr=self.avg["fr"]/2,
                        yerr=[self.avg["f"+parm], self.avg["f"+parm]], marker="o",
                        ls="", c="g" if self.name == "cygA" else "b", ms=4, alpha=alpha,
                        elinewidth=2, label="1.03 Msec Chandra\n(Wise+ in prep)")

    def plot_chandra_sector(self, parm="kT", merger=True, hot=True, cold=True, alpha=0.2):
        if self.name != "cygA":
            print "ERROR: Sectoranalysis not available for", self.name
            return
        if merger:
            pyplot.errorbar(self.merger["r"], self.merger[parm],
                            xerr=self.merger["fr"]/2,
                            yerr=[self.merger["f"+parm], self.merger["f"+parm]],
                            marker="o", ls="", c="g", ms=4, alpha=alpha,
                            elinewidth=2, label="Merger "+parm)
        if hot:
            pyplot.errorbar(self.hot["r"], self.hot[parm],
                            xerr=self.hot["fr"]/2,
                            yerr=[self.hot["f"+parm], self.hot["f"+parm]],
                            marker="o", ls="", c="r", ms=6, alpha=alpha,
                            elinewidth=2, label="Hot "+parm)
        if cold:
            pyplot.errorbar(self.cold["r"], self.cold[parm],
                            xerr=self.cold["fr"]/2,
                            yerr=[self.cold["f"+parm], self.cold["f"+parm]],
                            marker="o", ls="", c="purple", ms=6, alpha=alpha,
                            elinewidth=2, label="Cold "+parm)

    def plot_bestfit_betamodel(self):
        radii = numpy.arange(0, 1.1e3, 0.1)  # kpc
        fit = profiles.gas_density_betamodel(radii, self.mles[0],
                self.mles[1], self.mles[2], numpy.nan, do_cut=False)

        label = r"\begin{tabular}{lll}"
        label += " model & = & free beta \\\\"
        label += r" rho0 & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.mles[0])
        label += " beta & = & {0:.3f} kpc \\\\".format(self.mles[1])
        label += " rc & = & {0:.2f} kpc \\\\".format(self.mles[2])
        label += (" \end{tabular}")
        pyplot.plot(radii, fit, c="k", lw=4, label=label)
        pyplot.axvline(x=self.mles[2], lw=3, ls="dashed", c="k")

    def plot_bestfit_residuals(self):
        fit = profiles.gas_density_betamodel(self.avg["r"], self.mles[0],
                self.mles[1], self.mles[2], numpy.nan, do_cut=False)

        residuals = (self.avg["n"] - fit)/self.avg["n"]

        pyplot.errorbar(self.avg["r"], 100*residuals,
                        yerr=100*self.avg["fn"]/self.avg["n"],
                        ls="", c="k", lw=3, elinewidth=1)
        pyplot.errorbar(self.avg["r"]-self.avg["fr"]/2, 100*residuals, c="k",
                        lw=3, elinewidth=1, drawstyle="steps-post")
        pyplot.axvline(x=self.mles[2], lw=3, ls="dashed", c="k")
# ----------------------------------------------------------------------------
