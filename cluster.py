import numpy
import astropy
from matplotlib import pyplot

from cosmology import CosmologyCalculator
import parse
import convert
import profiles
import fit


# ----------------------------------------------------------------------------
# Class to hold Chandra observation
# ----------------------------------------------------------------------------
class ObservedCluster(object):
    """ Parse and store Chandra XVP (PI Wise) observation """
    def __init__(self, name, verbose=True):
        """ Read in the quiescent radial profiles of CygA/CygNW afer 1.03 Msec
            Chandra XVP observations (PI Wise). Data courtesy of M.N. de Vries.
            Files are copied over from Struis account martyndv.

            Available profiles: density, metallicity, pressure, temperature, Y
            Both average sector, and hot/cold/merger sectors available """

        self.name = name

        # Redshift of Cygnus cluster Owen+ 1997. CygNW might have different z.
        # We adopt concordance cosmology with generic cosmological parameters
        self.cc = CosmologyCalculator(z=0.0562, H0=70, WM=0.3, WV=0.7)

        self.avg = parse.chandra_quiescent(self.name)
        self.set_radius(self.avg)
        self.set_massdensity(self.avg)
        if self.name == "cygA":  # no have sectoranalysis for CygNW
            self.avg = self.mask_bins(self.avg, first=5, last=3)  # or 2 2
            self.merger, self.hot, self.cold = parse.chandra_sectors()
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

        self.set_bestfit_betamodel(verbose=verbose)
        self.set_total_gravitating_mass(verbose=verbose)

    def __str__(self):
        return str(self.avg)

    def set_radius(self, t):
        """ An adaptive binning routine is used for data extraction to ensure
        SNR==100. Therefore binsizes vary, but error bars are equal. """
        arcsec2kpc = self.cc.kpc_DA  # kpc
        t["r"] = (t["Radius1"] + t["Radius2"])/2 * arcsec2kpc  # radius
        t["fr"] = (t["Radius2"] - t["Radius1"]) * arcsec2kpc   # binsize

    def set_massdensity(self, t):
        """ Set mass density from number density """
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

    def set_bestfit_betamodel(self, verbose=False):
        mles, fmles = fit.betamodel_to_chandra(self, verbose=verbose)
        self.ne0 = mles[0]
        self.rho0 = convert.ne_to_rho(self.ne0)
        self.beta = mles[1]
        self.rc = mles[2]
        self.fne0 = fmles[0]
        self.frho0 = convert.ne_to_rho(self.fne0)
        self.fbeta = fmles[1]
        self.frc = fmles[2]

    def set_total_gravitating_mass(self, verbose=False):
        self.halo = fit.total_gravitating_mass(self, verbose=verbose)

    def plot_chandra_average(self, parm="kT", style=dict()):
        """ plot of observed average profile of parm """
        pyplot.errorbar(self.avg["r"], self.avg[parm], xerr=self.avg["fr"]/2,
                        yerr=[self.avg["f"+parm], self.avg["f"+parm]], **style)

    def plot_chandra_sector(self, parm="kT", merger=False, hot=False, cold=False,
                            style=dict()):
        if self.name != "cygA":
            print "ERROR: Sectoranalysis not available for", self.name
            return
        if merger:
            pyplot.errorbar(self.merger["r"], self.merger[parm],
                            xerr=self.merger["fr"]/2,
                            yerr=[self.merger["f"+parm], self.merger["f"+parm]],
                            **style)
        if hot:
            pyplot.errorbar(self.hot["r"], self.hot[parm],
                            xerr=self.hot["fr"]/2,
                            yerr=[self.hot["f"+parm], self.hot["f"+parm]],
                            **style)
        if cold:
            pyplot.errorbar(self.cold["r"], self.cold[parm],
                            xerr=self.cold["fr"]/2,
                            yerr=[self.cold["f"+parm], self.cold["f"+parm]],
                            **style)

    def plot_bestfit_betamodel(self, style=dict(), rho=True, do_cut=False):
        radii = numpy.arange(0.1, 1.1e4, 0.1)  # kpc
        fit = profiles.gas_density_betamodel(radii, self.rho0 if rho else self.ne0,
                self.beta, self.rc, None if not do_cut else self.halo["r200"],
                do_cut=do_cut)

        label = r"\begin{tabular}{p{2.5cm}ll}"
        # label += " model & = & free beta \\\\"
        if rho:
            label += r" rho0 & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.rho0)
        else:
            label += r" ne0 & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.ne0)
        label += " beta & = & {0:.3f} \\\\".format(self.beta)
        label += " rc & = & {0:.2f} kpc \\\\".format(self.rc)
        label += (" \hline \end{tabular}")
        pyplot.plot(radii, fit, label=label, **style)

        ymin = profiles.gas_density_betamodel(
            self.rc, self.rho0 if rho else self.ne0, self.beta, self.rc)
        pyplot.vlines(x=self.rc, ymin=ymin, ymax=9e-24 if rho else 9.15, **style)
        pyplot.text(self.rc+25 if self.name == "cygNW" else self.rc+1,
            4e-24 if rho else 4.06, r"$r_c$", ha="left", fontsize=22)

    def plot_bestfit_residuals(self, rho=False):
        fit = profiles.gas_density_betamodel(self.avg["r"],
            self.rho0 if rho else self.ne0, self.beta, self.rc)

        residuals = (self.avg["n"] - fit)/self.avg["n"]

        pyplot.errorbar(self.avg["r"], 100*residuals,
                        yerr=100*self.avg["fn"]/self.avg["n"],
                        ls="", c="k", lw=3, elinewidth=1)
        pyplot.errorbar(self.avg["r"]-self.avg["fr"]/2, 100*residuals, c="k",
                        lw=3, elinewidth=1, drawstyle="steps-post")
        pyplot.axvline(x=self.rc, lw=3, ls="dashed", c="k")

    def plot_inferred_nfw_profile(self, style=dict(), rho=True):
        rs = self.halo["rs"]
        radii = numpy.arange(0.1, 1.1e4, 0.1)  # kpc
        density = self.halo["rho0_dm"] if rho else self.halo["ne0_dm"]
        M_dm = profiles.dm_density_nfw(radii, density, rs)

        label = r"\begin{tabular}{p{2.5cm}ll}"
        # label += " model & = & NFW \\\\"
        label += r" rho0dm & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.halo["rho0_dm"])
        label += " rs & = & {0:.2f} kpc \\\\".format(rs)
        label += (" \hline \end{tabular}")
        pyplot.plot(radii, M_dm, label=label, **style)

        ymin = profiles.dm_density_nfw(rs, density, rs)
        pyplot.vlines(x=rs, ymin=ymin, ymax=9e-24 if rho else 9.15, **style)
        pyplot.text(rs-25, 4e-24 if rho else 4.06, r"$r_s$", ha="right", fontsize=22)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Class to hold Toycluster sampled clusters
# ----------------------------------------------------------------------------
class ToyCluster(object):
    """ Parse and store Toycluster sampled cluster """
    def __init__(self, name, verbose=True):
        """ TODO """

        self.name = name

        self.profiles = parse.toycluster_profiles("data/profiles_000.txt")
        self.header, self.gas, self.dm = parse.toycluster_icfile("IC_single_0")

        self.gas["rho"] = convert.toycluster_units_to_cgs(self.gas["rho"])
        self.gas["rhom"] = convert.toycluster_units_to_cgs(self.gas["rhom"])
