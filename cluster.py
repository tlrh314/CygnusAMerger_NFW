import scipy
import re
import glob
import numpy
import astropy
from matplotlib import pyplot

from cosmology import CosmologyCalculator
import parse
import convert
import profiles
import fit
from macro import *


# ----------------------------------------------------------------------------
# Class to hold Chandra observation
# ----------------------------------------------------------------------------
class ObservedCluster(object):
    """ Parse and store Chandra XVP (PI Wise) observation """
    def __init__(self, basedir, name, cNFW=None, bf=0.17, RCUT_R200_RATIO=None,
                 verbose=True, debug=False):
        """ Read in the quiescent radial profiles of CygA/CygNW afer 1.03 Msec
            Chandra XVP observations (PI Wise). Data courtesy of M.N. de Vries.
            Files are copied over from Struis account martyndv.

            Available profiles: density, metallicity, pressure, temperature, Y
            Both average sector, and hot/cold/merger sectors available """

        if name != "cygA" and name != "cygNW":
            print "ERROR: incorrect ObservedCluster name specified: '{0}'".format(name)
            return
        self.basedir = basedir
        self.name = name

        # Redshift of Cygnus cluster Owen+ 1997. CygNW might have different z.
        # We adopt concordance cosmology with generic cosmological parameters
        self.cc = CosmologyCalculator(z=0.0562, H0=70, WM=0.3, WV=0.7)

        self.avg = parse.chandra_quiescent(self.basedir, self.name)
        self.set_radius(self.avg)
        self.set_massdensity(self.avg)
        self.set_temperature_kelvin(self.avg)

        """ Spectral fitting broke for the last two bins, so we mask those.
            Otherwise for plotting we want to show all other bins, but for
            our fits we ignore the central (AGN) dominated emission. """
        if self.name == "cygA":  # no have sectoranalysis for CygNW
            self.avg_for_plotting = self.mask_bins(self.avg, first=2, last=2)
            self.avg = self.mask_bins(self.avg, first=5, last=4)
            self.merger, self.hot, self.cold = parse.chandra_sectors(self.basedir)
            self.set_radius(self.merger)
            self.set_radius(self.hot)
            self.set_radius(self.cold)
            self.set_massdensity(self.merger)
            self.set_massdensity(self.hot)
            self.set_massdensity(self.cold)
            # sector analysis fit broke for last two bins
            self.merger_for_plotting = self.mask_bins(self.merger, first=0, last=2)
            self.merger = self.mask_bins(self.merger, first=2, last=4)
            self.hot_for_plotting = self.mask_bins(self.hot, first=1, last=2)
            self.hot = self.mask_bins(self.hot, first=6, last=4)
            self.cold_for_plotting = self.mask_bins(self.cold, first=0, last=2)
            self.cold = self.mask_bins(self.cold, first=0, last=4)
        if self.name == "cygNW":
            self.avg_for_plotting = self.mask_bins(self.avg, first=0, last=1)
            self.avg = self.mask_bins(self.avg, first=0, last=2)

        self.ana_radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(1e5), 200))

        self.set_bestfit_betamodel(verbose=verbose)

        # M_HE(<r) from ne_obs and T_obs alone
        self.infer_hydrostatic_mass()

        # M(<r) under assumption DM follows NFW
        self.infer_NFW_mass(cNFW=cNFW, bf=bf, RCUT_R200_RATIO=RCUT_R200_RATIO,
                            verbose=verbose, debug=debug)

        # Set callable gas/dm density/mass profiles, and total mass profile
        self.set_inferred_profiles()

        # T(r) from hydrostatic equilibrium by plugging in rho_gas, M(<r)
        self.set_inferred_temperature(verbose=verbose, debug=debug)

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

    def set_temperature_kelvin(self, t):
        t["T"] = convert.keV_to_K(t["kT"])
        t["fT"] = convert.keV_to_K(t["fkT"])

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

    def infer_hydrostatic_mass(self):
        """ From Chandra density and temperature we infer total gravitating mass
            under the assumption that hydrostatic equilibrium holds.
            This does not make assumptions about the shape of the dark matter
            and this does not assume any temperature profile """
        # Hydrostatic mass equation eats cgs: feed the monster radii in cgs
        mask = numpy.where(self.ana_radii < 1000)  # Take analytical radii up to 1 Mpc
        self.HE_radii = self.ana_radii[mask]*convert.kpc2cm

        # Betamodel /w number density and its derivative
        self.HE_ne = profiles.gas_density_betamodel(
            self.HE_radii, self.ne0, self.beta, self.rc*convert.kpc2cm)
        self.HE_dne_dr = profiles.d_gas_density_betamodel_dr(
            self.HE_radii, self.ne0, self.beta, self.rc*convert.kpc2cm)

        # Only use unmasked values b/c splrep/splev breaks for masked values
        r = numpy.ma.compressed(self.avg["r"]*convert.kpc2cm)
        T = numpy.ma.compressed(self.avg["T"])

        # Fit a smoothed cubic spline to the data. Spline then gives dkT/dr
        if self.name == "cygA":
            T = scipy.ndimage.filters.gaussian_filter1d(T, 25)  # sigma=25
        elif self.name == "cygNW":
            T = scipy.ndimage.filters.gaussian_filter1d(T, 7)  # sigma=7
        self.T_spline = scipy.interpolate.splrep(r, T)  # built-in smoothing breaks

        # Evaluate spline, der=0 for fit to the data and der=1 for first derivative.
        self.HE_T = scipy.interpolate.splev(self.HE_radii, self.T_spline, der=0)
        self.HE_dT_dr = scipy.interpolate.splev(self.HE_radii, self.T_spline, der=1)

        self.HE_M_below_r = profiles.smith_hydrostatic_mass(
            self.HE_radii, self.HE_ne, self.HE_dne_dr, self.HE_T, self.HE_dT_dr)

    def infer_NFW_mass(self, cNFW=None, bf=0.17, RCUT_R200_RATIO=None,
                       verbose=False, debug=False):
        self.halo = fit.total_gravitating_mass(self, cNFW=cNFW, bf=bf,
            RCUT_R200_RATIO=RCUT_R200_RATIO, verbose=verbose, debug=debug)
        self.rcut_kpc = self.halo["rcut"]
        self.rcut_cm = self.halo["rcut"]*convert.kpc2cm if self.halo["rcut"] is not None else None

    def set_inferred_profiles(self):
        # We need callable gas profile, and a callable total mass profile
        rho0_gas = self.rho0
        rho0_dm = self.halo["rho0_dm"]

        self.rho_gas = lambda r: profiles.gas_density_betamodel(r, rho0_gas,
            self.beta, self.rc*convert.kpc2cm, rcut=self.rcut_cm)
        self.rho_dm = lambda r: profiles.dm_density_nfw(r, rho0_dm,
            self.halo["rs"]*convert.kpc2cm, rcut=self.rcut_cm)
        self.M_gas = lambda r: profiles.gas_mass_betamodel(r, rho0_gas,
            self.beta, self.rc*convert.kpc2cm, rcut=self.rcut_cm)
        self.M_dm = lambda r: profiles.dm_mass_nfw(r, rho0_dm,
            self.halo["rs"]*convert.kpc2cm, rcut=self.rcut_cm)

        self.M_tot = lambda r: (self.M_gas(r) + self.M_dm(r))

    def set_inferred_temperature(self, verbose=False, debug=False):
        """ Assume NFW for DM. Get temperature from hydrostatic equation by
            plugging in best-fit betamodel and the inferred best-fit total
            gravitating mass that retrieves the observed temperature. """
        print "Setting hydrostatic temperature"

        radii = self.ana_radii  # self.avg["r"]
        N = len(radii)
        hydrostatic = numpy.zeros(N)
        hydrostatic_pressure = numpy.zeros(N)  # ideal gas

        # R_sample = numpy.sqrt(3)/2*numpy.floor(2*self.halo["r200"])
        infinity = 1e25
        for i, r in enumerate(radii * convert.kpc2cm):
            if not r: continue  # to skip masked values

            hydrostatic[i] = profiles.hydrostatic_temperature(
                r, infinity, self.rho_gas, self.M_tot)
            hydrostatic_pressure[i] = profiles.hydrostatic_gas_pressure(
                r, infinity, self.rho_gas, self.M_tot)

            if verbose and (i == (N-1) or i%10 == 0):
                print_progressbar(i, N)
        print "\n"

        self.hydrostatic = convert.K_to_keV(hydrostatic)
        self.hydrostatic_pressure = hydrostatic_pressure

    def plot_chandra_average(self, parm="kT", style=dict()):
        """ plot of observed average profile of parm """
        # barsabove=True because otherwise NaN values raise ValueError
        pyplot.errorbar(self.avg_for_plotting["r"], self.avg_for_plotting[parm],
                        xerr=self.avg_for_plotting["fr"]/2,
                        yerr=[self.avg_for_plotting["f"+parm],
                              self.avg_for_plotting["f"+parm]],
                        barsabove=True, **style)

    def plot_chandra_sector(self, parm="kT", merger=False, hot=False, cold=False,
                            style=dict()):
        if self.name != "cygA":
            print "ERROR: Sectoranalysis not available for", self.name
            return
        if merger:
            pyplot.errorbar(self.merger_for_plotting["r"], self.merger_for_plotting[parm],
                            xerr=self.merger_for_plotting["fr"]/2,
                            yerr=[self.merger_for_plotting["f"+parm],
                                  self.merger_for_plotting["f"+parm]], **style)
        if hot:
            pyplot.errorbar(self.hot_for_plotting["r"], self.hot_for_plotting[parm],
                            xerr=self.hot_for_plotting["fr"]/2,
                            yerr=[self.hot_for_plotting["f"+parm],
                                  self.hot_for_plotting["f"+parm]], **style)
        if cold:
            pyplot.errorbar(self.cold_for_plotting["r"], self.cold_for_plotting[parm],
                            xerr=self.cold_for_plotting["fr"]/2,
                            yerr=[self.cold_for_plotting["f"+parm],
                                  self.cold_for_plotting["f"+parm]], **style)

    def plot_bestfit_betamodel(self, style=dict(), rho=True):
        fit = profiles.gas_density_betamodel(self.ana_radii,
                self.rho0 if rho else self.ne0, self.beta, self.rc, self.rcut_kpc)

        if "label" not in style:
            label = r"\begin{tabular}{p{2.5cm}ll}"
            # label += " model & = & free beta \\\\"
            if rho:
                label += r" rho0 & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.rho0)
            else:
                label += r" ne0 & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.ne0)
            label += " beta & = & {0:.3f} \\\\".format(self.beta)
            label += " rc & = & {0:.2f} kpc \\\\".format(self.rc)
            label += (" \hline \end{tabular}")
        pyplot.plot(self.ana_radii, fit, **style)

        ymin = profiles.gas_density_betamodel(
            self.rc, self.rho0 if rho else self.ne0, self.beta, self.rc)
        pyplot.vlines(x=self.rc, ymin=ymin, ymax=1e-10 if rho else 9.15,
                      **{ k: style[k] for k in style.keys() if k != "label" })
        pyplot.text(self.rc-25 if self.name == "cygNW" else self.rc-1,
            3e-23 if rho else 4.06, r"$r_c$", ha="right", fontsize=22)

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
        density = self.halo["rho0_dm"] if rho else self.halo["ne0_dm"]
        rho_dm = profiles.dm_density_nfw(self.ana_radii, density, rs, rcut=self.rcut_kpc)

        if "label" not in style:
            label = r"\begin{tabular}{p{2.5cm}ll}"
            # label += " model & = & NFW \\\\"
            label += r" rho0dm & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.halo["rho0_dm"])
            label += " rs & = & {0:.2f} kpc \\\\".format(rs)
            label += (" \hline \end{tabular}")
        pyplot.plot(self.ana_radii, rho_dm, **style)

        ymin = profiles.dm_density_nfw(rs, density, rs)
        pyplot.vlines(x=rs, ymin=ymin, ymax=1e-10 if rho else 9.15,
                      **{ k: style[k] for k in style.keys() if k != "label" })
        pyplot.text(rs-25, 3e-23 if rho else 4.06, r"$r_s$", ha="right", fontsize=22)

    def plot_bestfit_betamodel_mass(self, style=dict()):
        mass = convert.g2msun*self.M_gas(self.ana_radii*convert.kpc2cm)
        pyplot.plot(self.ana_radii, mass , **style)

    def plot_inferred_nfw_mass(self, style=dict()):
        mass = convert.g2msun*self.M_dm(self.ana_radii*convert.kpc2cm)
        pyplot.plot(self.ana_radii, mass, **style)

    def plot_inferred_total_gravitating_mass(self, style=dict()):
        mass = convert.g2msun*self.M_tot(self.ana_radii*convert.kpc2cm)
        pyplot.plot(self.ana_radii, mass, **style)

    def plot_hydrostatic_mass(self, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "HE"
        style["color"] = "b"
        pyplot.plot(self.HE_radii*convert.cm2kpc,
            self.HE_M_below_r*convert.g2msun, **style)

    def plot_verlinde(self, ax1, ax2, ax3, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "Verlinde"
        style["color"] = "r"

        rho_gas = lambda r: profiles.gas_density_betamodel(r*convert.kpc, self.rho0,
            self.beta, self.rc*convert.kpc2cm)
        M_gas = lambda r: profiles.gas_mass_betamodel(r*convert.kpc, self.rho0,
            self.beta, self.rc*convert.kpc2cm)
        M_verlinde = lambda r: profiles.verlinde_apparent_DM_mass(
            r*convert.kpc, self.rho0, self.beta, self.rc*convert.kpc2cm)
        M_tot = lambda r: (M_gas(r) + M_verlinde(r))

        radii = self.ana_radii  # if not fit else self.avg["r"]
        N = len(radii)
        mass = numpy.zeros(N)
        temperature = numpy.zeros(N)
        pressure = numpy.zeros(N)

        infinity = 1e25
        for i, r in enumerate(self.ana_radii*convert.kpc2cm):
            if not r: continue  # to skip masked values

            mass[i] = M_verlinde(r)
            temperature[i] = profiles.hydrostatic_temperature(
                r, infinity, rho_gas, M_tot)
            pressure[i] = profiles.hydrostatic_gas_pressure(
                r, infinity, rho_gas, M_tot)

        ax1.plot(radii, mass*convert.g2msun, **style)
        ax2.plot(radii, convert.K_to_keV(temperature), **style)
        ax3.plot(radii, pressure, **style)

    def plot_verlinde_pressure(self, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "Verlinde"
        style["color"] = "r"

    def plot_inferred_temperature(self, style=dict()):
        radii = self.ana_radii
        pyplot.plot(radii, self.hydrostatic,
            label="cNFW={0:.3f}, bf={1:.4f}".format(
            self.halo["cNFW"], self.halo["bf200"]),
            **{ k: style[k] for k in style.keys() if k != "label" })

    def plot_inferred_pressure(self, style=dict(), do_cut=False):
        pyplot.plot(self.ana_radii, self.hydrostatic_pressure, **style)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Class to hold Toycluster sampled clusters
# ----------------------------------------------------------------------------
class Toycluster(object):
    """ Parse and store Toycluster sampled cluster """
    def __init__(self, icdir, both=False, verbose=True):
        """ Class to hold Toycluster simulation output
        @param icdir: path to the directory with Toycluster output, string
        @return     : instance of Toycluster class"""

        self.profiles = dict()
        for filename in glob.glob(icdir+"profiles_*.txt"):
            halonumber = re.search("(?!(.+)(profiles_))(\d{3})", filename).group()
            self.profiles[halonumber] = parse.toycluster_profiles(filename)
        self.header, self.gas, self.dm = parse.toycluster_icfile(icdir+"IC_single_0")
        self.parms = parse.read_toycluster_parameterfile(glob.glob(icdir+"*.par")[0])
        self.makefile_options = parse.read_toycluster_makefile(glob.glob(icdir+"Makefile_Toycluster")[0])
        for k, v in self.makefile_options.iteritems():
            if "-DRCUT_R200_RATIO=" in v:
                self.RCUT_R200_RATIO = float(v.split("-DRCUT_R200_RATIO=")[1])

        self.r_sample = self.header["boxSize"]/2

        self.gas["rho"] = convert.toycluster_units_to_cgs(self.gas["rho"])
        self.gas["rhom"] = convert.toycluster_units_to_cgs(self.gas["rhom"])
        self.gas["kT"] = convert.K_to_keV(convert.gadget_u_to_t(self.gas["u"]))
        if not both:
            self.set_gas_mass()
            self.set_gas_pressure()
            self.M_dm_tot = self.header["ndm"] * self.header["massarr"][1] * 1e10
            self.M_gas_tot = self.header["ngas"] * self.header["massarr"][0] * 1e10
            self.set_dm_mass()
            self.set_dm_density()
        else:
            print "TODO: implement eating two clusters in box"

    def __str__(self):
        tmp = "Toycluster ICfile header:\n"
        for k, v in self.header.iteritems(): tmp += "    {0:<25}: {1}\n".format(k, v)
        return tmp

    def set_gas_mass(self, NGB=295):
        """ Set the gas mass from the SPH density, see Price (2012, eq. 11)
            Mtot = 4/3 pi R_kern^3 rho, where R_kern^3 = hsml^3/NGB.
            Toycluster: Wendland C6, NGB=295; Gadget-2: M4, NGB=50.

            @param DESNNGB: 50 for Gadget-2 B-spline, 295 for toycluster WC6"""

        self.gas.sort("r")
        rho = convert.density_cgs_to_msunkpc(self.gas["rho"])
        self.gas["mass"] = (4./3*numpy.pi*(p3(self.gas["hsml"])/NGB)*rho).cumsum()

    def set_dm_mass(self, verbose=True):
        """ Count particles <r (= number density). Obtain DM mass from it """

        if verbose: print "    Counting nr. of particles with radius < r to obtain M(<r)"

        radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(1e5), 1001))
        dr = radii[1:] - radii[:-1]
        self.dm_radii = radii[:-1]
        N = len(self.dm_radii)

        particles = numpy.zeros(N)
        gas_particles = numpy.zeros(N)
        for i, r in enumerate(self.dm_radii):
            particles[i] = ((numpy.where(self.dm["r"] < r)[0]).size)
            gas_particles[i] = ((numpy.where(self.gas["r"] < r)[0]).size)
            if verbose and (i==(N-1) or i%100 == 0):
                print_progressbar(i, N, whitespace="    ")

        particles_in_shell = numpy.zeros(len(particles))
        gas_particles_in_shell = numpy.zeros(len(gas_particles))
        for i in range(1, len(particles)):
            particles_in_shell[i-1] = particles[i] - particles[i-1]
            gas_particles_in_shell[i-1] = gas_particles[i] - gas_particles[i-1]

        self.dm_volume = 4 * numpy.pi * self.dm_radii**2 * dr
        self.n_dm_in_shell = particles_in_shell
        self.n_gas_in_shell = gas_particles_in_shell
        self.M_dm_below_r = particles * self.M_dm_tot/self.header["ndm"]
        self.M_gas_below_r = gas_particles * self.M_gas_tot/self.header["ngas"]

    def set_dm_density(self):
        self.rho_dm_below_r = (self.M_dm_tot*convert.msun2g
                * (self.n_dm_in_shell/self.header["ndm"])
                / (self.dm_volume * p3(convert.kpc2cm)))

    def set_gas_pressure(self):
        self.gas["P"] = convert.rho_to_ne(self.gas["rho"]) *\
            convert.keV_to_erg(self.gas["kT"])

# ----------------------------------------------------------------------------
# Class to hold Gadget-2 simulation snaphots
# ----------------------------------------------------------------------------
class Gadget2Output(object):  # TODO parse individual snapshots, split box in half, etc
    """ Parse and store Gadget-2 simulation snapshots"""
    def __init__(self, simdir, verbose=True):
        """ Class to hold Gadgget-2 simulation output
        @param simdir: path to the directory with Gadget-2 output, string
        @return      : instance of Gadget2Output class"""
        self.parms = parse.read_gadget2_parms(simdir+"gadget2.par")

    def __str__(self):
        tmp = "Gadget-2 parameters:\n"
        for k, v in self.parms.iteritems(): tmp += "    {0:<25}: {1}\n".format(k, v)
        return tmp


# ----------------------------------------------------------------------------
# Class to hold Gadget-3 simulation snaphots
# ----------------------------------------------------------------------------
class Gadget2Output(object):  # TODO parse individual snapshots, split box in half, etc
    """ Parse and store Gadget-3 simulation snapshots"""
    def __init__(self, simdir, verbose=True):
        """ Class to hold Gadgget-2 simulation output
        @param simdir: path to the directory with Gadget-3 output, string
        @return      : instance of Gadget2Output class"""
        self.parms = parse.read_gadget3_parms(simdir+"gadget3.par")

    def __str__(self):
        tmp = "Gadget-3 parameters:\n"
        for k, v in self.parms.iteritems(): tmp += "    {0:<25}: {1}\n".format(k, v)
        return tmp


# ----------------------------------------------------------------------------
# Class to hold P-Smac2 simulation snaphots
# ----------------------------------------------------------------------------
class PSmac2Output(object):
    """ Parse and store Gadget-2 simulation snapshots"""
    def __init__(self, sim, verbose=True):
        """ Class to hold Gadgget-2 simulation output
        @param analysisdir: path to the directory with P-Smac2 output, string
        @return           : instance of PSmac2Output class"""

        self.eat_all_fitsfiles(sim)

    def __str__(self, debug=False):
        available = self.available_smac_cubes()
        tmp = "P-Smac2 fits cubes available:\n"
        tmp += "    {0}\n".format(available)
        if not debug: return tmp
        for avail in available:
            tmp += "\n    Header of attribute: '{0}'\n".format(avail)
            for k, v in getattr(self, avail+"_header").iteritems():
                tmp += "    {0:<25}: {1}\n".format(k, v)
        return tmp

    def eat_all_fitsfiles(self, sim):
        # Set attribute depending on the name of the fitsfile
        attributes = {
            "physical-density": "rhogas", "dm-density": "rhodm",
            "temperature-spectroscopic": "tspec",
            "temperature-emission-weighted": "tem",
            "xray-surface-brightness": "xray",
            "velocity": "vel"
        }

        smaccubes = glob.glob(sim.analysisdir+"*.fits.fz")
        for path in smaccubes:
            for cubename, attr in attributes.iteritems():
                if cubename in path:
                    break
            else:
                print "ERROR: unknown fits filename '{0}'".format(path)
                continue
            header, data = parse.psmac2_fitsfile(path)
            setattr(self, attr+"_header", header)
            setattr(self, attr, data)

    def available_smac_cubes(self):
        return [i for i in self.__dict__.keys() if i[:1] != "_" and "_header" not in i]
