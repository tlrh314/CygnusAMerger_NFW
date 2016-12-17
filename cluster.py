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
    def __init__(self, name, cNFW=None, bf=0.17, verbose=True, debug=False,
                 do_cut=False):
        """ Read in the quiescent radial profiles of CygA/CygNW afer 1.03 Msec
            Chandra XVP observations (PI Wise). Data courtesy of M.N. de Vries.
            Files are copied over from Struis account martyndv.

            Available profiles: density, metallicity, pressure, temperature, Y
            Both average sector, and hot/cold/merger sectors available """

        if name != "cygA" and name != "cygNW":
            print "ERROR: incorrect ObservedCluster name specified: '{0}'".format(name)
        self.name = name

        # Redshift of Cygnus cluster Owen+ 1997. CygNW might have different z.
        # We adopt concordance cosmology with generic cosmological parameters
        self.cc = CosmologyCalculator(z=0.0562, H0=70, WM=0.3, WV=0.7)

        self.avg = parse.chandra_quiescent(self.name)
        self.set_radius(self.avg)
        self.set_massdensity(self.avg)
        self.set_temperature_kelvin(self.avg)
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

        self.ana_radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(1e5), 200))

        self.set_bestfit_betamodel(verbose=verbose)

        # M_HE(<r) from ne_obs and T_obs alone
        self.infer_hydrostatic_mass()

        # M(<r) under assumption DM follows NFW
        self.infer_NFW_mass(verbose=verbose, cNFW=cNFW, bf=bf)
        # T(r) from hydrostatic equilibrium by plugging in rho_gas, M(<r)
        self.set_inferred_temperature(verbose=verbose, debug=debug, do_cut=do_cut)

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
        T = scipy.ndimage.filters.gaussian_filter1d(T, 10)  # sigma=10
        self.T_spline = scipy.interpolate.splrep(r, T)  # built-in smoothing breaks

        # Evaluate spline, der=0 for fit to the data and der=1 for first derivative.
        self.HE_T = scipy.interpolate.splev(self.HE_radii, self.T_spline, der=0)
        self.HE_dT_dr = scipy.interpolate.splev(self.HE_radii, self.T_spline, der=1)

        self.HE_M_below_r = profiles.smith_hydrostatic_mass(
            self.HE_radii, self.HE_ne, self.HE_dne_dr, self.HE_T, self.HE_dT_dr)

    def infer_NFW_mass(self, cNFW=None, bf=0.17, verbose=False):
        self.halo = fit.total_gravitating_mass(self,
            cNFW=cNFW, bf=bf, verbose=verbose)

    def set_inferred_temperature(self, fit=False, do_cut=False,
                                 verbose=False, debug=False):
        """ Assume NFW for DM. Get temperature from hydrostatic equation by
            plugging in best-fit betamodel and the inferred best-fit total
            gravitating mass that retrieves the observed temperature. """
        print "Setting hydrostatic temperature"

        radii = self.avg["r"] if fit else self.ana_radii
        N = len(radii)
        hydrostatic = numpy.zeros(N)
        hydrostatic_pressure = numpy.zeros(N)  # ideal gas

        # We need callable gas profile, and a callable total mass profile
        rho0_gas = self.rho0
        rho0_dm = self.halo["rho0_dm"]
        if do_cut:
            rho_gas = lambda r: profiles.gas_density_betamodel(r, rho0_gas,
                self.beta, self.rc*convert.kpc2cm,
                rcut=self.halo["r200"]*convert.kpc2cm, do_cut=True)
            rho_dm = lambda r: profiles.dm_density_nfw(r, rho0_dm,
                self.halo["rs"]*convert.kpc2cm, rcut=1e10*convert.kpc2cm, do_cut=True)
            M_gas = lambda r: profiles.gas_mass_betamodel_cut(r, rho0_gas,
                self.beta, self.rc*convert.kpc2cm, self.halo["r200"]*convert.kpc2cm)
            M_dm = lambda r: profiles.dm_mass_nfw_cut(r, rho0_dm,
                self.halo["rs"]*convert.kpc2cm, 1e10*convert.kpc2cm)
        else:
            rho_gas = lambda r: profiles.gas_density_betamodel(r, rho0_gas,
                self.beta, self.rc*convert.kpc2cm)
            rho_dm = lambda r: profiles.dm_density_nfw(r, rho0_dm,
                self.halo["rs"]*convert.kpc2cm)
            M_gas = lambda r: profiles.gas_mass_betamodel(r, rho0_gas,
                self.beta, self.rc*convert.kpc2cm)
            M_dm = lambda r: profiles.dm_mass_nfw(r, rho0_dm,
                self.halo["rs"]*convert.kpc2cm)

        M_tot = lambda r: (M_gas(r) + M_dm(r))

        # R_sample = numpy.sqrt(3)/2*numpy.floor(2*self.halo["r200"])
        Infinity = 1e25
        for i, r in enumerate(radii * convert.kpc2cm):
            if not r: continue  # to skip masked values

            hydrostatic[i] = profiles.hydrostatic_temperature(
                r, Infinity, rho_gas, M_tot)
            hydrostatic_pressure[i] = profiles.hydrostatic_gas_pressure(
                r, Infinity, rho_gas, M_tot)

            if verbose and (i%10 == 0 or i==(N-1)):
                print_progressbar(i, N)
        print "\n"

        self.hydrostatic = convert.K_to_keV(hydrostatic)
        self.hydrostatic_pressure = hydrostatic_pressure

        if debug:
            print "Showing all profiles plugged into hydrostatic equation"

            rhogas_check = numpy.zeros(N)
            rhodm_check = numpy.zeros(N)
            massgas_check = numpy.zeros(N)
            massdm_check = numpy.zeros(N)
            masstot_check = numpy.zeros(N)
            hydrostatic_gas = numpy.zeros(N)
            hydrostatic_dm = numpy.zeros(N)
            hydrostatic_pressure = numpy.zeros(N)

            for i, r in enumerate(radii * convert.kpc2cm):
                if not r: continue  # to skip masked values

                rhogas_check[i] = rho_gas(r)
                rhodm_check[i] = rho_dm(r)
                massgas_check[i] = M_gas(r)
                massdm_check[i] = M_dm(r)
                masstot_check[i] = M_tot(r)
                hydrostatic_gas[i] = profiles.hydrostatic_temperature(
                    r, Infinity, rho_gas, M_gas)
                hydrostatic_dm[i] = profiles.hydrostatic_temperature(
                    r, Infinity, rho_gas, M_dm)
                hydrostatic_pressure[i] = profiles.hydrostatic_gas_pressure(
                    r, Infinity, rho_gas, M_tot)

            fig, ((ax0, ax1), (ax2, ax3)) = pyplot.subplots(2, 2, sharex=True,
                figsize=(18, 16))
            gas = { "color": "k", "lw": 1, "linestyle": "dotted", "label": "gas" }
            dm  = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "dm" }
            tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "tot" }
            hydrostatic_gas = convert.K_to_keV(hydrostatic_gas)
            hydrostatic_dm = convert.K_to_keV(hydrostatic_dm)
            pyplot.sca(ax0)
            self.plot_chandra_average(parm="rho")
            ax0.loglog(radii, rhogas_check, **gas)
            ax0.loglog(radii, rhodm_check, **dm)
            ax1.loglog(radii, convert.g2msun*massgas_check, **gas)
            ax1.loglog(radii, convert.g2msun*massdm_check, **dm)
            ax1.loglog(radii, convert.g2msun*masstot_check, **tot)
            pyplot.sca(ax2)
            self.plot_chandra_average(parm="kT")
            ax2.semilogx(radii, hydrostatic_gas, **gas)
            ax2.semilogx(radii, hydrostatic_dm, **dm)
            ax2.semilogx(radii, self.hydrostatic, **tot)
            ax2.set_ylim(-1, 10)
            pyplot.sca(ax3)
            self.plot_chandra_average(parm="P")
            ax3.loglog(radii, hydrostatic_pressure, **tot)
            for ax in [ax0, ax1, ax2, ax3]:
                ax.set_xlabel("Radius [kpc]")
                # ax.legend()
            ax0.set_ylabel("Density [g/cm$^3$]")
            ax1.set_ylabel("Mass [MSun]")
            ax2.set_ylabel("Temperature [keV]")
            ax3.set_ylabel("Pressure [erg/cm$^3$]")
            pyplot.tight_layout()
            pyplot.savefig(
                "out/{0}_donnert2014figure1_cNFW={1:.3f}_bf={2:.4f}{3}.pdf"
                    .format(self.name,  self.halo["cNFW"], self.halo["bf200"],
                            "_cut" if do_cut else ""), dpi=300)

    def plot_chandra_average(self, parm="kT", style=dict()):
        """ plot of observed average profile of parm """
        # barsabove=True because otherwise NaN values raise ValueError
        pyplot.errorbar(self.avg["r"], self.avg[parm], xerr=self.avg["fr"]/2,
                        yerr=[self.avg["f"+parm], self.avg["f"+parm]],
                        barsabove=True, **style)

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
        fit = profiles.gas_density_betamodel(self.ana_radii,
                self.rho0 if rho else self.ne0,
                self.beta, self.rc, None if not do_cut else self.halo["r200"],
                do_cut=do_cut)

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

    def plot_inferred_nfw_profile(self, style=dict(), rho=True, do_cut=False):
        rs = self.halo["rs"]
        density = self.halo["rho0_dm"] if rho else self.halo["ne0_dm"]
        rho_dm = profiles.dm_density_nfw(self.ana_radii, density, rs,
                rcut=1e10 if do_cut else None, do_cut=do_cut)

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

    def plot_bestfit_betamodel_mass(self, style=dict(), do_cut=False):
        if do_cut:
            mass = numpy.zeros(len(self.ana_radii))
            for i, r in enumerate(self.ana_radii):
                mass[i] = profiles.gas_mass_betamodel_cut(
                    r, convert.density_cgs_to_msunkpc(self.rho0),
                    self.beta, self.rc, self.halo["r200"])
        else:
            mass = profiles.gas_mass_betamodel(self.ana_radii,
                convert.density_cgs_to_msunkpc(self.rho0), self.beta, self.rc)

        pyplot.plot(self.ana_radii, mass, **style)

    def plot_inferred_nfw_mass(self, style=dict(), do_cut=False):
        rs = self.halo["rs"]
        rho0_dm = convert.density_cgs_to_msunkpc(self.halo["rho0_dm"])

        if do_cut:
            mass = profiles.dm_mass_nfw_cut(r, rho0_dm, rs*convert.kpc2cm,
                                            1e10*convert.kpc2cm)
        else:
            mass = profiles.dm_mass_nfw(self.ana_radii, rho0_dm, rs)

        pyplot.plot(self.ana_radii, mass, **style)

    def plot_inferred_total_gravitating_mass(self, style=dict(), do_cut=False):
        rs = self.halo["rs"]
        rho0_dm = convert.density_cgs_to_msunkpc(self.halo["rho0_dm"])
        rho0_gas = convert.density_cgs_to_msunkpc(self.rho0)
        if do_cut:
            dm = lambda r: profiles.dm_mass_nfw_cut(r, rho0_dm,
                rs*convert.kpc2cm, 1e10*convert.kpc2cm)
            gas = lambda r: profiles.gas_mass_betamodel_cut(r, rho0_gas,
                self.beta, self.rc, self.halo["r200"])
            total = lambda r: ( dm(r) + gas(r) )

            mass = numpy.zeros(len(self.ana_radii))
            for i, r in enumerate(self.ana_radii):
                mass[i] = total(r)
        else:
            mass = profiles.dm_mass_nfw(self.ana_radii, rho0_dm, rs) +  \
                profiles.gas_mass_betamodel(self.ana_radii, rho0_gas,
                                            self.beta, self.rc)

        pyplot.plot(self.ana_radii, mass, **style)

    def plot_hydrostatic_mass(self, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "HE"
        style["color"] = "b"
        pyplot.plot(self.HE_radii*convert.cm2kpc,
            self.HE_M_below_r*convert.g2msun, **style)

    def plot_verlinde_apparent_darkmatter_mass(self, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "Verlinde"
        style["color"] = "r"
        verlinde = lambda r: profiles.verlinde_apparent_DM_mass(
            r, self.rho0, self.beta, self.rc*convert.kpc2cm)

        mass = numpy.zeros(len(self.ana_radii))
        for i, r in enumerate(self.ana_radii*convert.kpc2cm):
            mass[i] = verlinde(r)

        pyplot.loglog(self.ana_radii, mass*convert.g2msun, **style)

    def plot_inferred_temperature(self, fit=False, style=dict()):
        if fit:
            radii = self.avg["r"]
        else:
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

        self.r_sample = self.header["boxSize"]/2

        self.gas["rho"] = convert.toycluster_units_to_cgs(self.gas["rho"])
        self.gas["rhom"] = convert.toycluster_units_to_cgs(self.gas["rhom"])
        self.gas["kT"] = convert.K_to_keV(convert.gadget_u_to_t(self.gas["u"]))
        if not both:
            self.set_gas_mass()
            self.M_dm = self.header["ndm"] * self.header["massarr"][1] * 1e10
            self.set_dm_mass()
            self.set_dm_density()
        else:
            print "TODO: implement eating two clusters in box"

    def __str__(self):
        tmp = "Toycluster ICfile header:\n"
        for k, v in self.header.iteritems(): tmp += "    {0:<25}: {1}\n".format(k, v)
        return tmp

    def set_gas_mass(self, NGB=50):
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
        for i, r in enumerate(self.dm_radii):
            particles[i] = ((numpy.where(self.dm["r"] < r)[0]).size)
            if verbose and (i==(N-1) or i%100 == 0):
                print_progressbar(i, N, whitespace="    ")

        particles_in_shell = numpy.zeros(len(particles))
        for i in range(1, len(particles)):
            particles_in_shell[i-1] = particles[i] - particles[i-1]

        self.dm_volume = 4 * numpy.pi * self.dm_radii**2 * dr
        self.n_dm_in_shell = particles_in_shell
        self.M_dm_below_r = particles * self.M_dm/self.header["ndm"]

    def set_dm_density(self):
        self.rho_dm_below_r = (self.M_dm*convert.msun2g
                * (self.n_dm_in_shell/self.header["ndm"])
                / (self.dm_volume * p3(convert.kpc2cm)))


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
