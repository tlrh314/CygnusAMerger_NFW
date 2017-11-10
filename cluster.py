# -*- coding: utf-8 -*-

import os
import re
import glob
import copy

import numpy
import scipy
from scipy.ndimage.filters import gaussian_filter1d
import astropy
import matplotlib
import peakutils
import dill

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
        self.RCUT_R200_RATIO = RCUT_R200_RATIO

        # Redshift of Cygnus cluster Owen+ 1997. CygNW might have different z.
        # We adopt concordance cosmology with generic cosmological parameters
        self.cc = CosmologyCalculator(z=0.0562, H0=70, WM=0.3, WV=0.7)

        self.avg = parse.chandra_quiescent(self.basedir, self.name)
        self.force_symmetrical_error(self.avg)
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
            for t in [self.merger, self.hot, self.cold]:
                self.force_symmetrical_error(t)
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

        self.ana_radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(5e4), 64))

        self.set_bestfit_betamodel(verbose=verbose)

        # M_HE(<r) from ne_obs and T_obs alone
        self.infer_hydrostatic_mass()
        self.hydrostatic_mass_with_error()

        # M(<r) under assumption DM follows NFW
        self.infer_NFW_mass(cNFW=cNFW, bf=bf, RCUT_R200_RATIO=RCUT_R200_RATIO,
                            verbose=verbose, debug=debug)
        if verbose: self.print_halo_properties()

        # Set callable gas/dm density/mass profiles, and total mass profile
        # self.set_inferred_profiles()

        # T(r) from hydrostatic equilibrium by plugging in rho_gas, M(<r)
        self.set_inferred_temperature(verbose=verbose)

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

    def force_symmetrical_error(self, t):
        t["fn"] = 0.5 * ( t["fn_hi"]+t["fn_lo"] )
        t["fP"] =  0.5 * ( t["fP_lo"]+t["fP_hi"] )
        t["fkT"] =  0.5 * ( t["fkT_lo"]+t["fkT_hi"] )

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

    def hydrostatic_mass_with_error(self):
        r = numpy.ma.compressed(self.avg_for_plotting["r"]*convert.kpc2cm)
        T = numpy.ma.compressed(self.avg_for_plotting["T"])
        T_plus = numpy.ma.compressed(self.avg_for_plotting["T"]+self.avg_for_plotting["fT"])
        T_min = numpy.ma.compressed(self.avg_for_plotting["T"]-self.avg_for_plotting["fT"])

        # Fit a smoothed cubic spline to the data. Spline then gives dkT/dr
        T_smooth = gaussian_filter1d(T, 25 if self.name == "cygA" else 7)
        T_smooth_plus = gaussian_filter1d(T_plus, 25 if self.name == "cygA" else 7)
        T_smooth_min = gaussian_filter1d(T_min, 25 if self.name == "cygA" else 7)
        T_spline = scipy.interpolate.splrep(r, T_smooth)
        T_spline_plus = scipy.interpolate.splrep(r, T_smooth_plus)
        T_spline_min = scipy.interpolate.splrep(r, T_smooth_min)

        n = profiles.gas_density_betamodel(r, self.ne0, self.beta, self.rc*convert.kpc2cm)
        dn_dr = profiles.d_gas_density_betamodel_dr(r, self.ne0, self.beta, self.rc*convert.kpc2cm)

        n_plus = numpy.zeros(len(n))
        dn_plus = numpy.zeros(len(n))
        n_min = numpy.zeros(len(n))
        dn_min = numpy.zeros(len(n))
        for i, ri in enumerate(r):
            n_err_plus = n[i]
            n_err_min = n[i]
            dn_err_plus = 0
            dn_err_min = 0
            for ne0 in [self.ne0, self.ne0-self.fne0, self.ne0+self.fne0]:
                for beta in [self.beta, self.beta-self.fbeta, self.beta+self.fbeta]:
                    for rc in [self.rc, self.rc-self.frc, self.rc+self.frc]:
                        rc *= convert.kpc2cm

                        ni = profiles.gas_density_betamodel(ri, ne0, beta, rc)
                        dn_dri = profiles.d_gas_density_betamodel_dr(ri, ne0, beta, rc)

                        if ni > n_err_plus:
                            n_err_plus = ni
                            dn_err_plus = dn_dri
                        if ni < n_err_min:
                            n_err_min = ni
                            dn_err_min = dn_dri
            n_plus[i] = n_err_plus
            dn_plus[i] = dn_err_plus
            n_min[i] = n_err_min
            dn_min[i] = dn_err_min

        # Evaluate spline, der=0 for fit to the data and der=1 for first derivative.
        T = scipy.interpolate.splev(r, T_spline, der=0)
        T_plus = scipy.interpolate.splev(r, T_spline_plus, der=0)
        T_min = scipy.interpolate.splev(r, T_spline_min, der=0)
        dT_dr = scipy.interpolate.splev(r, T_spline, der=1)
        dT_plus = scipy.interpolate.splev(r, T_spline_plus, der=1)
        dT_min = scipy.interpolate.splev(r, T_spline_min, der=1)


        M_below_r = profiles.smith_hydrostatic_mass(r, n, dn_dr, T, dT_dr)
        M_below_r_plus = numpy.zeros(len(n))
        M_below_r_min = numpy.zeros(len(n))
        for i, ri in enumerate(r):
            M_low = M_below_r[i]
            M_high = M_below_r[i]
            for ni, dni in zip([n, n_plus, n_min], [dn_dr, dn_plus, dn_min]):
                for Ti, dTi in zip([T, T_plus, T_min], [dT_dr, dT_plus, dT_min]):
                    M = profiles.smith_hydrostatic_mass(ri, ni[i], dni[i], Ti[i], dTi[i])
                    if M < M_low:
                        M_low = M
                    if M > M_high:
                        M_high = M
            M_below_r_plus[i] = M_high
            M_below_r_min[i] = M_low

        self.avg_for_plotting["M_HE"] = numpy.zeros_like(self.avg_for_plotting["r"])
        self.avg_for_plotting["M_HE_plus"] = numpy.zeros_like(self.avg_for_plotting["r"])
        self.avg_for_plotting["M_HE_min"] = numpy.zeros_like(self.avg_for_plotting["r"])
        numpy.place(self.avg_for_plotting["M_HE"], ~self.avg_for_plotting["r"].mask,
            M_below_r*convert.g2msun)
        numpy.place(self.avg_for_plotting["M_HE_plus"], ~self.avg_for_plotting["r"].mask,
            M_below_r_plus*convert.g2msun)
        numpy.place(self.avg_for_plotting["M_HE_min"], ~self.avg_for_plotting["r"].mask,
                M_below_r_min*convert.g2msun)

    def infer_NFW_mass(self, cNFW=None, bf=0.17, RCUT_R200_RATIO=None,
                       verbose=False, debug=False):
        self.halo = fit.total_gravitating_mass(self, cNFW=cNFW, bf=bf,
            RCUT_R200_RATIO=RCUT_R200_RATIO, verbose=verbose, debug=debug)

        # if halo != Halo[0] in Toycluster then the cutoff is different
        if self.name == "cygA":
            R200_TO_RMAX_RATIO = 3.75
            Boxsize = numpy.floor(2*R200_TO_RMAX_RATIO * self.halo["r200"]);
            self.r_sample_dm = Boxsize/2
        elif self.name == "cygNW":
            self.r_sample_dm = 1.5 * self.halo["r200"]

        self.rcut_kpc = self.halo["rcut"]
        if self.halo["rcut"] is not None:
            self.rcut_cm = self.halo["rcut"]*convert.kpc2cm
            self.rcut_nfw_kpc = self.r_sample_dm
            self.rcut_nfw_cm = self.r_sample_dm*convert.kpc2cm
        else:
            self.rcut_cm = None
            self.rcut_nfw_kpc = None
            self.rcut_nfw_cm = None

        self.halo["r500"], self.halo["M500"] = fit.find_r500(self)

    def print_halo_properties(self):
        if self.RCUT_R200_RATIO is None:
            rcut = None
        else:
            rcut = self.halo["r200"] * self.RCUT_R200_RATIO

        halo = self.halo
        print "  Assuming fixed baryon fraction constrains DM properties:"
        print "    r200                   = {0:3.1f}".format(halo["r200"])
        print "    r500                   = {0:3.1f}".format(halo["r500"])
        if rcut is not None:
            print "    rcut                   = {0:3.1f}".format(halo["rcut"])
        else:
            print "    rcut                   = {0}".format(halo["rcut"])
        print "    rho_avg(r200)/rho_crit = {0:.1f}". format(halo["rho200_over_rhocrit"])
        print "    bf200                  = {0:1.4f}".format(halo["bf200"])
        print "    rho0                   = {0:1.4e}".format(halo["rho0"])
        print "    ne0                    = {0:1.4e}".format(halo["ne0"])
        print "    rc                     = {0:.3f}". format(halo["rc"])
        print "    beta                   = {0:.3f}". format(halo["beta"])
        print "    Mgas200                = {0:1.4e}".format(halo["Mgas200"])
        print "    Mdm200                 = {0:1.4e}".format(halo["Mdm200"])
        print "    M200                   = {0:1.4e}".format(halo["M200"])
        print "    M500                   = {0:1.4e}".format(halo["M500"])
        print "    cNFW                   = {0:1.4f}".format(halo["cNFW"])
        print "    rs                     = {0:3.1f}".format(halo["rs"])
        print "    rho0_dm                = {0:1.4e}".format(halo["rho0_dm"])
        print "    ne0_dm                 = {0:1.4e}".format(halo["ne0_dm"])
        print

    def rho_gas(self, r):
        return profiles.gas_density_betamodel(r, self.rho0, self.beta,
            self.rc*convert.kpc2cm, rcut=self.rcut_cm)

    def rho_dm(self, r):
        return profiles.dm_density_nfw(r, self.halo["rho0_dm"],
            self.halo["rs"]*convert.kpc2cm, rcut=self.rcut_nfw_cm)

    def M_gas(self, r):
        return profiles.gas_mass_betamodel(r, self.rho0,
            self.beta, self.rc*convert.kpc2cm, rcut=self.rcut_cm)

    def M_dm(self, r):
        return profiles.dm_mass_nfw(r, self.halo["rho0_dm"],
        self.halo["rs"]*convert.kpc2cm, rcut=self.rcut_nfw_cm)

    def M_tot(self, r):
        return (self.M_gas(r) + self.M_dm(r))

    def set_inferred_temperature(self, verbose=False):
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

    def set_compton_y(self, verbose=False):
        print "Setting compton-y"
        print "Does not work yet"
        return

        radii = self.ana_radii  # self.avg["r"]
        N = len(radii)
        comptony = numpy.zeros(N)

        # R_sample = numpy.sqrt(3)/2*numpy.floor(2*self.halo["r200"])
        infinity = 1e25
        for i, r in enumerate(radii * convert.kpc2cm):
            if not r: continue  # to skip masked values

            comptony[i] = profiles.compton_y(
                r, infinity, self.rho_gas, self.M_tot)

            if verbose and (i == (N-1) or i%10 == 0):
                print_progressbar(i, N)
        print "\n"

        self.comptony = comptony

    def plot_chandra_average(self, ax, parm="kT", style=dict()):
        """ plot of observed average profile of parm """
        # compressed, to avoid "UserWarning: Warning: converting a masked element to nan"
        ax.errorbar(numpy.ma.compressed(self.avg_for_plotting["r"]),
                    numpy.ma.compressed(self.avg_for_plotting[parm]),
                    xerr=numpy.ma.compressed(self.avg_for_plotting["fr"])/2,
                    yerr=[numpy.ma.compressed(self.avg_for_plotting["f"+parm]),
                          numpy.ma.compressed(self.avg_for_plotting["f"+parm])],
                    rasterized=True, **style)

    def plot_chandra_sector(self, ax, parm="kT", merger=False, hot=False, cold=False,
                            style=dict()):
        if self.name != "cygA":
            print "ERROR: Sectoranalysis not available for", self.name
            return

        if merger:
            ax.errorbar(numpy.ma.compressed(self.merger_for_plotting["r"]),
                        numpy.ma.compressed(self.merger_for_plotting[parm]),
                        xerr=numpy.ma.compressed(self.merger_for_plotting["fr"])/2,
                        yerr=[numpy.ma.compressed(self.merger_for_plotting["f"+parm]),
                              numpy.ma.compressed(self.merger_for_plotting["f"+parm])], **style)
        if hot:
            ax.errorbar(numpy.ma.compressed(self.hot_for_plotting["r"]),
                        numpy.ma.compressed(self.hot_for_plotting[parm]),
                        xerr=numpy.ma.compressed(self.hot_for_plotting["fr"])/2,
                        yerr=[numpy.ma.compressed(self.hot_for_plotting["f"+parm]),
                              numpy.ma.compressed(self.hot_for_plotting["f"+parm])], **style)
        if cold:
            ax.errorbar(numpy.ma.compressed(self.cold_for_plotting["r"]),
                        numpy.ma.compressed(self.cold_for_plotting[parm]),
                        xerr=numpy.ma.compressed(self.cold_for_plotting["fr"])/2,
                        yerr=[numpy.ma.compressed(self.cold_for_plotting["f"+parm]),
                              numpy.ma.compressed(self.cold_for_plotting["f"+parm])], **style)

    def plot_bestfit_betamodel(self, ax, style=dict(), rho=True):
        fit = profiles.gas_density_betamodel(self.ana_radii,
                self.rho0 if rho else self.ne0, self.beta, self.rc, self.rcut_kpc)

        if "label" not in style:
            label = r"\begin{tabular}{p{2.5cm}ll}"
            # label += " model & = & free beta \\\\"
            if rho:
                label += r" rho0 & = & {0:.2e} g$/$cm$^{{3}}$ \\".format(self.rho0)
            else:
                label += r" ne0 & = & {0:.2e} g$/$cm$^{{3}}$ \\".format(self.ne0)
            label += " beta & = & {0:.3f} \\\\".format(self.beta)
            label += " rc & = & {0:.2f} kpc \\\\".format(self.rc)
            label += (" \hline \end{tabular}")
            style["label"] = label
        ax.plot(self.ana_radii, fit, **style)

        ymin = profiles.gas_density_betamodel(
            self.rc, self.rho0 if rho else self.ne0, self.beta, self.rc)
        ax.vlines(x=self.rc, ymin=ymin, ymax=1e-10 if rho else 9.15,
                  **{ k: style[k] for k in style.keys() if k != "label" })


        # The y coordinates are axes while the x coordinates are data
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(self.rc-6 if self.name == "cygA" else self.rc-60, 0.98, r"$r_c$",
                ha="left", va="top", fontsize=22, transform=trans)

    def plot_bestfit_residuals(self, ax, rho=False):
        fit = profiles.gas_density_betamodel(self.avg["r"],
            self.rho0 if rho else self.ne0, self.beta, self.rc)

        residuals = (self.avg["n"] - fit)/self.avg["n"]

        ax.errorbar(self.avg["r"], 100*residuals,
                    yerr=100*self.avg["fn"]/self.avg["n"],
                    ls="", c="k", lw=3, elinewidth=1)
        ax.errorbar(self.avg["r"]-self.avg["fr"]/2, 100*residuals, c="k",
                    lw=3, elinewidth=1, drawstyle="steps-post")
        ax.axvline(x=self.rc, lw=3, ls="dashed", c="k")

    def plot_inferred_nfw_profile(self, ax, style=dict(), rho=True):
        rs = self.halo["rs"]
        density = self.halo["rho0_dm"] if rho else self.halo["ne0_dm"]
        rho_dm = profiles.dm_density_nfw(self.ana_radii, density, rs, rcut=self.rcut_nfw_kpc)

        if "label" not in style:
            label = r"\begin{tabular}{p{2.5cm}ll}"
            # label += " model & = & NFW \\\\"
            label += r" rho0dm & = & {0:.2e} g$\cdot$cm$^{{-3}}$ \\".format(self.halo["rho0_dm"])
            label += " rs & = & {0:.2f} kpc \\\\".format(rs)
            label += (" \hline \end{tabular}")
        ax.plot(self.ana_radii, rho_dm, **style)

        ymin = profiles.dm_density_nfw(rs, density, rs)
        ax.vlines(x=rs, ymin=ymin, ymax=1e-10 if rho else 9.15,
                  **{ k: style[k] for k in style.keys() if k != "label" })

        # The y coordinates are axes while the x coordinates are data
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(rs-25, 0.98, r"$r_s$", ha="right", va="top",
                transform=trans, fontsize=22)

    def plot_bestfit_betamodel_mass(self, ax, style=dict()):
        mass = convert.g2msun*self.M_gas(self.ana_radii*convert.kpc2cm)
        ax.plot(self.ana_radii, mass , **style)

    def plot_inferred_nfw_mass(self, ax, style=dict()):
        mass = convert.g2msun*self.M_dm(self.ana_radii*convert.kpc2cm)
        ax.plot(self.ana_radii, mass, **style)

    def plot_inferred_total_gravitating_mass(self, ax, style=dict()):
        mass = convert.g2msun*self.M_tot(self.ana_radii*convert.kpc2cm)
        ax.plot(self.ana_radii, mass, **style)

    def plot_hydrostatic_mass(self, ax, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "data"
        style["color"] = "b"
        ax.plot(self.HE_radii*convert.cm2kpc, self.HE_M_below_r*convert.g2msun,
                rasterized=True, **style)

    def plot_hydrostatic_mass_err(self, ax, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "data"
        style["color"] = "b"
        # style = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
        #         "elinewidth": 0.5, "label": "HE" }

        mask = self.avg_for_plotting.mask
        if self.name == "cygA":
             self.avg_for_plotting[10::2].mask = [True for i
                in range(len(self.avg_for_plotting.columns))]

        ax.errorbar(self.avg_for_plotting["r"], self.avg_for_plotting["M_HE"],
            xerr=self.avg_for_plotting["fr"]/2, yerr=[self.avg_for_plotting["M_HE_min"],
                self.avg_for_plotting["M_HE_plus"]], rasterized=True, **style)
        self.avg_for_plotting.mask = mask

    def M_verlinde(self, r):
        return profiles.verlinde_apparent_DM_mass(r, self.rho0, self.beta,
                                                  self.rc*convert.kpc2cm)

    def M_tot_verlinde(self, r):
        return (self.M_gas(r) + self.M_verlinde(r))

    def plot_verlinde(self, ax1, ax2, ax3, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "Verlinde"
        style["color"] = "r"


        radii = self.ana_radii  # if not fit else self.avg["r"]
        N = len(radii)
        mass = numpy.zeros(N)
        temperature = numpy.zeros(N)
        pressure = numpy.zeros(N)

        infinity = 1e25
        for i, r in enumerate(self.ana_radii*convert.kpc2cm):
            if not r: continue  # to skip masked values

            mass[i] = self.M_verlinde(r)
            temperature[i] = profiles.hydrostatic_temperature(
                r, infinity, self.rho_gas, self.M_tot_verlinde)
            pressure[i] = profiles.hydrostatic_gas_pressure(
                r, infinity, self.rho_gas, self.M_tot_verlinde)

        ax1.plot(radii, mass*convert.g2msun, **style)
        ax2.plot(radii, convert.K_to_keV(temperature), **style)
        ax3.plot(radii, pressure, **style)

    def plot_verlinde_pressure(self, style=dict()):
        style = { k: style[k] for k in style.keys() if k not in ["label", "c", "color"] }
        style["label"] = "Verlinde"
        style["color"] = "r"

    def plot_inferred_temperature(self, ax, style=dict()):
        radii = self.ana_radii
        label = "cNFW={0:.3f}, bf={1:.4f}".format(self.halo["cNFW"], self.halo["bf200"])
        ax.plot(radii, self.hydrostatic, label=style.get("label", label) ,
            **{ k: style[k] for k in style.keys() if k != "label" })

    def plot_inferred_pressure(self, ax, style=dict(), do_cut=False):
        ax.plot(self.ana_radii, self.hydrostatic_pressure, **style)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Class to hold Toycluster sampled clusters
# ----------------------------------------------------------------------------
class Toycluster(object):
    """ Parse and store Toycluster single cluster """
    def __init__(self, icdir, single=False, verbose=True):
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

        self.set_header_properties()

        # rhom only Toycluster
        self.gas["rhom"] = convert.toycluster_units_to_cgs(self.gas["rhom"])
        self.gas["rho"] = convert.toycluster_units_to_cgs(self.gas["rho"])
        self.gas["kT"] = convert.K_to_keV(convert.gadget_u_to_t(self.gas["u"]))

        if single:
            # 0 < Pos < boxSize. Set radius given that the center is at boxhalf
            self.gas["r"] = numpy.sqrt(p2(self.gas["x"] - self.boxhalf) +
                p2(self.gas["y"] - self.boxhalf) +  p2(self.gas["z"] - self.boxhalf))
            self.dm["r"] = numpy.sqrt(p2(self.dm["x"] - self.boxhalf) +
                p2(self.dm["y"] - self.boxhalf) + p2(self.dm["z"] - self.boxhalf))

            self.compute_profiles(verbose=verbose)
        else:
            if verbose: print "    Found two clusters in box --> running find_dm_centroid"

            # First find dark matter centroids
            if not self.find_dm_centroid(single=single, verbose=verbose):
                print "ERROR: find_dm_centroid failed!"
                return

            # Assign particles to left or right halo
            self.com = (self.centroid0[0] + self.centroid1[0])/2   # midpoint between both haloes
            left = numpy.where(self.gas["x"] < self.com)
            right = numpy.where(self.gas["x"] > self.com)

            # Create Cluster instances to hold the per-halo particles
            self.halo0 = Cluster(self.header)
            self.halo1 = Cluster(self.header)
            self.halo0.set_toycluster_halo(self.gas[left], self.dm[left], self.centroid0, verbose=verbose)
            self.halo1.set_toycluster_halo(self.gas[right], self.dm[right], self.centroid1, verbose=verbose)

    def __str__(self):
        tmp = "Toycluster ICfile header:\n"
        for k, v in self.header.iteritems(): tmp += "    {0:<17}: {1}\n".format(k, v)
        return tmp

    def set_header_properties(self):
        self.boxsize = self.header["boxSize"]
        self.boxhalf = self.header["boxSize"]/2

        self.M_dm_tot = self.header["ndm"] * self.header["massarr"][1] * 1e10
        self.M_gas_tot = self.header["ngas"] * self.header["massarr"][0] * 1e10

    def compute_profiles(self, verbose=True):
        self.set_gas_mass()
        self.set_gas_pressure()
        self.set_dm_mass(verbose=verbose)
        self.set_dm_density()

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

        if verbose:
            print "    Counting nr. of particles with radius < r to obtain M(<r)"

        radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(1e5), 257))
        dr = radii[1:] - radii[:-1]
        self.dm_radii = radii[:-1]
        N = len(self.dm_radii)

        particles = numpy.zeros(N)
        # gas_particles = numpy.zeros(N)
        for i, r in enumerate(self.dm_radii):
            particles[i] = ((numpy.where(self.dm["r"] < r)[0]).size)
            # gas_particles[i] = ((numpy.where(self.gas["r"] < r)[0]).size)
            if verbose and (i==(N-1) or i%100 == 0):
                print_progressbar(i, N, whitespace="    ")

        particles_in_shell = numpy.zeros(len(particles))
        # gas_particles_in_shell = numpy.zeros(len(gas_particles))
        for i in range(1, len(particles)):
            particles_in_shell[i-1] = particles[i] - particles[i-1]
            # gas_particles_in_shell[i-1] = gas_particles[i] - gas_particles[i-1]

        self.dm_volume = 4 * numpy.pi * self.dm_radii**2 * dr
        self.n_dm_in_shell = particles_in_shell
        # self.n_gas_in_shell = gas_particles_in_shell
        self.M_dm_below_r = particles * self.M_dm_tot/self.header["ndm"]
        # self.M_gas_below_r = gas_particles * self.M_gas_tot/self.header["ngas"]

    def set_dm_density(self):
        self.rho_dm_below_r = (self.M_dm_tot*convert.msun2g
                * (self.n_dm_in_shell/self.header["ndm"])
                / (self.dm_volume * p3(convert.kpc2cm)))

    def set_gas_pressure(self):
        self.gas["P"] = convert.rho_to_ne(self.gas["rho"]) *\
            convert.keV_to_erg(self.gas["kT"])

    def find_dm_peak(self, expected, dim="x"):
        if dim != "x" and dim != "y" and dim != "z":
            print "ERROR: please use 'x', 'y', or 'z' as dimension in find_dm_peak"
            return None
        nbins = int(numpy.sqrt(self.header["ndm"]))
        hist, edges = numpy.histogram(self.dm[dim], bins=nbins, normed=True)
        edges = (edges[:-1] + edges[1:])/2

        # savgol = scipy.signal.savgol_filter(hist, 21, 5)
        hist_smooth = scipy.ndimage.filters.gaussian_filter1d(hist, 5)
        spline = scipy.interpolate.splrep(edges, hist_smooth)
        xval = numpy.arange(0, self.boxsize, 0.1)
        hist_splev = scipy.interpolate.splev(xval, spline, der=0)
        peaks = peakutils.indexes(hist_splev)

        if len(peaks) != expected:
            print "ERROR: more than one {0}peak found".format(dim)
            return None

        # pyplot.figure()
        # pyplot.plot(edges, hist, **dm)
        # pyplot.plot(xval, hist_splev)
        # pyplot.ylim(0, 1.1*numpy.max(hist))
        # pyplot.xlabel(dim)
        # pyplot.ylabel("Normed Counts")
        # for peak in xval[peaks]: pyplot.axvline(peak)
        # pyplot.tight_layout()
        # pyplot.savefig(sim.outdir+"dm_peak_{0}".format(dim)+snapnr+".png", dpi=300)
        # pyplot.close()

        return xval[peaks]

    def find_dm_centroid(self, single=True, verbose=True):
        """ TODO: It is important to get halo centroid right, otherwise we plot puffy
        profiles while the sampled profiles could be sharper ...
        Toycluster does print the xpeaks and ypeaks at runtime (so for ICs we can
        verify this method) """
        if single:
            exp_x = 1
            exp_y = 1
            exp_z = 1
        else:  # two clusters
            if self.parms["ImpactParam"] == 0.0:
                exp_x = 2
                exp_y = 1
                exp_z = 1
                # TODO: investigate if this makes profiles less puffy
                # ypeaks[0] = 0.0
                # zpeaks[0] = 0.0
            else:
                """ TODO: The histogram does not have enough resolution to find two ypeaks
                if the impactparam is not 0 (e.g. 50 kpc). We could split the haloes based
                on x-position and then look for the y peaks in self.dm["y"] """
                pass
                # TODO: implement for impactparam
                # exp_x = 2
                # exp_y = 2
                # exp_z = 1

        xpeaks = self.find_dm_peak(exp_x, "x")
        ypeaks = self.find_dm_peak(exp_y, "y")
        zpeaks = self.find_dm_peak(exp_z, "z")

        if type(xpeaks) != numpy.ndarray or type(ypeaks) != numpy.ndarray \
                or type(zpeaks) != numpy.ndarray : return False

        halo0 = xpeaks[0], ypeaks[0], zpeaks[0]
        halo1 = xpeaks[1 if exp_x == 2 else 0], ypeaks[1 if exp_y == 2 else 0], zpeaks[0]

        distance = numpy.sqrt(p2(halo0[0] - halo1[0]) + p2(halo0[1] - halo1[1]) +
                              p2(halo0[2] - halo1[2]))
        if single: halo1 = None
        self.centroid0, self.centroid1, self.distance = halo0, halo1, distance
        if verbose:
            print "    Success: found {0} xpeaks, {1} ypeak, and {2} zpeak!"\
                .format(exp_x, exp_y, exp_z)
            print "      halo0:  (x, y, z) = {0}".format(halo0)
            print "      halo1:  (x, y, z) = {0}".format(halo1)
            print "      distance          = {0:.2f} kpc\n".format(distance)
        return True  # success status


class Cluster(Toycluster):
    def __init__(self, header, verbose=True):
        self.header = header
        if verbose: "  Created Cluster instance"

    def set_toycluster_halo(self, gas, dm, centroid, verbose=True):
        self.ics = True
        self.set_header_properties()
        self.gas = gas
        self.dm = dm
        self.centroid = centroid

        # Shift halo to [0, 0, 0]
        self.gas["x"] -= self.centroid[0]
        self.gas["y"] -= self.centroid[1]
        self.gas["z"] -= self.centroid[2]
        self.dm["x"] -= self.centroid[0]
        self.dm["y"] -= self.centroid[1]
        self.dm["z"] -= self.centroid[2]

        self.gas["r"] = numpy.sqrt(p2(self.gas["x"]) + p2(self.gas["y"]) +  p2(self.gas["z"]))
        self.dm["r"] = numpy.sqrt(p2(self.dm["x"]) + p2(self.dm["y"]) +  p2(self.dm["z"]))

        self.compute_profiles(verbose=verbose)

    def set_gadget_single_halo(self, snapnr, path_to_snaphot, verbose=True):
        self.ics = False
        self.header, self.gas, self.dm = parse.toycluster_icfile(path_to_snaphot)
        self.set_header_properties()

        self.gas["rho"] = convert.toycluster_units_to_cgs(self.gas["rho"])
        self.gas["kT"] = convert.K_to_keV(convert.gadget_u_to_t(self.gas["u"]))
        self.gas["r"] = numpy.sqrt(p2(self.gas["x"] - self.boxhalf) +
            p2(self.gas["y"] - self.boxhalf) +  p2(self.gas["z"] - self.boxhalf))
        self.dm["r"] = numpy.sqrt(p2(self.dm["x"] - self.boxhalf) +
            p2(self.dm["y"] - self.boxhalf) +  p2(self.dm["z"] - self.boxhalf))

        self.compute_profiles(verbose=verbose)

    def set_gadget_double_halo(self, gas, dm, centroid, verbose=True):
        self.ics = False
        self.set_header_properties()
        self.gas = gas
        self.dm = dm
        self.centroid = centroid

        # Shift halo to [0, 0, 0]
        self.gas["x"] -= self.centroid[0]
        self.gas["y"] -= self.centroid[1]
        self.gas["z"] -= self.centroid[2]
        self.dm["x"] -= self.centroid[0]
        self.dm["y"] -= self.centroid[1]
        self.dm["z"] -= self.centroid[2]

        self.gas["r"] = numpy.sqrt(p2(self.gas["x"]) + p2(self.gas["y"]) +  p2(self.gas["z"]))
        self.dm["r"] = numpy.sqrt(p2(self.dm["x"]) + p2(self.dm["y"]) +  p2(self.dm["z"]))

        self.compute_profiles(verbose=verbose)

        # TODO

# ----------------------------------------------------------------------------
# Class to hold Gadget-2 simulation snaphots
# ----------------------------------------------------------------------------
class Gadget2Output(object):  # TODO parse individual snapshots, split box in half, etc
    """ Parse and store Gadget-2 simulation snapshots"""
    def __init__(self, simdir, verbose=True):
        """ Class to hold Gadget-2 simulation output
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
class Gadget3Output(object):  # TODO parse individual snapshots, split box in half, etc
    """ Parse and store Gadget-3 simulation snapshots"""
    def __init__(self, simdir, verbose=True):
        """ Class to hold Gadget-3 simulation output
        @param simdir: path to the directory with Gadget-3 output, string
        @return      : instance of Gadget3Output class"""

        self.parms = parse.read_gadget3_parms(simdir+"gadget3.par")
        self.set_snapshot_paths(simdir)

    def __str__(self):
        tmp = "Gadget-3 parameters:\n"
        for k, v in self.parms.iteritems(): tmp += "    {0:<39}: {1}\n".format(k, v)
        return tmp

    def set_snapshot_paths(self, simdir):
        self.snapshots = sorted(glob.glob(simdir+"snapshot_*"), key=os.path.getmtime)


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
        attr_renamed = {
            "rhogas": "rhogas", "DMrho": "rhodm",
            "Tspec": "tspec",
            "SZ": "sz",
            "Tem": "tem",
            "xray": "xray",
            "vel": "vel"
        }

        smaccubes = glob.glob(sim.analysisdir+"*.fits.fz")
        smaccubes = glob.glob(sim.analysisdir+"*xray_0.fits.fz")
        [ smaccubes.append(s) for s in glob.glob(sim.analysisdir+"*best.fits.fz") ]
        [ smaccubes.append(s) for s in glob.glob(sim.analysisdir+"*best765.fits.fz") ]
        # smaccubes.append(glob.glob(sim.analysisdir+"*Tspec_rot-ea0.fits.fz")[0])
        for path in smaccubes:
            for cubename, attr in attributes.iteritems():
                if cubename in path:
                    break
            else:
                for cubename, attr in attr_renamed.iteritems():
                    if cubename in path:
                        break
                else:
                    print "ERROR: unknown fits filename '{0}'".format(path)
                    continue
            print path
            if "best765" in path:
                EA2 = (path.split("_")[-2]).split(".fits.fz")[0]
                attr = attr+EA2+"best765"
            elif "best" in path:
                EA2 = (path.split("_")[-2]).split(".fits.fz")[0]
                attr = attr+EA2+"best"
            else:
                attr = attr+(path.split("_")[-1]).split(".fits.fz")[0]
            print attr
            header, data = parse.psmac2_fitsfile(path)
            setattr(self, attr+"_header", header)
            setattr(self, attr, data)

    def available_smac_cubes(self):
        return [i for i in self.__dict__.keys() if i[:1] != "_" and "_header" not in i]
