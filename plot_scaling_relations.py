# -*- coding: utf-8 -*-

import numpy
import astropy.units as u
from matplotlib import pyplot
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = None
from astropy.table import Table

from cosmology import CosmologyCalculator
from plotsettings import PlotSettings
style = PlotSettings()


def eat_arnaud2010():
    header = [ "name", "R500", "Yx", "Ysph(R2500)", "Ysph(R500)", \
                "P500", "P0", "c500", "alpha", "gamma", "chisq", "dof" ]
    data = { i: numpy.zeros(31) for i in range(len(header)) }
    data[0] = numpy.zeros(31, dtype="|S16")
    # Table C.1. Cluster physical parameters from ArXiV source / tex file
    with open("Arnaud2010.tex", "r") as f:
        for n, line in enumerate(f):
            for i, col in enumerate(line.split("&")):
                value = col.replace("$", "")
                # Yx, Ysph(R2500), Ysph(R500)
                if "\pm" in value: value=value.split("\pm")[0]
                # chisq/dof
                if "/" in value:
                     chisq, dof = value.split("/")
                     data[i][n] = chisq.strip()
                     data[i+1][n] = dof.replace("\\", "").strip()
                else:
                    data[i][n] = value.strip()

    return Table([data[i] for i in range(12)],
                 names=[header[i] for i in range(12)])


def eat_pratt2009():
    header = [ "name", "z", "T1", "L1", "T2", \
                "L2", "T3", "Yx", "R500", "CC", "disturbed" ]
    header = [ "name", "z", "T1", "L1", "T2", "L2", "T3", "Yx", "R500", "Rdet"]
    data = { i: numpy.zeros(31) for i in range(len(header)) }
    data[0] = numpy.zeros(31, dtype="|S16")
    # Table 1. Cluster properties.
    n = 0
    with open("Pratt2009.tex", "r") as f:
        for line in f:
            for i, col in enumerate(line.split("&")[0:10]):
                value = col.replace("$", "").strip()
                if i == 0:
                    value=value.replace("\,", " ")
                # Remove asymmetrical errors
                if i >= 2 and i <= 7:  # columns T1 up to Yx
                    value=value.split("_")[0]
                data[i][n] = value.strip()
            n += 1

    return Table([data[i] for i in range(len(header))],
                 names=[header[i] for i in range(len(header))])


def eat_pratt2010():
    header = [ "name", "z", "kT", "M500", "K(0.1R200)", \
                "K(R2500)", "K(R1000)", "K(R500)", "CC", "disturbed" ]
    header = [ "name", "z", "kT", "M500"]
    data = { i: numpy.zeros(31) for i in range(len(header)) }
    data[0] = numpy.zeros(31, dtype="|S16")
    # Table C.1. Cluster physical parameters.
    n = 0
    with open("Pratt2010.tex", "r") as f:
        for line in f:
            if line == "\n" or len(line) < 10:
                continue
            for i, col in enumerate(line.split("&")[0:4]):
                value = col.replace("$", "").strip()
                if i == 0:
                    value=value.replace("\,", " ")
                if i == 2:  # kT in [0.15-0.75] R500 aperture /w asymm. errors
                    value=value[0:4]
                if i == 3:  # Mass [1/h_70 1e14 Msun]  /w asymm. errors
                    value=value[0:4]
                data[i][n] = value.strip()
            n += 1

    return Table([data[i] for i in range(len(header))],
                 names=[header[i] for i in range(len(header))])

def eat_reichert2011():
    # 0 contains data; 1 contains references
    return Vizier.get_catalogs("J/A+A/535/4")[0]


def plot_LxT500(Pratt2009):
    pyplot.figure()
    # T2 and L2 are in the [0.15-1] R500 region
    pyplot.scatter(Pratt2009["T2"], Pratt2009["L2"], lw=1, s=20, marker="D",
                   c="w", label="Pratt et al. 2009")

    T = numpy.arange(0, 40, 0.1)
    cc = CosmologyCalculator(z=0.0562)  #  z=numpy.mean(Pratt2009["z"]))
    # Boehringer+ 2011 eq. 4
    E_of_z = cc.rho_crit() / CosmologyCalculator(z=0).rho_crit()
    # Boehringer+ 2011 eq. 31
    pyplot.plot(T, 0.079*T**(2.7)*E_of_z**(-0.23), c="k",
                label="Boehringer et al. 2011")


    pyplot.xlabel("T$_{500}$, T$_{\\rm spectr}$ [keV]")
    pyplot.ylabel("L$_{\\rm X-ray, bol}$ [10$^{44}$ erg s$^{-1}$]")
    pyplot.xlim(1, 10)
    pyplot.ylim(0.2, 25)
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xticks([1, 10], ["1", "10"])
    pyplot.yticks([1, 10], ["1", "10"])
    pyplot.tight_layout()
    pyplot.legend(loc="upper left", fontsize=18)
    return pyplot.gcf().number


def plot_M500T500(Reichert2011):
    cc = CosmologyCalculator(z=0.0562)
    # Boehringer+ 2011 eq. 4
    E_of_z = cc.rho_crit() / CosmologyCalculator(z=0).rho_crit()

    pyplot.figure()
    pyplot.scatter(Reichert2011["kT"], Reichert2011["Mass"], lw=1, s=20,
                   marker="D", c="w", label="Reichert et al. 2011")

    T = numpy.arange(0, 40, 0.1)
    # Boehringer+ 2011 eq. 30
    # Slightly too high for El Gordo
    pyplot.plot(1/0.291*T**(1./1.62)*E_of_z**(-1./1.04), T, c="k",
                label="Boehringer et al. 2011")

    # Eyeballed it from Donnert 2014. z=0.87 for El Gordo
    # pyplot.plot(2*T**(1./1.62), T, c="k", label="Boehringer et al. 2011")

    pyplot.xlabel("T$_{500},<$T$_{\\rm sim}$(R$_{500})>$ [keV]")
    pyplot.ylabel("M$_{500}$ [10$^{14}$ M$_{\\rm sol}$]")
    pyplot.xlim(1, 13)
    pyplot.ylim(0.3, 40)
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xticks([1, 10], ["1", "10"])
    pyplot.yticks([1, 10], ["1", "10"])
    pyplot.tight_layout()
    pyplot.legend(loc="upper left", fontsize=18)
    return pyplot.gcf().number


def plot_M500Lx(Reichert2011):
    pyplot.figure()
    pyplot.scatter(Reichert2011["LX"], Reichert2011["Mass"], lw=1, s=20,
                   marker="D", c="w", label="Reichert et al. 2011")

    cc = CosmologyCalculator(z=0.0562)
    # Boehringer+ 2011 eq. 4
    E_of_z = cc.rho_crit() / CosmologyCalculator(z=0).rho_crit()
    M = numpy.arange(0, 40, 0.1)
    # Boehringer+ 2011 eq. 32
    pyplot.plot(1/1.64*M**(1/0.52)*E_of_z**(-1./0.90), M, c="k",
                label="Boehringer et al. 2011")

    pyplot.xlabel("L$_{\\rm X-ray, bol}$ [10$^{44}$ erg s$^{-1}$]")
    pyplot.ylabel("M$_{500}$ [10$^{14}$ M$_{\\rm sol}$]")
    pyplot.xlim(0.4, 37)
    pyplot.ylim(0.3, 40)
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xticks([1, 10], ["1", "10"])
    pyplot.yticks([1, 10], ["1", "10"])
    pyplot.tight_layout()
    pyplot.legend(loc="upper left", fontsize=18)
    return pyplot.gcf().number


def arnaud2010_eq16(M500):
    """ Numerical coefficients given in the corresponding Eqs. (5) and (16) are
        obtained for f_B = 0.175, mu = 0.59 and mu_e = 1.14, the values adopted
        by Nagai+ (2007), allowing a direct comparison with their best fitting
        GNFW model. - Arnaud+ 2010 appendix A """

    return 2.925e-5 * M500**(5./3)  # ignoring h(z) and h=70 assumption


def plot_Y500M500(Arnaud2010, Pratt2010):
    pyplot.figure()

    M500 = numpy.arange(0, 40, 0.1)
    pyplot.plot(M500, 10**(-4.739) * (M500/3)**(1.790), c="k", ls="solid",
                label="$Y_{\\rm sph}$ - M$_{500}$ (Arnaud et al. 2010, eq. 20)")
    pyplot.plot(M500, 2.925e-5 * (M500/3)**(5./3), c="k", ls="dashed",
                label="Y$_{\\rm 500,upp}$ (Arnaud et al. 2010, eq. 16)")
    pyplot.scatter(Pratt2010["M500"], Arnaud2010["Ysph(R500)"]*1e-5, lw=1,
                   s=20, marker="D", c="w", label="Arnaud et al. 2010")

    pyplot.xlabel("M$_{500}$ [10$^{14}$ M$_{\\rm sol}$]")
    pyplot.ylabel("Y($<$R$_{\\rm 500}$) [Mpc$^2$]")
    pyplot.xlim(0.4, 18)
    pyplot.ylim(1e-6, 3.8e-4)
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xticks([1, 10], ["1", "10"])
    pyplot.tight_layout()
    pyplot.legend(loc="upper left", fontsize=18)
    return pyplot.gcf().number


if __name__ == "__main__":
    # Eat datasets
    Arnaud2010 = eat_arnaud2010() # R500 and Compton-Y
    Pratt2009 = eat_pratt2009()   # Lx
    Pratt2010 = eat_pratt2010()   # Arnaud 2010 gets M500 from Pratt 2010
    Reichert2011 = eat_reichert2011()

    # find differences in names between catalogs
    # for i, line in enumerate(Arnaud2010):
    #     if line["name"] != Pratt2010[i]["name"]:
    #         print Arnaud2010[i]["name"],  Pratt2010[i]["name"]  # 1 and 19

    # Fix slight difference in name
    Arnaud2010[1]["name"] = "RXC J0006.0-3443"  # RXC J0006.6-3443
    Arnaud2010[19]["name"] = "RXC J1516.3+0005"  # RXC J1516+0005

    # Build Donnert 2014 Figure 3
    # TODO: set both z=0 and z=1 scaling relation line in plots?
    f1 = plot_LxT500(Pratt2009)               # upper right: Lx-T500
    f2 = plot_M500T500(Reichert2011)          # upper left: M500-T500
    f3 = plot_M500Lx(Reichert2011)            # lower right: M500-Lx
    f4 = plot_Y500M500(Arnaud2010, Pratt2010) # lower left: Y(<R500)-M500


    # Add CygA and CygNW to scaling relation plots
    import main
    a = main.new_argument_parser().parse_args()
    a.do_cut = True; a.clustername = "both"
    cygA, cygNW = main.set_observed_clusters(a)

    import scipy
    import convert
    import profiles
    M500A = cygA.halo["M500"]/1e14
    M500NW = cygNW.halo["M500"]/1e14
    # Spline breaks beyond 1 Mpc
    # T500A = convert.K_to_keV(scipy.interpolate.splev(
    #     cygA.halo["r500"]*convert.kpc2cm, cygA.T_spline))
    # T500NW = convert.K_to_keV(scipy.interpolate.splev(
    #     cygNW.halo["r500"]*convert.kpc2cm, cygNW.T_spline))
    T500A = convert.K_to_keV(profiles.hydrostatic_temperature(
        cygA.halo["r500"]*convert.kpc2cm, 1e25, cygA.rho_gas, cygA.M_tot))
    T500NW = convert.K_to_keV(profiles.hydrostatic_temperature(
        cygNW.halo["r500"]*convert.kpc2cm, 1e25, cygNW.rho_gas, cygNW.M_tot))

    # TODO: this is from observation and does not go out to R500!
    # From cygnus_lss_fill_flux.fits [photons/s/cm^2/arcsec^2] 1.03 Msec
    # DS9 circle, center = (299.86582 40.737717); radius = 249.664''
    LxA_obs = 0.013785478 * convert.keV_to_erg(5.9) * \
        (4*numpy.pi*(cygA.cc.DL_Mpc*1000*convert.kpc2cm)**2)
    LxA_obs /= 1e44
    # DS9 circle, center = (299.70382 40.886607); radius = 196.52''
    LxNW_obs = 0.0018499238 * convert.keV_to_erg(9.8) * \
        (4*numpy.pi*(cygNW.cc.DL_Mpc*1000*convert.kpc2cm)**2)
    LxNW_obs /= 1e44

    # Lx Smac of ICs, then (Sum/pix^2 in region of size R500)*(XYlen*kpc2cm)**2
    LxA = 7.57
    LxNW = 2.13


    # cut
    pyplot.figure(f1)
    pyplot.scatter(T500A, LxA, lw=0, s=50, marker="D", c="r", label="cut A")
    pyplot.scatter(T500NW, LxNW, lw=0, s=50, marker="D", c="g", label="cut NW")
    pyplot.axhline(LxA_obs, c="r", lw=1, ls="dotted")
    pyplot.axhline(LxNW_obs, c="g", lw=1, ls="dotted")

    pyplot.figure(f2)
    pyplot.scatter(T500A, M500A, lw=0, s=50, marker="D", c="r", label="cut A")
    pyplot.scatter(T500NW, M500NW, lw=0, s=50, marker="D", c="g", label="cut NW")

    pyplot.figure(f3)
    pyplot.scatter(LxA, M500A, lw=0, s=50, marker="D", c="r", label="cut A")
    pyplot.scatter(LxNW, M500NW, lw=0, s=50, marker="D", c="g", label="cut NW")
    pyplot.axvline(LxA_obs, c="r", lw=1, ls="dotted")
    pyplot.axvline(LxNW_obs, c="g", lw=1, ls="dotted")


    a.do_cut = False
    cygA, cygNW = main.set_observed_clusters(a)

    M500A = cygA.halo["M500"]/1e14
    M500NW = cygNW.halo["M500"]/1e14
    T500A = convert.K_to_keV(profiles.hydrostatic_temperature(
        cygA.halo["r500"]*convert.kpc2cm, 1e25, cygA.rho_gas, cygA.M_tot))
    T500NW = convert.K_to_keV(profiles.hydrostatic_temperature(
        cygNW.halo["r500"]*convert.kpc2cm, 1e25, cygNW.rho_gas, cygNW.M_tot))

    # TODO: Lx from smac cube at R500
    # LxA
    # LxNW

    pyplot.figure(f1)
    # pyplot.scatter(T500A, LxA, lw=0, s=60, marker="o", c="r", label="uncut A")
    # pyplot.scatter(T500NW, LxNW, lw=0, s=60, marker="o", c="g", label="uncut NW")
    pyplot.legend(loc="upper left", fontsize=18)
    pyplot.savefig("out/Lx_vs_kT500.pdf", dpi=600)

    pyplot.figure(f2)
    pyplot.scatter(T500A, M500A, lw=0, s=60, marker="o", c="r", label="uncut A")
    pyplot.scatter(T500NW, M500NW, lw=0, s=60, marker="o", c="g", label="uncut NW")
    pyplot.legend(loc="upper left", fontsize=18)
    pyplot.savefig("out/M500_vs_kT500.pdf", dpi=600)

    pyplot.figure(f3)
    # pyplot.scatter(LxA, M500A, lw=0, s=60, marker="o", c="r", label="uncut A")
    # pyplot.scatter(LxNW, M500NW, lw=0, s=60, marker="o", c="g", label="uncut NW")
    pyplot.legend(loc="upper left", fontsize=18)
    pyplot.savefig("out/M500_vs_Lx.pdf", dpi=600)

    # TODO: fix compton-y in profiles.py and in cluster.py, then plot...
    pyplot.figure(f4)
    pyplot.savefig("out/SZY_vs_M500.pdf", dpi=600)

    pyplot.show()

