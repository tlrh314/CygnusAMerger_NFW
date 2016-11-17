import numpy
from scipy import special
import astropy.units as u

import convert
from macro import *


def dm_density_hernquist(r, M_dm, a):
    """ Hernquist (1990; eq. 2) profile for dark matter halo density.
        Also see Binney & Tremaine (1987; eq. 2.64)
        @param r:     radius, int or array
        @param M_dm:  total dark matter mass, float
        @param a:     Hernquist scale length, float
        @return:      Hernquist DM density profile rho(r), int or array """

    rho_dm = M_dm/(2*numpy.pi) * a / (r*p3(r + a))
    return rho_dm


def dm_mass_hernquist(r, M_dm, a):
    """ Hernquist (1990; eq. 3) profile for dark matter halo mass.
        @param r:     radius, int or array
        @param M_dm:  total dark matter mass, float
        @param a:     Hernquist scale length, float
        @return:      Hernquist DM mass profile M(<r), int or array """

    M_dm = M_dm * p2(r) / p2(r + a)
    return M_dm


def dm_density_nfw(r, rho0_dm, rs, rcut=1e10, do_cut=False):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density
        @param r:        radius, int or array
        @param rho0_dm:  dark matter central denisty, float
        @param rs:       NFW scale radius, float
        @param rcut:     NFW does not converge. Cut at sample radius
        @param do_cut:   Flag to cut at rcut (when sampling), or not (in fit), bool
        @return:         NFW DM density profile rho(r), int or array """

    ra = r/rs

    rho_nfw = rho0_dm / (ra * p2(1+ra))
    if do_cut:
        rho_nfw /= (1 + p2(r/rcut))  # with cutoff
    return rho_nfw

def cNFW(M200):
    """ Duffy+ (2008) concentration parameter for the dark matter halo
    NB we assume H0 = 70 km/s/Mpc.
    @param M200: total mass enclosed in virial radius r200 [MSun], float
    @return:     cNFW """

    M200 *= convert.g2msun  # because M200 should be in Msun here
    A = 5.74
    B = -0.097
    C = -0.47
    Mpivot = 2e12 / 0.7
    return A * numpy.power( M200/Mpivot, B)  # * numpy.power( 1+z, C)


def dm_mass_nfw(r, rho0_dm, rs):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density
        @param r:        radius, int or array
        @param rho0_dm:  dark matter central denisty, float
        @param rs:       NFW scale radius, float
        @return:         NFW DM mass profile M(<r), int or array """

    return 4*numpy.pi*rho0_dm*p3(rs) * (numpy.log((rs+r)/rs) - r/(rs+r))


def gas_density_betamodel(r, rho0, beta, rc, rcut=None, do_cut=False):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)
        @param r:      Radius, int or array
        @param rho0:   Baryonic matter central density, float
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float
        @param rcut:   Numerical cutoff: keep local baryon fraction above unity, float
        @param do_cut: Flag to cut at rcut (when sampling), or not (in fit), bool
        @return:       NFW DM density profile rho(r), int or array """

    rho_gas = rho0 * numpy.power(1 + p2(r/rc), -3.0/2.0*beta)
    if do_cut:
        rho_gas /= (1 + p3(r/rcut))
    return rho_gas


def gas_mass_betamodel(r, rho0, beta, rc):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)
        @param r:      Radius, int or array
        @param rho0:   Baryonic matter central density, float
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float
        @return:       NFW DM density profile rho(r), int or array """

    # beta = 2/3 (e.g. Mastropietro & Burkert 2008) has an analytic solution
    # if beta == 2./3:
    #     M_gas = 4*numpy.pi*p3(rc)*rho0 * (r/rc - numpy.arctan(r/rc))

    # Arbitrary beta: solved /w Scipy built-in Gauss Hypergeometric function
    M_gas = special.hyp2f1(1.5, 1.5*beta, 2.5, -p2(r/rc))
    M_gas *= 4*numpy.pi*rho0*p3(r)/3
    return M_gas


def sarazin_coolingtime(n_p, T_g):
    """ Sarazin (1988; eq. 5.23) cooling time w/e line cooling (T_g > 3e7 K)
        @param n_p: proton (or electron) number density [cm^-3], float
        @param T_g: gas temperature [K], float
        @return:    cooling timescale in clusters of galaxies [yr]
    """

    return 8.5e10*u.yr * (1e-3/u.cm**3)/n_p * numpy.sqrt(T_g/(1e8*u.K))

if __name__ == "__main__":
    r = numpy.arange(0.1, 2000, 0.1)
    rho1 = dm_density_nfw(r, 1.0, 367.94, do_cut=True)
    rho2 = dm_density_nfw(r, 1.0, 367.94, do_cut=False)

    from matplotlib import pyplot
    # pyplot.figure(figsize=(12,9))
    # pyplot.loglog(r, rho1, label="yes cut")
    # pyplot.loglog(r, rho2, label="no  cut")
    # pyplot.ylabel("DM Density NFW")
    # pyplot.xlabel("Radius [kpc]")
    # pyplot.legend()

    Mdm1 = dm_mass_hernquist(r, 7e8, 557.22)
    Mdm2 = dm_mass_nfw(r, 1, 367.94)

    pyplot.figure(figsize=(12,9))
    pyplot.loglog(r, Mdm1, label="Hernquist")
    pyplot.loglog(r, Mdm2, label="NFW")
    pyplot.ylabel("DM Mass")
    pyplot.xlabel("Radius [kpc]")
    pyplot.legend(loc="upper left")
    pyplot.show()
