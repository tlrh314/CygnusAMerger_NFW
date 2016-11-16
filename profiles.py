import numpy
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


def dm_density_nfw(r, rho0_dm, rs, rcut=1e10):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density
        @param r:        radius, int or array
        @param rho0_dm:  dark matter central denisty, float
        @param rs:       NFW scale radius, float
        @param rcut:     NFW does not converge. Cut at sample radius
        @return:         NFW DM density profile rho(r), int or array """

    ra = r/rs

    rho_nfw = rho0_dm / (ra * p2(1+ra))
    return rho_nfw / (1 + pr(r/rcut))  # with cutoff


def gas_density_betamodel(r, rho0, beta, rc, rcut, do_cut=False):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)
        @param r:     radius, int or array
        @param rho0:  baryonic matter central denisty, float
        @param beta:  ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:    core radius (profile is constant within rc), float
        @param rcut:  numerical cutoff: keep local baryon fraction above unity, float
        @return:      NFW DM density profile rho(r), int or array """

    rho_gas = rho0 * numpy.power(1 + p2(r/rc), -3.0/2.0*beta)
    if do_cut:
        rho_gas /= (1 + p3(r/rcut))
    return rho_gas
