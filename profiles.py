import numpy
import scipy
from scipy import special
import astropy.units as u
import astropy.constants as const

import convert
from macro import *


def dm_density_hernquist(r, M_dm, a):
    """ Hernquist (1990; eq. 2) profile for dark matter halo density.
        Also see Binney & Tremaine (1987; eq. 2.64)
        @param r:     radius, float or array
        @param M_dm:  total dark matter mass, float
        @param a:     Hernquist scale length, float
        @return:      Hernquist DM density profile rho(r), float or array """

    rho_dm = M_dm/(2*numpy.pi) * a / (r*p3(r + a))
    return rho_dm


def dm_mass_hernquist(r, M_dm, a):
    """ Hernquist (1990; eq. 3) profile for dark matter halo mass.
        @param r:     radius, float or array
        @param M_dm:  total dark matter mass, float
        @param a:     Hernquist scale length, float
        @return:      Hernquist DM mass profile M(<r), float or array """

    M_dm = M_dm * p2(r) / p2(r + a)
    return M_dm


def dm_density_nfw(r, rho0_dm, rs, rcut=1e10, do_cut=False):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density
        @param r:        radius, float or array
        @param rho0_dm:  dark matter central denisty, float
        @param rs:       NFW scale radius, float
        @param rcut:     NFW does not converge. Cut at sample radius
        @param do_cut:   Flag to cut at rcut (when sampling), or not (in fit), bool
        @return:         NFW DM density profile rho(r), float or array """

    ra = r/rs

    rho_nfw = rho0_dm / (ra * p2(1+ra))
    if do_cut:
        rho_nfw /= (1 + p2(r/rcut))  # with cutoff
    return rho_nfw


def dm_mass_nfw(r, rho0_dm, rs):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density
        @param r:        radius, float or array
        @param rho0_dm:  dark matter central denisty, float
        @param rs:       NFW scale radius, float
        @return:         NFW DM mass profile M(<r), float or array """

    return 4*numpy.pi*rho0_dm*p3(rs) * (numpy.log((rs+r)/rs) - r/(rs+r))


def dm_mass_nfw_cut(rmax, rho0_dm, rs, rcut):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density
        An additional cut-off at rcut=r_sample is used when sampling the density
        @param r:        radius, float (does not work for arrays)
        @param rho0_dm:  dark matter central denisty, float
        @param rs:       NFW scale radius, float
        @param rcut:     NFW scale radius, float
        @return:         NFW DM mass profile M(<r), float """

    M_dm = scipy.integrate.quad(lambda r:
        p2(r)*dm_density_nfw(r, rho0_dm, rs, rcut=rcut, do_cut=True),
        0, rmax)
    return 4*numpy.pi*M_dm[0]


def cNFW(M200):
    """ Duffy+ (2008) concentration parameter for the dark matter halo
    NB we assume H0 = 70 km/s/Mpc.
    @param M200: total mass enclosed in virial radius r200 [MSun], float
    @return:     cNFW """

    M200 *= convert.g2msun  # because M200 should be in Msun here
    A = 5.74
    B = -0.097
    C = -0.47  # C200 = 0 for redshift == 0, otherwise -0.47+/-0.04
    Mpivot = 2e12 / 0.7
    return A * numpy.power( M200/Mpivot, B)  # * numpy.power( 1+z, C)


def gas_density_betamodel(r, rho0, beta, rc, rcut=None, do_cut=False):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)
        @param r:      Radius, float or array
        @param rho0:   Baryonic matter central density, float
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float
        @param rcut:   Numerical cutoff: keep local baryon fraction above unity, float
        @param do_cut: Flag to cut at rcut (when sampling), or not (in fit), bool
        @return:       NFW DM density profile rho(r), float or array """

    rho_gas = rho0 * numpy.power(1 + p2(r/rc), -3.0/2.0*beta)
    if do_cut:
        rho_gas /= (1 + p3(r/rcut))
    return rho_gas


def gas_mass_betamodel(r, rho0, beta, rc):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)
        @param r:      Radius, float or array
        @param rho0:   Baryonic matter central density, float
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float
        @return:       NFW DM density profile rho(r), float or array """

    # beta = 2/3 (e.g. Mastropietro & Burkert 2008) has an analytic solution
    # if beta == 2./3:
    #     M_gas = 4*numpy.pi*p3(rc)*rho0 * (r/rc - numpy.arctan(r/rc))

    # Arbitrary beta: solved /w Scipy built-in Gauss Hypergeometric function
    M_gas = special.hyp2f1(1.5, 1.5*beta, 2.5, -p2(r/rc))
    M_gas *= 4*numpy.pi*rho0*p3(r)/3
    return M_gas


def gas_mass_betamodel_cut(rmax, rho0, beta, rc, rcut):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)
        M_gas (<rmax) from numerically integrated betamodel with additional cut-off
        No solution in terms of known mathematical functions exists.
        @param rmax:   Radius, float (does not work for arrays)
        @param rho0:   Baryonic matter central density, float
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float
        @param rcut:   Cut radius (cut is at r200), float
        @return:       NFW DM density profile rho(r), float or array """

    M_gas = scipy.integrate.quad(lambda r:
        p2(r)*gas_density_betamodel(r, rho0, beta, rc, rcut=rcut, do_cut=True),
        0, rmax)
    return 4*numpy.pi*M_gas[0]


""" Standard analytical temperature profile from Donnert 2014.
To avoid negative temperatures we define rmax*sqrt3 as outer radius """
def F0(r, rc, a):
    rc2 = p2(rc)
    a2 = p2(a)

    result = (a2-rc2)*numpy.arctan(r/rc) - rc*(a2+rc2)/(a+r) \
                + a*rc * numpy.log10( (a+r)*(a+r) / (rc2 + r*r) )

    result *= rc / p2(a2 + rc2)

    return result


def F1(r, rc):
    return p2(numpy.arctan(r/rc)) / (2*rc) + numpy.arctan(r/rc)/r


# the equations in Donnert 2014 are for Hernquist and betamodel: beta==2/3
# TODO: derive for NFW (with cut) and free betamodel (with cut)
def hydrostatic_temperature_TODO(r, rho0, rc, rho0_dm, rs, Rmax):
    """ In a relaxed galaxy cluster, the ICM is in approximate hydrostatic
        equilibrium in the gravitational potential of the cluster:
            1/rho_gas dP_gas/dr = -G Mtot(<r) / r^2
        so that with the ideal gas law: P = nkBT, the temperature of the ICM
        in hydrostatic equilibrium is given by (e.g. Mastropietro 2005)
            T(r) = mu m_p/kB G/rho_gas(r) int_r^Rmax rho_gas(t)/t^2 Mtot(<t) dt
        See Donnert (2014; eq. 8-13)    """

    m_p = const.m_p.to(u.g).value
    kB = const.k_B.to(u.erg/u.K).value
    fac = const.G.value*convert.umu * m_p/kB

    Mdm = dm_mass_nfw(r, rho0_dm, rs)
    temperature = fac * (1 + p2(r/rc)) * (Mdm*F0(r) + 4*numpy.pi*p3(rc)*rho0*F1(r, rc) )


def hydrostatic_temperature(r, Rmax, rho_gas, M_tot):
    m_p = const.m_p.to(u.g).value
    kB = const.k_B.to(u.erg/u.K).value
    fac = convert.umu * m_p/kB * const.G.cgs.value / rho_gas(r)

    # gas_mass_betamodel_cut(rmax, rho0, beta, rc, rcut)
    # dm_mass_nfw(r, rho0_dm, rs)
    # gas_density_betamodel(r, rho0, beta, rc, rcut=None, do_cut=False)

    tmp = scipy.integrate.quad(lambda t: rho_gas(t)/p2(t)*M_tot(t), r, Rmax)
    return fac*tmp[0]

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
