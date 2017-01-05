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

        @param r   : radius, float or array, [cm]
        @param M_dm: total dark matter mass, float, [g]
        @param a   : Hernquist scale length, float, [cm]
        @return    : Hernquist DM density profile rho(r), float or array, [g/cm^3] """

    rho_dm = M_dm/(2*numpy.pi) * a / (r*p3(r + a))
    return rho_dm


def dm_mass_hernquist(r, M_dm, a):
    """ Hernquist (1990; eq. 3) profile for dark matter halo mass.

        @param r   : radius, float or array, [cm or kpc]
        @param M_dm: total dark matter mass, float, [g or MSun]
        @param a   : Hernquist scale length, float, [cm or kpc]
        @return    : Hernquist DM mass profile M(<r), float or array, [g or MSun] """

    M_dm = M_dm * p2(r) / p2(r + a)
    return M_dm


def dm_density_nfw(r, rho0_dm, rs, rcut=None):
    """ Navarro, Frenk, White (1996) profile for dark matter halo density

        @param r      : radius, float or array, [cm or kpc]
        @param rho0_dm: dark matter central density, float, [1/cm^3, g/cm^3, or MSun/kpc^3]
        @param rs     : NFW scale radius, float, [cm or kpc]
        @param rcut   : NFW does not converge. Optional cut-off radius, float, [cm or kpc]
                            Set 'rcut' to 'None' for no cut-off.
        @return       : NFW DM density profile rho(r), float or array, [same unit as rho0_dm] """

    ra = r/rs

    rho_nfw = rho0_dm / (ra * p2(1+ra))
    if rcut is not None:
        rho_nfw /= (1 + p3(r/rcut))  # with cutoff
    return rho_nfw


def dm_mass_nfw(r, rho0_dm, rs, rcut=None):
    """ Navarro, Frenk, White (1996) profile for dark matter halo mass

        @param r      : radius, float or array, [cm or kpc]
        @param rho0_dm: dark matter central density, float, [g/cm^3 or MSun/kpc^3]
        @param rs     : NFW scale radius, float, [cm or kpc]
        @param rcut   : NFW does not converge. Optional cut-off radius, float, [cm or kpc]
                            Set 'rcut' to 'None' for no cut-off.
        @return       : NFW DM mass profile M(<r), float or array, [g or MSun depending on rho0_dm] """

    if rcut is None:
        return 4*numpy.pi*rho0_dm*p3(rs) * (numpy.log((rs+r)/rs) - r/(rs+r))
    else:
        r = numpy.array([r]) if type(r) != numpy.ndarray else r
        N = len(r)
        mass = numpy.zeros(N)
        for i, ri in enumerate(r):  # deco threaded slows things down
            mass[i] = scipy.integrate.quad(lambda r: 4*numpy.pi*p2(r)
                * dm_density_nfw(r, rho0_dm, rs, rcut=rcut), 0, ri)[0]
        return mass[0] if N == 1 else mass


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


def gas_density_betamodel(r, rho0, beta, rc, rcut=None):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)

        @param r:      Radius, float or array, [cm or kpc]
        @param rho0:   Baryonic matter central density, float, [1/cm^3, g/cm^3, or MSun/kpc^3]
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float, [cm or kpc]
        @param rcut:   Numerical cutoff: keep local baryon fraction above unity, float, [cm or kpc]
                           Set 'rcut' to 'None' for no cut-off.
        @return:       Gas DM density profile rho(r), float or array, [same unit as rho0] """

    rho_gas = rho0 * numpy.power(1 + p2(r/rc), -3.0/2.0*beta)
    if rcut is not None:
        rho_gas /= (1 + p3(r/rcut))
    return rho_gas


def d_gas_density_betamodel_dr(r, rho0, beta, rc):
    """ d/dr of uncut Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)

        @param r:      Radius, float or array, [cm or kpc]
        @param rho0:   Baryonic matter central density, float, [1/cm^3, g/cm^3, or MSun/kpc^3]
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float, [cm or kpc]
        @return:       Gas DM density profile derivative drho/dr, float or array,
                           [g/cm^4 or MSun/kpc^4 depending on rho0] """

    rho_gas = rho0*(-3*beta*r/p2(rc))*numpy.power(1 + p2(r/rc),-1.5*beta-1)
    return rho_gas


def gas_mass_betamodel(r, rho0, beta, rc, rcut=None):
    """ Cavaliere & Fusco-Femiano (1978) betamodel for baryonic mass density
        Also see Donnert (2014; eq. 6), Donnert (2017, in prep)

        @param r:      Radius, float or array, [cm or kpc]
        @param rho0:   Baryonic matter central density, float, [1/cm^3, g/cm^3, or MSun/kpc^3]
        @param beta:   Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc:     Core radius (profile is constant within rc), float, [cm or kpc]
        @param rcut:   Cut radius (cut is at r200), float, [cm or kpc]
                           Set 'rcut' to 'None' for no cut-off.
        @return:       Gas DM density profile rho(r), float or array """

    # beta = 2/3 (e.g. Mastropietro & Burkert 2008) has an analytic solution
    # if beta == 2./3:
    #     M_gas = 4*numpy.pi*p3(rc)*rho0 * (r/rc - numpy.arctan(r/rc))

    if rcut is None:
        # Arbitrary beta: solved /w Scipy built-in Gauss Hypergeometric function
        M_gas = special.hyp2f1(1.5, 1.5*beta, 2.5, -p2(r/rc))
        return M_gas * 4*numpy.pi*rho0*p3(r)/3
    else:
        r = numpy.array([r]) if type(r) != numpy.ndarray else r
        N = len(r)
        mass = numpy.zeros(N)
        for i, ri in enumerate(r):  # deco threaded slows things down
            mass[i] = scipy.integrate.quad(lambda r: 4*numpy.pi*p2(r) * gas_density_betamodel(
                r, rho0, beta, rc, rcut=rcut), 0, ri)[0]
        return mass[0] if N == 1 else mass


def verlinde_apparent_DM_mass(rmax, rho0, beta, rc):
    """ Emergent Gravity (Verlinde 2016) apparent dark matter mass.
        Equation adopted from Brouwer+ (2016; eq. 17)
        https://arxiv.org/pdf/1612.03034.pdf

        M_D(r) ^2 = cH_0/6G d(M_b(r)) / dr
        @param rmax: radius
        @param rho0: Baryonic matter central density, float
        @param beta: Ratio specific kinetic energy of galaxies to gas; slope, float
        @param rc  : Core radius (profile is constant within rc), float
        @return    : Apparent Dark Matter mass """

    H0 = 2.269e-18  # 70 km/s/Mpc --> one over second in cgs
    fac = (const.c.cgs.value*H0*p2(rmax)/(6*const.G.cgs.value))*4*numpy.pi*rho0*p3(rmax)
    # print numpy.sqrt(fac*scipy.misc.derivative(lambda r: gas_mass_betamodel(
    #    r, rho0, beta, rc), rmax))

    M_Dapparent = ( (p2(rmax/rc) + 1)**(-3*beta/2) -
                    special.hyp2f1(1.5, 1.5*beta, 2.5, -p2(rmax/rc)) +
                    4./3 * special.hyp2f1(1.5, 1.5*beta, 2.5, -p2(rmax/rc)) )

    return numpy.sqrt(fac * M_Dapparent)


def smith_centrally_decreasing_temperature(r, a, b, c):
    """ Smith+ (2002; eq. 4) """
    return a - b * numpy.exp(-1.0*r/c)


def smith_hydrostatic_mass(r, n, dn_dr, T, dT_dr):
    """ Smith+ (2002; eq. 3) Hydrostatic Mass from T(r), n_e(r)
        M(<r) = - k T(r) r^2/(mu m_p G) * (1/n_e * dn_e/dr + 1/T * dT/dr)

        @param r     : radius [cm], float/array
        @param n     : Radial number density [1/cm^3], float/array
        @param dn_dr : Derivative of number density /wr radius r, float/array
        @param T     : Temperature [Kelvin], float/array
        @param dT_dr : Derivative of temperature, float/array
        @param return: Radial hydrostatic mass-estimate M_HE(r), array """

    m_p = const.m_p.cgs.value
    kB = const.k_B.cgs.value
    fac = - kB / (convert.umu * m_p * const.G.cgs.value)

    return fac * T*p2(r) * ( 1/n * dn_dr + 1/T * dT_dr )


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
    fac = const.G.cgs.value*convert.umu * m_p/kB

    Mdm = dm_mass_nfw(r, rho0_dm, rs)
    temperature = fac * (1 + p2(r/rc)) * (Mdm*F0(r) + 4*numpy.pi*p3(rc)*rho0*F1(r, rc) )


def hydrostatic_temperature(r, Rmax, rho_gas, M_tot):
    """ In a relaxed galaxy cluster, the ICM is in approximate hydrostatic
        equilibrium in the gravitational potential of the cluster:
            1/rho_gas dP_gas/dr = -G Mtot(<r) / r^2
        so that with the ideal gas law: P = nkBT, the temperature of the ICM
        in hydrostatic equilibrium is given by (e.g. Mastropietro 2005)
            T(r) = mu m_p/kB G/rho_gas(r) int_r^Rmax rho_gas(t)/t^2 Mtot(<t) dt
        See Donnert (2014; eq. 8-13)

        CAUTION: constants used in this function are in cgs. Make sure that the
        rho_gas, M_tot and r are all in cgs units!!

        @param Rmax   : maximum value in integration, float, [cm]
        @param rho_gas: callable gas density profile, function pointer
        @param M_tot  : callable total mass profile, function pointer
        @return       : hydrostatic temperature in Kelvin"""

    m_p = const.m_p.to(u.g).value
    kB = const.k_B.to(u.erg/u.K).value
    fac = convert.umu * m_p/kB * const.G.cgs.value / rho_gas(r)

    temperature = scipy.integrate.quad(lambda t: rho_gas(t)/p2(t)*M_tot(t), r, Rmax)
    return fac*temperature[0]


def hydrostatic_gas_pressure(r, Rmax, rho_gas, M_tot):
    """ Gas pressure from hydrostatic equation (Donnert 2014, eq. 8)
        @param Rmax   : maximum value in integration, float, [cm]
        @param rho_gas: callable gas density profile, function pointer
        @param M_tot  : callable total mass profile, function pointer
        @return       : hydrostatic gas pressure erg/cm^3 """

    fac = const.G.cgs.value

    pressure = scipy.integrate.quad(lambda t: rho_gas(t)/p2(t)*M_tot(t), r, Rmax)
    return fac*pressure[0]


def sarazin_coolingtime(n_p, T_g):
    """ Sarazin (1988; eq. 5.23) cooling time w/e line cooling (T_g > 3e7 K)
        @param n_p: proton (or electron) number density [cm^-3], float
        @param T_g: gas temperature [K], float
        @return:    cooling timescale in clusters of galaxies [yr]
    """

    return 8.5e10*u.yr * (1e-3/u.cm**3)/n_p * numpy.sqrt(T_g/(1e8*u.K))
