import numpy
import astropy.units as u
import astropy.constants as const
import matplotlib
from matplotlib import pyplot
from matplotlib import gridspec

import profiles
import convert


# ----------------------------------------------------------------------------
# Plots for Chandra observations
# ----------------------------------------------------------------------------
def quiescent_parm(c, parm="rho"):
    """ @param c:  ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available".format(parm)
        return

    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(parm=parm, style=avg)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="best")
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_quiescent_{1}.pdf".format(c.name, parm), dpi=150)


def sector_parm(c, parm="kT"):
    """ @param c:  ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 2, "label": "Average "+parm }
    merger = { "marker": "o", "ls": "", "c": "g", "ms": 4, "alpha": 1,
               "elinewidth": 2, "label": "Merger "+parm }
    hot = { "marker": "o", "ls": "", "c": "r", "ms": 4, "alpha": 0.5,
            "elinewidth": 2, "label": "Hot "+parm }
    cold =  { "marker": "o", "ls": "", "c": "purple", "ms": 4, "alpha": 0.5,
              "elinewidth": 2, "label": "Cold "+parm }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available".format(parm)
        return

    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(parm=parm, style=avg)
    c.plot_chandra_sector(parm=parm, merger=True, style=merger)
    c.plot_chandra_sector(parm=parm, hot=True, style=hot)
    c.plot_chandra_sector(parm=parm, cold=True, style=cold)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")  # Linear better shows merger-affected radii
    pyplot.xlim(3, 2000)
    if parm == "kT":
        pyplot.ylim(2, 12)
    else:
        pyplot.yscale("log")
    pyplot.legend(loc="best", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_sector_{1}.pdf".format(c.name, parm), dpi=150)


def chandra_coolingtime(c):
    """ @param c:  ObservedCluster """
    Tcool = profiles.sarazin_coolingtime(c.avg["n"]/u.cm**3,
            convert.keV_to_K(c.avg["kT"]))

    pyplot.figure(figsize=(12, 9))
    pyplot.plot(c.avg["r"], Tcool)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Cooling Time [yr]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.savefig("out/{0}_cooling_time.pdf".format(c.name), dpi=150)
    pyplot.tight_layout()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Plots analytical models based on observation
# ----------------------------------------------------------------------------
def bestfit_betamodel(c):
    """ Plot best-fit betamodel with residuals """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    fit = { "color": "k", "lw": 4, "linestyle": "solid" }

    fig, (ax, ax_r) = pyplot.subplots(2, 2, sharex=True, figsize=(16, 12))
    gs1 = matplotlib.gridspec.GridSpec(3, 3)
    gs1.update(hspace=0)
    ax = pyplot.subplot(gs1[:-1,:])
    ax_r = pyplot.subplot(gs1[-1,:], sharex=ax)  # residuals

    # Plot Chandra observation and betamodel with mles
    pyplot.sca(ax)

    c.plot_chandra_average(parm="rho", style=avg)
    c.plot_bestfit_betamodel(style=fit)
    pyplot.ylabel("Density [1/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.ylim(numpy.min(c.avg["rho"])/1.5, numpy.max(c.avg["rho"])*1.3)
    pyplot.legend(loc="lower left", fontsize=22)

    # Plot residuals
    pyplot.sca(ax_r)
    c.plot_bestfit_residuals()
    pyplot.axhline(y=0, lw=3, ls="dashed", c="k")
    pyplot.ylabel("Residuals [\%]")
    pyplot.xlabel("Radius [kpc]")
    pyplot.xscale("log")
    pyplot.xlim(2 if c.name == "cygA" else 20, 1e3 if c.name == "cygA" else 1.1e3)
    pyplot.ylim(-35, 35)

    # Fix for overlapping y-axis markers
    from matplotlib.ticker import MaxNLocator
    ax.tick_params(labelbottom="off")
    nbins = len(ax_r.get_yticklabels())
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="upper"))

    # Force axis labels to align
    ax.get_yaxis().set_label_coords(-0.07, 0.5)
    ax_r.get_yaxis().set_label_coords(-0.07, 0.5)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_bestfit_betamodel.pdf".format(c.name), dpi=150)
    pyplot.sca(ax)


def inferred_nfw_profile(c):
    """ Plot the observed gas density, best-fit betamodel and the inferred
        best-fit NFW profile for the cluster
        @param c   : ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    fit = { "color": "k", "lw": 1, "linestyle": "dashed" }
    dm = { "color": "k", "lw": 1, "linestyle": "solid" }

    pyplot.figure(figsize=(12,9))
    c.plot_chandra_average(parm="rho", style=avg)
    c.plot_bestfit_betamodel(style=fit, do_cut=True)
    c.plot_inferred_nfw_profile(style=dm)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e-32, 9e-24,
        facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=c.halo["r200"], c="k", lw=1)
    pyplot.text(c.halo["r200"]+100, 4e-24, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-32, ymax=9e-24)
    pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_inferred_nfw_cNFW={1:.3f}_bf={2:.4f}.png"
        .format(c.name, c.halo["cNFW"], c.halo["bf200"]), dpi=150)
    pyplot.close()


def inferred_temperature(c):
    """ Plot the observed temperature profile and the inferred hydrostatic
        temperature for the best-fit betamodel and inferred total mass profile
        M_tot (spherically symmetric volume-integrated NFW plus ~ betamodel)
        @param c: ObservedCluster """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 2, "label": "Average kT" }

    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(parm="kT", style=avg)
    c.plot_inferred_temperature()

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("kT (keV)")
    pyplot.xscale("log")
    pyplot.xlim(3, 2000)
    pyplot.ylim(0.1, 12)
    pyplot.legend(loc="best", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_hydrostatic-temperature_cNFW={1:.3f}_bf={2:.4f}.png"
        .format(c.name, c.halo["cNFW"], c.halo["bf200"]), dpi=150)
    pyplot.close()
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Plots numerical haloes sampled with Toycluster
# ----------------------------------------------------------------------------
def toycluster_profiles(obs, ics):
    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas = { "marker": "o", "ls": "", "c": "g",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    dm = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas_a = { "color": "k", "lw": 1, "linestyle": "dashed", "label": "" }
    dm_a = { "color": "k", "lw": 1, "linestyle": "solid", "label": "" }

    pyplot.figure(figsize=(12,9))

    obs.plot_chandra_average(parm="rho", style=avg)
    rho_gas = convert.toycluster_units_to_cgs(ics.profiles["rho_gas"])
    rho_dm = convert.toycluster_units_to_cgs(ics.profiles["rho_dm"])
    pyplot.plot(ics.profiles["r"], rho_gas, **fit_a)
    pyplot.plot(ics.profiles["r"], rho_dm, **dm_a)
    # obs.plot_bestfit_betamodel(style=fit, do_cut=True)
    # obs.plot_inferred_nfw_profile(style=dm)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e-32, 9e-24,
        facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    pyplot.text(obs.halo["r200"]+100, 4e-24, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-32, ymax=9e-24)
    pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()


def toyclustercheck(obs, ics):
    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 1,
            "markeredgecolor": "none",  "label": ""}
    dm = { "marker": "o", "ls": "", "c": "b", "ms": 2, "alpha": 1,
            "markeredgecolor": "none", "label": ""}
    dashed = { "color": "k", "lw": 1, "linestyle": "dashed" }
    dotted = { "color": "k", "lw": 1, "linestyle": "dotted" }
    solid = { "color": "k", "lw": 1, "linestyle": "solid" }

    radii = numpy.arange(1, 1e4, 1)

    pyplot.figure(figsize=(12,9))
    pyplot.plot(ics.gas["r"], ics.gas["rho"], **gas)
    pyplot.plot(ics.dm_radii, ics.rho_dm_below_r, **dm)
    obs.plot_chandra_average(parm="rho", style=avg)
    obs.plot_bestfit_betamodel(style=dashed, do_cut=True)
    obs.plot_bestfit_betamodel(style=dotted, do_cut=False)
    obs.plot_inferred_nfw_profile(style=dotted)
    rho_dm_cut = profiles.dm_density_nfw(radii, obs.halo["rho0_dm"],
        obs.halo["rs"], ics.r_sample, do_cut=True)
    pyplot.plot(radii, rho_dm_cut, **solid)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e-32, 9e-24,
        facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    pyplot.text(obs.halo["r200"]+100, 4e-24, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-32, ymax=9e-24)
    pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_sampled_rho.png".format(obs.name), dpi=150)


    pyplot.figure(figsize=(12,9))
    pyplot.plot(ics.gas["r"], ics.gas["mass"], **gas)
    pyplot.plot(ics.dm_radii, ics.M_dm_below_r, **dm)
    mdm = profiles.dm_mass_nfw(radii,
        convert.cgs_density_to_msunkpc(obs.halo["rho0_dm"]), obs.halo["rs"])
    mdm_cut = [profiles.dm_mass_nfw_cut(r,
        convert.cgs_density_to_msunkpc(obs.halo["rho0_dm"]), obs.halo["rs"], ics.r_sample)
        for r in radii]
    mgas = profiles.gas_mass_betamodel(radii,
        convert.cgs_density_to_msunkpc(obs.rho0), obs.beta, obs.rc)
    mgas_cut = [profiles.gas_mass_betamodel_cut(r,
        convert.cgs_density_to_msunkpc(obs.rho0), obs.beta, obs.rc, obs.halo["r200"])
        for r in radii]

    pyplot.plot(radii, mdm, **dotted)
    pyplot.plot(radii, mdm_cut, **solid)
    pyplot.plot(radii, mgas, **dotted)
    pyplot.plot(radii, mgas_cut, **dashed)

    pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e7, 1e15,
       facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    pyplot.text(obs.halo["r200"]+100, 3e14, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass ($<$r) [MSun]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e7, ymax=1e15)
    # pyplot.legend(loc="lower left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_sampled_mass.png".format(obs.name), dpi=150)


def toyclustercheck_T(obs, ics):
    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "k",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    gas = { "marker": "o", "ls": "", "c": "g", "ms": 1, "alpha": 1,
            "markeredgecolor": "none",  "label": "sampled kT"}

    pyplot.figure(figsize=(12,9))
    pyplot.errorbar(obs.avg["r"], obs.avg["kT"], xerr=obs.avg["fr"]/2,
                    yerr=[obs.avg["fkT"], obs.avg["fkT"]], **avg)
    pyplot.plot(ics.gas["r"], ics.gas["kT"], **gas)
    pyplot.plot(ics.profiles["r"],
        convert.K_to_keV(convert.gadget_u_to_t(ics.profiles["u_gas"]).value),
        label="u\_gas")
    pyplot.plot(ics.profiles["r"],
        convert.K_to_keV(convert.gadget_u_to_t(ics.profiles["u_ana"]).value),
        label="u\_ana")

    # pyplot.fill_between(numpy.arange(2000, 1e4, 0.01), 1e7, 1e15,
    #   facecolor="grey", edgecolor="grey", alpha=0.2)
    pyplot.axvline(x=obs.halo["r200"], c="k", lw=1)
    # pyplot.text(obs.halo["r200"]+100, 3e14, r"$r_{200}$", ha="left", fontsize=22)

    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Temperature [keV]")
    pyplot.yscale("log")
    pyplot.xscale("log")
    pyplot.xlim(xmin=1, xmax=1e4)
    pyplot.ylim(ymin=1e-1, ymax=1e2)
    pyplot.legend(loc="upper left", fontsize=22)
    pyplot.tight_layout()
    pyplot.savefig("out/{0}_sampled_temperature.png".format(obs.name), dpi=150)


def psmac_xrays_with_dmrho_peakfind(sim, snapnr, xsum, ysum, xpeaks, ypeaks, distance):
    """ We find the peaks of the haloes by summing the dark matter density
        in the line-of-sight integrated P-Smac2 output. """

    fig = pyplot.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 3)
    axx = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    axy = fig.add_subplot(gs.new_subplotspec((1, 2), rowspan=2))
    axd = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2, rowspan=2),
                          sharex=axx, sharey=axy)
    axt = fig.add_subplot(gs.new_subplotspec((0, 2)))
    axt.axis("off")
    gs.update(wspace=0, hspace=0, top=0.94, left=0.15)
    axx.text(0.5, 1.01, "Summed (smoothed) Dark Matter Density X", ha="center", va="bottom",
             transform=axx.transAxes, fontsize=16)
    axx.plot(range(sim.xlen), xsum)
    axx.plot(xpeaks, xsum[(xpeaks+0.5).astype(int)], "ro", markersize=10)
    axx.set_ylim(0, 1.3)

    axy.text(1.01, 0.5, "Summed (smoothed) Dark Matter Density Y", ha="left", va="center",
             transform=axy.transAxes, fontsize=16, rotation=-90)
    axy.plot(ysum, range(sim.ylen))
    axy.plot(ysum[(ypeaks+0.5).astype(int)], ypeaks, "ro", markersize=10)
    axy.set_xlim(0, 1.3)

    axx.set_xlim(800, 1400)
    axy.set_ylim(sim.xlen/2-300,sim.xlen/2+300)
    axd.set_axis_bgcolor("k")
    axd.set_xlabel("x [pixel]")
    axd.set_ylabel("y [pixel]")
    for ax in [axx, axy, axt]:
        for tl in ax.get_xticklabels() + ax.get_yticklabels()\
                + ax.get_xticklines()  + ax.get_yticklines():
            tl.set_visible(False)
    pyplot.sca(axd)
    axt.text(0.02, 1, "T = {0:2.2f} [Gyr]".format(snapnr*sim.dt),
             ha="left", va="top")
    axt.text(0.02, 0.8, "R = {0:5.2f} [kpc]".format(distance),
             ha="left", va="top")

    if not numpy.isnan(distance):
        axd.plot(xpeaks[0], ypeaks[0], "ro", markersize=10)
        axd.plot(xpeaks[1], ypeaks[0], "ro", markersize=10)
    axd.imshow(numpy.log10(sim.psmac.xray[snapnr].clip(min=2e-8, max=0.02)),
               origin="lower", cmap="spectral")
    axd.text(0.5, 0.95, "X-ray Surface Brightness", ha="center", va="top",
            color="white", transform=axd.transAxes)
    pyplot.savefig(sim.outdir+"xray_peakfind_{0:03d}.png".format(snapnr))
    pyplot.close()


def simulated_quiescent_parm(c, sim, snapnr, parm="kT"):
    """ @param c     :  ObservedCluster instance
        @param sim   :  Simulation instance
        @param snapnr:  Number of snapshot to plot
        @param parm  :  Parameter to plot: ['kT', 'n', 'rho', 'P'] """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "g" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }

    parmnames = { "kT": "kT [keV]",
                  "n": "Density [1/cm$^3$]",
                  "rho": "Mass Density [g/cm$^3$]",
                  "P": "Pressure [erg/cm$^3$]" }
    parmmapping = { "kT": "tspec", }
                   #"n": "N/A",
                   # "rho": "rhogas"}
                   #"P": "N/A" }
    if not parmnames.get(parm, None):
        print "ERRROR: parm '{0}' is not available in observation".format(parm)
        return
    if not parmmapping.get(parm, None):
        print "ERRROR: parm '{0}' is not available in simulation".format(parm)
        return

    r, val, fval = sim.create_quiescent_profile(snapnr, parm=parmmapping[parm])

    pyplot.figure(figsize=(12, 9))
    c.plot_chandra_average(parm=parm, style=avg)
    pyplot.errorbar(r, val, yerr=[fval, fval])
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel(parmnames[parm])
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.xlim(1, 2000)
    # pyplot.ylim(-0.02, 0.2)
    pyplot.legend(loc="best")
    pyplot.tight_layout()
    pyplot.savefig(sim.outdir+"{0}_{1}_{2:03d}.png".format(parm, c.name, snapnr))
    pyplot.close()
