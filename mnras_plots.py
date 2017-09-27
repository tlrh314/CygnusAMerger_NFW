import glob
import numpy
import scipy
from scipy import ndimage
import astropy
import astropy.units as u
from astropy.io import ascii, fits
import matplotlib
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
import aplpy
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = None
import colorcet

import main
import plot
import convert
from macro import print_progressbar
from simulation import Simulation
from plotsettings import PlotSettings
style = PlotSettings()


def adjustable_cmap(lss):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    f = fits.open(lss)
    im = pyplot.imshow(numpy.log10(scipy.ndimage.gaussian_filter(f[0].data, 9)),
        cmap=colorcet.cm["linear_bmw_5_95_c86"], origin="lower",
        vmin=numpy.log10(7.0e-10), vmax=numpy.log10(1.0e-7))

    min0 = numpy.log10(7.0e-10)
    max0 = numpy.log10(2.0e-6)

    fig.colorbar(im)

    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axmax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    smin = Slider(axmin, 'Min', -12, -6, valinit=min0)
    smax = Slider(axmax, 'Max', -8, -2, valinit=max0)

    def update(val):
        im.set_clim([smin.val, smax.val])
        fig.canvas.draw()
    smin.on_changed(update)
    smax.on_changed(update)

    pyplot.show()


def plot_mosaic(mosaic, cygA, cygNW, is_lss=False, is_kT=False):
    """ Zoomin of the merger region with distance measure
        @param mosaic: path to the Chandra x-ray mosaic fits file
        @param cygA  : tuple with RA, dec of CygA centroid
        @param cygNW : tuple with RA, dec of CygNW centroid
        @param is_lss: bool to indicate full mosaic or zoom-in (lss) """

    gc = aplpy.FITSFigure(mosaic)

    if not is_kT:
        # Add smoothed log-stretch of the entire mosaic
        gc.show_colorscale(vmin=5.0e-10, vmax=1.0e-7, stretch="log",
            cmap=colorcet.cm["linear_bmw_5_95_c86"], smooth=9)
    else:
        gc.show_colorscale(vmin=3.5, vmax=12, stretch="linear",
            cmap=colorcet.cm["linear_kryw_5_100_c67"], smooth=9)

    ax = pyplot.gca()

    # Find the pixels of the centroids
    cygA_x, cygA_y = gc.world2pixel(cygA[0], cygA[1])
    cygNW_x, cygNW_y = gc.world2pixel(cygNW[0], cygNW[1])

    # Add scale. Length is 500 kpc after unit conversions
    gc.add_scalebar(0.13227513)
    gc.scalebar.set_corner("bottom left" if is_lss or is_kT else "bottom right")
    gc.scalebar.set_length(0.1)
    gc.scalebar.set_linewidth(4)
    gc.scalebar.set_font_size(22 if is_lss or is_kT else 18)
    gc.scalebar.set_label("500 kpc")
    gc.scalebar.set_color("white")

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm")
    gc.frame.set_color("white")

    ax.tick_params(axis="both", which="both", reset=True, color="w", labelcolor="k",
        pad=8, width=2, size=4, direction="in", top="on", right="on")
    ax.tick_params(axis="both", which="major", size=8)

    # CygA and CygNW label
    ax.text(cygA_x, 0.80*cygA_y if not (is_lss or is_kT) else 0.60*cygA_y, "CygA",
            ha="center", va="center", color="white", fontsize=22)
    ax.text(cygNW_x, 1.12*cygNW_y, "CygNW", ha="center", va="center",
            color="white", fontsize=22)

    if is_lss:
        # Merger axis
        ax.plot([cygA_x, cygNW_x], [cygA_y, cygNW_y], c="w", lw=2, ls="--")

        # Eyeballed coordinates in ds9 :-) ...
        text_x, text_y = gc.world2pixel( 299.78952, 40.816273 )
        ax.text(text_x, text_y, "701.3''", ha="center", va="center", color="white",
                rotation=51, weight="bold", fontsize=22)
        # gc.recenter(299.78952, 40.81, width=0.185, height=0.185)
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_xticklabels("")
        # ax.set_yticklabels("")

        cax = pyplot.colorbar(gc.image, ax=ax, shrink=0.45, pad=0.03,
                              aspect=12, orientation="horizontal")
        cax.ax.xaxis.set_ticks_position("both")
        cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16)
        cax.ax.set_xlabel(r"X-ray Surface Brightness $\left[\frac{\mathrm{counts}}{\mathrm{cm}^2 \, \mathrm{s}} \right]$", fontsize=18)

        # Yes we must set ticks manually... +_+
        cax.set_ticks([1e-9, 1e-8, 1e-7])
        cax.set_ticklabels(["$10^{-9}$", "$10^{-8}$", "$10^{-7}$"])
        cax.ax.minorticks_on()
        minorticks = gc.image.norm(numpy.hstack([numpy.arange(5, 10, 1)/1e10,
            numpy.arange(2, 10, 1)/1e9, numpy.arange(2, 10, 1)/1e8]))
        cax.ax.tick_params(which="minor", length=3, width=1, direction="in")
        cax.ax.xaxis.set_ticks(minorticks, minor=True)

        # Wedge angles: 6, 96, 225 and 315 deg. Only merger+quiescent wedge
        radii = numpy.linspace(0, 1887, 100)   # radial profiles up to 1 Mpc, 0.53 kpc/pixel --> 1887 pixels
        x6 = numpy.zeros(len(radii))
        x96 = numpy.zeros(len(radii))
        y6 = numpy.zeros(len(radii))
        y96 = numpy.zeros(len(radii))
        for i, r in enumerate(radii):
            x6[i] = r*numpy.cos(9*numpy.pi/180)
            y6[i] = r*numpy.sin(6*numpy.pi/180)

            x96[i] = r*numpy.cos(96*numpy.pi/180)
            y96[i] = r*numpy.sin(96*numpy.pi/180)

        ax.plot(x6+cygA_x, y6+cygA_y, c="w", lw=2)
        ax.plot(x96+cygA_x, y96+cygA_y, c="w", lw=2)

        ax.plot(-x6+cygNW_x, -y6+cygNW_y, c="w", lw=2, ls=":")
        ax.plot(-x96+cygNW_x, -y96+cygNW_y, c="w", lw=2, ls=":")

        for label in ax.get_ymajorticklabels() + ax.get_yminorticklabels():
            label.set_rotation_mode("anchor")
            label.set_rotation(90)
            label.set_horizontalalignment("center")

        pyplot.xlabel("RA (J2000)")
        ax.xaxis.set_tick_params(labeltop="on", labelbottom="off")
        ax.xaxis.set_label_position("top")
        pyplot.ylabel("Dec (J2000)")
        start, end = ax.get_xlim()
        # ax.xaxis.set_ticks([0, 500, 1000, 1500, 2000], 'top')
    else:
        # Annotate top and low left corner with observation details
        # ax.text(0.5, 0.98, "Chandra X-ray Surface Brightness", weight="bold",
        #            fontsize=22, color="white", ha="center", va="top", transform=ax.transAxes)
        # ax.text(0.05, 0.05, "ACIS-I Mosaic\n0.5-7.0 keV\n1.03 Msec total exposure", color="white",
        #            fontsize=18, ha="left", va="bottom", transform=ax.transAxes)
        for label in ax.get_ymajorticklabels() + ax.get_yminorticklabels():
            label.set_rotation_mode("anchor")
            label.set_rotation(90)
            label.set_horizontalalignment("center")
    if is_kT:
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_xticklabels("")
        # ax.set_yticklabels("")
        pyplot.xlabel("RA (J2000)")
        ax.xaxis.set_tick_params(labeltop="on", labelbottom="off")
        ax.xaxis.set_label_position("top")
        pyplot.ylabel("Dec (J2000)")
        start, end = ax.get_xlim()
        # ax.xaxis.set_ticks([0, 500, 1000, 1500, 2000], 'top')

        cax = pyplot.colorbar(gc.image, ax=ax, shrink=0.45, pad=0.03,
                              aspect=12, orientation="horizontal")
        cax.ax.xaxis.set_ticks_position("both")
        cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16)
        cax.ax.set_xlabel(r"Temperature [keV]", fontsize=18)

        cax.ax.tick_params(which="minor", length=3, width=1, direction="in")

    # For some reason shows up double...
    ax.tick_params(axis="both", which="both", top="off", right="off")

    pyplot.tight_layout()
    if not is_kT:
        gc.save("out/mosaic_xray{0}.png".format("_lss" if is_lss else ""), dpi=300)
        gc.save("out/mosaic_xray{0}.pdf".format("_lss" if is_lss else ""), dpi=300)
    else:
        gc.save("out/lss_kT_9.png", dpi=300)
        gc.save("out/lss_kT_9.pdf", dpi=300)


def bestfit_betamodel(c):
    """ Plot best-fit betamodel with residuals """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "data (1.03 Msec)" }
    fit = { "color": "k", "lw": 4, "linestyle": "solid", "label": "best fit" }

    fig, (ax, ax_r) = pyplot.subplots(2, 2, sharex=True, figsize=(12, 9))
    gs1 = matplotlib.gridspec.GridSpec(3, 3)
    gs1.update(hspace=0)
    ax = pyplot.subplot(gs1[:-1,:])
    ax_r = pyplot.subplot(gs1[-1,:], sharex=ax)  # residuals

    # Plot Chandra observation and betamodel with mles
    pyplot.sca(ax)

    c.plot_chandra_average(ax, parm="rho", style=avg)
    c.plot_bestfit_betamodel(ax, style=fit)
    ax.collections[2].remove()
    ax.axvline(c.rc, ls="--", c="k", lw=3)
    ax.texts[0].remove()
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.1*c.rc, 0.98, r"$r_c$", ha="left", va="top", fontsize=22, transform=trans)

    pyplot.ylabel("Density [g/cm$^3$]")
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.ylim(numpy.min(c.avg["rho"])/1.5, numpy.max(c.avg["rho"])*1.3)
    pyplot.legend(loc="lower left", fontsize=22)

    # Plot residuals
    pyplot.sca(ax_r)
    c.plot_bestfit_residuals(ax_r)
    pyplot.axhline(y=0, lw=3, ls="dashed", c="k")
    pyplot.ylabel("Residuals [\%]")
    pyplot.xlabel("Radius [kpc]")
    pyplot.xscale("log")
    pyplot.xlim(1 if c.name == "cygA" else 20, 1.1e3)
    pyplot.ylim(-35, 35)

    # Fix for overlapping y-axis markers
    ax.tick_params(labelbottom="off")
    nbins = len(ax_r.get_yticklabels())
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="upper"))

    # Force axis labels to align
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    ax_r.get_yaxis().set_label_coords(-0.1, 0.5)

    ax.tick_params(axis="both", which="both", top="on", right="on")
    ax_r.tick_params(axis="both", which="both", top="on", right="on")
    pyplot.tight_layout()
    pyplot.savefig("out/bestfit_betamodel_{0}.pdf".format(c.name), dpi=150)
    pyplot.sca(ax)


def plot_mass_ratio(cygA, cygNW, cut=None):
    fig = pyplot.figure(figsize=(12, 10))
    ax = pyplot.gca()

    hydrostatic = ( scipy.ndimage.filters.gaussian_filter1d(cygA.HE_M_below_r, 1) /
                    scipy.ndimage.filters.gaussian_filter1d(cygNW.HE_M_below_r, 2) )
    pyplot.plot(cygA.HE_radii * convert.cm2kpc, hydrostatic, c="k", ls="--", label="Wise+ 2017")

    radii = cygA.ana_radii * convert.kpc2cm
    dark = cygA.M_dm(radii) / cygNW.M_dm(radii)
    gas = cygA.M_gas(radii) / cygNW.M_gas(radii)
    tot = cygA.M_tot(radii) / cygNW.M_tot(radii)

    pyplot.plot(cygA.ana_radii, gas, c="k", ls=":", label="betamodel")
    pyplot.plot(cygA.ana_radii, dark, c="k", ls="-.", label="NFW")
    pyplot.plot(cygA.ana_radii, tot, c="k", ls="-", label="NFW + betamdodel")

    pyplot.axvline(cygA.halo["r200"], ls="-", c="magenta", lw=1, label="cygA")
    pyplot.axvline(cygA.halo["r500"], ls="-", c="magenta", lw=1)
    pyplot.axvline(cygNW.halo["r200"], ls="-", c="blue", lw=1, label="cygNW")
    pyplot.axvline(cygNW.halo["r500"], ls="-", c="blue", lw=1)
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.97*cygA.halo["r200"], 0.98, r"$r_{200}$", ha="right", va="top",
            fontsize=28, transform=trans)
    ax.text(0.97*cygNW.halo["r500"], 0.98, r"$r_{500}$", ha="right", va="top",
            fontsize=28, transform=trans)

    pyplot.xscale("log")
    # pyplot.xlim(200, 1e4)
    # pyplot.ylim(0.5, 2.5)
    pyplot.xlim(60, 1.1*cygA.halo["r200"])
    pyplot.ylim(1, 4)
    pyplot.xlabel("Radius [kpc]", fontsize=32)
    pyplot.ylabel("Mass Ratio [CygA/CygNW]", fontsize=32)
    pyplot.legend(loc="upper center", fontsize=24)

    ax.tick_params(axis="both", which="both", top="on", right="on", labelsize=28)
    pyplot.tight_layout()
    pyplot.savefig("out/mass_ratio_{0}cut.pdf".format("" if cut else "un"), dpi=600)


def show_puffup():
    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 1,
            "elinewidth": 1, "label": "Quiescent, Chandra"}
    merger = { "marker": "o", "ls": "", "c": "g", "ms": 4, "alpha": 1,
            "elinewidth": 1, "label": "Merger, Chandra" }
    gas = { "linestyle": "solid", "color": "green", "linewidth": "2", "label": "Simulation, 230 Myr" }
    tot = { "color": "k", "lw": 1, "linestyle": "solid", "label": "Best-fit" }

    import main
    a = main.new_argument_parser().parse_args()
    a.do_cut = True; a.clustername = "both"
    cygA, cygNW = main.set_observed_clusters(a)

    from simulation import Simulation
    sim50 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0905", name="both",
                     set_data=False)
    sim = sim50
    sim.set_gadget_paths()
    snapnr = 23
    path_to_snaphot = sim.gadget.snapshots[snapnr]
    sim.set_gadget_snap_double(snapnr, path_to_snaphot)

    for c in [cygA, cygNW]:
        fig = pyplot.figure()
        ax = pyplot.gca()

        c.plot_chandra_average(ax, parm="kT", style=avg)
        c.plot_inferred_temperature(ax, style=tot)

        halo = getattr(sim, "{0}{1}".format(c.name, snapnr), None)
        ax.plot(halo.gas["r"], halo.gas["kT"], **gas)

        ax.axvline(c.halo["r200"], c="k")
        # The y coordinates are axes while the x coordinates are data
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(c.halo["r200"]+150, 0.98, r"$r_{200}$", ha="left", va="top",
                fontsize=22, transform=trans)
        ax.axvline(c.halo["r500"], c="k")
        ax.text(c.halo["r500"]-150, 0.98, r"$r_{500}$", ha="right", va="top",
                fontsize=22, transform=trans)
        ax.fill_between(numpy.arange(c.r_sample_dm, 1e4, 0.01), 0, 1,
            facecolor="grey", edgecolor="grey", alpha=0.2,
            transform=trans)

        inner = numpy.where(halo.gas["r"] < 50)
        hsml = 2*numpy.median(halo.gas["hsml"][inner])
        ax.axvline(x=hsml, c="g", ls=":")
        ax.text(hsml+6, 0.98, r"$2 h_{sml}$", ha="left", va="top",
            color="g", fontsize=22, transform=trans)

        ax.set_xscale("log")
        ax.set_xlim(10, 5000)
        ax.set_ylim(-1, 10)
        ax.set_xlabel("Radius [kpc]")
        ax.set_ylabel("Temperature [keV]")

        ax.legend(fontsize=22, loc=3)
        pyplot.tight_layout()
        pyplot.savefig("out/{0}_puffup.png".format(c.name))
        pyplot.close(fig)


def plot_simulated_wedges():
    from simulation import Simulation
    # sim50 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0905", name="both",
    #                  set_data=False)
    # sim75 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0906", name="both",
    #                  set_data=False)
    sim25 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0907", name="both",
                     set_data=False)

    avg = { "marker": "o", "ls": "", "c": "b", "ms": 4, "alpha": 0.4,
            "elinewidth": 1, "label": "Quiescent, Chandra"}
    merger = { "marker": "o", "ls": "", "c": "g", "ms": 4, "alpha": 0.4,
            "elinewidth": 1, "label": "Merger, Chandra" }


    import main
    a = main.new_argument_parser().parse_args()
    a.do_cut = False; a.clustername = "both"
    cygA, cygNW = main.set_observed_clusters(a)

    # pyplot.switch_backend("Agg")

    from panda import create_panda
    for Xe_i, sim in enumerate([ sim25 ]):
        # sim.read_ics()
        # sim.set_gadget_paths()
        sim.read_smac(verbose=True)
        sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
        sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)
        sim.dt = 0.01

        rmax = 900/sim.pixelscale
        radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(rmax), 42))
        dr = radii[1:] - radii[:-1]
        radii = radii[:-1]
        N = len(radii)

        for EA2_i, inclination in enumerate([ 45 ]):
            quiescent_temperature = numpy.zeros(N)
            quiescent_temperature_std = numpy.zeros(N)
            merger_temperature = numpy.zeros(N)
            merger_temperature_std = numpy.zeros(N)

            data = getattr(sim.psmac, "tspec{0}best765".format(inclination))[0]
            header = getattr(sim.psmac, "tspec{0}best765_header".format(inclination))
            snapnr = int(header["Input_File"].strip("'")[-3:])
            cA, cNW, distance = sim.find_cluster_centroids_psmac_dmrho(
                snapnr=0, EA2=inclination)

            for (xc, yc), name in zip([cA, cNW], ["cygA", "cygNW"]):
                fig = pyplot.figure()
                ax = pyplot.gca()
                for i, r in enumerate(radii):
                    print_progressbar(i, N)
                    angle1 = 96 if name == "cygA" else 276
                    angle2 = 6 if name == "cygA" else 186
                    quiescent_mask = create_panda(sim.xlen, sim.ylen, xc, yc,
                                                  r, angle1, angle2)
                    quiescent_temperature[i] = convert.K_to_keV(
                        numpy.median(data[quiescent_mask]))
                    quiescent_temperature_std[i] = convert.K_to_keV(
                        numpy.std(data[quiescent_mask]))

                    angle1 = 6 if name == "cygA" else 186
                    angle2 = 96 if name == "cygA" else 276
                    merger_mask = create_panda(sim.xlen, sim.ylen, xc, yc,
                                               r, angle1, angle2)
                    merger_temperature[i] = convert.K_to_keV(
                        numpy.median(data[merger_mask]))
                    merger_temperature_std[i] = convert.K_to_keV(
                        numpy.std(data[merger_mask]))

                # info = r"\begin{tabular}{lll}"
                # info += r" ID & = & {0} \\".format(sim.timestamp)
                # info += " $X_E$ & = & {0:.2f} \\\\".format((Xe_i+1)*0.25)
                # info += " snapnr & = & {0:03d} \\\\".format(snapnr)
                # info += " time & = & {0:04.2f} Gyr \\\\".format(snapnr*sim.dt)
                # info += " distance & = & {0:03.2f} kpc \\\\".format(distance)
                # info += (" \end{tabular}")

                ax.errorbar(radii*sim.pixelscale, quiescent_temperature,
                    [quiescent_temperature_std, quiescent_temperature_std],
                    c="b", lw=2, elinewidth=2, label="Quiescent, simulated")
                ax.errorbar(radii*sim.pixelscale, merger_temperature,
                    [merger_temperature_std, merger_temperature_std],
                    c="g", lw=2, elinewidth=2, label="Merger, simulated")

                if name == "cygA":
                    cygA.plot_chandra_average(ax, parm="kT", style=avg)
                    cygA.plot_chandra_sector(ax, parm="kT", merger=True, style=merger)
                elif name == "cygNW":
                    cygNW.plot_chandra_average(ax, parm="kT", style=avg)

                ax.set_yscale("linear")
                ax.set_xscale("log")
                ax.set_ylim(3.5, 12 if name=="cygA" else 10)
                ax.set_xlim(10 if name=="cygA" else 20, 900)
                ax.set_xlabel("Radius [kpc]")
                ax.set_ylabel("Temperature [keV]")
                filename = "{0}_{1}_{2}_{3}_{4}.pdf".format(
                    sim.timestamp, name, "0"+str((Xe_i+1)*25), snapnr, int(distance+0.5))
                pyplot.legend(loc="upper left", fontsize=22)
                pyplot.tight_layout()
                pyplot.savefig("out/"+filename, dpi=300)
                pyplot.close()


def plot_compton_y():
    # Open observation lss [counts/s/arcsec^2]
    lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
    lss_Lx = lss+".dir/Frame1/cygnus_lss_fill_flux.fits"
    lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"

    mosaic_Lx = fits.open(lss_Lx)
    mosaic_kT = fits.open(lss_kT)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    mosaic_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_Lx[0].data, 9)
    contour_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_Lx[0].data, 25)
    temperature_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_kT[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic_Lx[0].data.max()
    maxcounts_obs_index = mosaic_Lx[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic_Lx[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic_Lx[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic_Lx[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic_Lx[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    from simulation import Simulation
    sim = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0907",
        name="both", set_data=False)
    sim.read_smac(verbose=True)
    sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
    sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)
    sim.dt = 0.01
    sim.dtfine = 0.01/40

    inclination = 45

    data = getattr(sim.psmac, "sz{0}best765".format(inclination))
    header = getattr(sim.psmac, "sz{0}best765_header".format(inclination))

    maxcounts_sim = data.max()
    maxcounts_sim_index = data.argmax()  # of flattened array
    ylen_sim_pix, xlen_sim_pix = data[0].shape
    ylen_sim_kpc = xlen_sim_kpc = float(header["XYSize"])
    xcenter_sim = maxcounts_sim_index % xlen_sim_pix
    ycenter_sim = maxcounts_sim_index / xlen_sim_pix
    pix2kpc_sim = float(header["XYSize"])/int(header["XYPix"])

    cA, cNW, distance = sim.find_cluster_centroids_psmac_dmrho(
        snapnr=0, EA2=inclination)

    print "Smac Cube, i =", inclination
    print "  Shape ({0},   {1})   pixels".format(xlen_sim_pix, ylen_sim_pix)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_sim_kpc, ylen_sim_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_sim, ycenter_sim, maxcounts_sim)
    print "  Core Separaton = {0:3.2f}".format(distance)

    # Cut relevant part from the simulation
    desired_xlen_sim_kpc = xlen_obs_kpc
    desired_ylen_sim_kpc = ylen_obs_kpc
    desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
    desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
    xoffset = int((xcenter_sim * pix2kpc_sim -
        xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
    yoffset = int((ycenter_sim * pix2kpc_sim -
        ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

    equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                         xoffset: xoffset+desired_xlen_sim_pix]

    zoomx = float(ylen_obs_pix) / equal_boxsize_kpc_smaccube.shape[0]
    zoomy = float(xlen_obs_pix) / equal_boxsize_kpc_smaccube.shape[1]
    shape_matched = scipy.ndimage.zoom(equal_boxsize_kpc_smaccube,  [zoomx, zoomy], order=3)

    obsdir = "../runs/ChandraObservation/"
    confiles = glob.glob(obsdir+"xray/xray_contours_*.con")

    mosaic = "../runs/ChandraObservation/xray/cygnus_tot_flux.fits"
    gc = aplpy.FITSFigure(mosaic)

    pyplot.figure()
    ax = pyplot.gca()

    sz = pyplot.imshow(shape_matched, vmin=7e-6, vmax=1e-4,
        norm=matplotlib.colors.LogNorm(), origin="lower",
        cmap=colorcet.cm["diverging_rainbow_bgymr_45_85_c67"],
        extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
    cax = pyplot.colorbar(sz, ax=ax, shrink=0.45, pad=0.03,
        aspect=12, orientation="horizontal")
    cax.ax.xaxis.set_ticks_position("both")
    cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16, direction="in")
    cax.ax.set_xlabel(r"Compton-Y Parameter", fontsize=18)
    cax.ax.tick_params(which="minor", length=3, width=1, direction="in")

    # for c in confiles:
    #     contours = ascii.read(c)
    #     xray_ra, xray_dec = gc.world2pixel(contours['col1'], contours["col2"])
    #     # Eyeballed. DEAL WITH IT
    #     pyplot.plot(xray_ra - xlen_obs_pix+75, xray_dec - ylen_obs_pix, "w", lw=1)

    delta = 1
    x = numpy.arange(0, xlen_obs_pix, delta)
    y = numpy.arange(0, ylen_obs_pix, delta)
    X, Y = numpy.meshgrid(x, y)
    CS = pyplot.contour(X, Y, numpy.log10(contour_smooth.clip(10**-8.8)), 7,
        colors="black", linestyles="solid", ls=4)
    pyplot.clabel(CS, fontsize=16, inline=1, colors="black")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    pyplot.tight_layout()
    pyplot.savefig("out/SZ_prediction_XE={0}_i={1}.pdf".format(0.25, 45), dpi=300)  # Yes, hardcoded. DEAL WITH IT
    pyplot.close()

    # import main
    # a = main.new_argument_parser().parse_args()
    # a.do_cut = False; a.clustername = "both"
    # cygA, cygNW = main.set_observed_clusters(a)



def find_bestfit_snapshots(verbose=False):
    # Find bestfit betamodel snapshots, coarse
    tofind = [765/numpy.cos(numpy.pi*i/180) for i in range(75, -15, -15)]

    from simulation import Simulation
    sim50 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0905", name="both",
                     set_data=False)
    sim75 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0906", name="both",
                     set_data=False)
    sim25 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0907", name="both",
                     set_data=False)

    from main import compute_distance
    i, prev_distance, prev_snapnr = 0, 4000, 0
    for sim in [sim75]:
        sim.set_gadget_paths(verbose=verbose)
        # sim = copy.deepcopy(sim)

        for snapnr, path_to_snaphot in enumerate(sim.gadget.snapshots):
            snapnr = int(path_to_snaphot[-3:])
            distance = compute_distance(sim, snapnr, path_to_snaphot, verbose=verbose) #.get()
            if distance == numpy.nan or not distance: continue

            # We want to find the last snapshot with larger core separation than
            # observed because then we can integrate from there on to gain finer
            # interpolation
            if distance < tofind[i]:
                print prev_snapnr, prev_distance
                i += 1
            if i >= len(tofind): break
            prev_distance = distance
            prev_snapnr = snapnr


def build_matrix(residuals=False, residuals_minmax=100):
    # Open observation lss [counts/s/arcsec^2]
    lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
    lss_Lx = lss+".dir/Frame1/cygnus_lss_fill_flux.fits"
    lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"

    mosaic_Lx = fits.open(lss_Lx)
    mosaic_kT = fits.open(lss_kT)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    mosaic_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_Lx[0].data, 9)
    temperature_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_kT[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic_Lx[0].data.max()
    maxcounts_obs_index = mosaic_Lx[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic_Lx[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic_Lx[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic_Lx[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic_Lx[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    fig = pyplot.figure(figsize=(12, 15))
    axes = []
    for y in range(6):
        for x in range(6):
            ax = pyplot.subplot2grid((7, 6), (y, x))
            ax.set_xticks([], []); ax.set_yticks([], [])
            axes.append(ax)

    cax_left = fig.add_axes([0.05, 0.11, 0.4, 0.03])
    cax_left.set_xticks([], []); cax_left.set_yticks([], [])
    cax_right = fig.add_axes([0.55, 0.11, 0.4, 0.03])
    cax_right.set_xticks([], []); cax_right.set_yticks([], [])

    # pyplot.show()
    # return

    from simulation import Simulation
    sim50 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0905", name="both",
                     set_data=False)
    sim75 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0906", name="both",
                     set_data=False)
    sim25 = Simulation(base="/Volumes/Cygnus/timoh", timestamp="20170115T0907", name="both",
                     set_data=False)

    for Xe_i, sim in enumerate([sim25, sim50, sim75]):
        # sim.read_ics()
        # sim.set_gadget_paths()
        sim.read_smac(verbose=True)
        sim.nsnaps, sim.xlen, sim.ylen = sim.psmac.xray0.shape
        sim.pixelscale = float(sim.psmac.xray0_header["XYSize"])/int(sim.xlen)

        for EA2_i, inclination in enumerate([0, 15, 30, 45, 60, 75]):
            data = getattr(sim.psmac, "xray{0}best765".format(inclination))
            header = getattr(sim.psmac, "xray{0}best765_header".format(inclination))

            maxcounts_sim = data.max()
            maxcounts_sim_index = data.argmax()  # of flattened array
            ylen_sim_pix, xlen_sim_pix = data[0].shape
            ylen_sim_kpc = xlen_sim_kpc = float(header["XYSize"])
            xcenter_sim = maxcounts_sim_index % xlen_sim_pix
            ycenter_sim = maxcounts_sim_index / xlen_sim_pix
            pix2kpc_sim = float(header["XYSize"])/int(header["XYPix"])

            cA, cNW, distance = sim.find_cluster_centroids_psmac_dmrho(
                snapnr=0, EA2=inclination)

            print "Smac Cube, i =", inclination
            print "  Shape ({0},   {1})   pixels".format(xlen_sim_pix, ylen_sim_pix)
            print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_sim_kpc, ylen_sim_kpc)
            print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_sim, ycenter_sim, maxcounts_sim)
            print "  Core Separaton = {0:3.2f}".format(distance)

            # Cut relevant part from the simulation
            desired_xlen_sim_kpc = xlen_obs_kpc
            desired_ylen_sim_kpc = ylen_obs_kpc
            desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
            desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
            xoffset = int((xcenter_sim * pix2kpc_sim -
                xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
            yoffset = int((ycenter_sim * pix2kpc_sim -
                ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

            equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                                 xoffset: xoffset+desired_xlen_sim_pix]

            # Convolve with 2D Gaussian, radius converted to kpc in simulation
            # from a 9 pixel radius in the Chandra observation
            smooth_obs_kpc = 9 * pix2kpc_obs * arcsec2kpc
            smooth_sim_kpc = smooth_obs_kpc / pix2kpc_sim
            smaccube_smooth = scipy.ndimage.filters.gaussian_filter(
                equal_boxsize_kpc_smaccube, smooth_sim_kpc)

            # central value of CygNW observation / central value of simulated CygNW in 0th snapshot
            magic = 5.82e-9 / 1.77e-5   # for 20170115T0905_xray_45_best.fits
            # magic = 5.82e-9 / 2.788e-5   # for 20170115T0905_xray_0_best.fits
            smaccube_smooth *= magic

            if not residuals:
                if Xe_i == 2 and EA2_i == 5:
                    # Show the observation
                    Lx = axes[6*EA2_i+Xe_i].imshow(mosaic_smooth, vmin=5.0e-10, vmax=1.0e-7,
                        norm=matplotlib.colors.LogNorm(), origin="lower",
                        cmap=colorcet.cm["linear_bmw_5_95_c86"],
                        extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
                else:
                    # Display the cut-out, zoomed-in, correctly smoothed Smac Cube
                    Lx = axes[6*EA2_i+Xe_i].imshow(smaccube_smooth, vmin=5.0e-10, vmax=1.0e-7,
                        norm=matplotlib.colors.LogNorm(), origin="lower", cmap=colorcet.cm["linear_bmw_5_95_c86"],
                        extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
            else:
                zoomx = float(ylen_obs_pix) / smaccube_smooth.shape[0]
                zoomy = float(xlen_obs_pix) / smaccube_smooth.shape[1]
                shape_matched = scipy.ndimage.zoom(smaccube_smooth,  [zoomx, zoomy], order=3)

                Lx = axes[6*EA2_i+Xe_i].imshow(
                    100*(mosaic_smooth-shape_matched)/shape_matched,
                    vmin=-residuals_minmax, vmax=residuals_minmax, origin="lower",
                    # cmap=colorcet.cm["diverging_gwv_55_95_c39_r"],
                    cmap=colorcet.cm["diverging_bwr_40_95_c42"],
                    extent=[0, xlen_obs_pix, 0, ylen_obs_pix]
                )

            data = getattr(sim.psmac, "tspec{0}best765".format(inclination))
            equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                                 xoffset: xoffset+desired_xlen_sim_pix]
            tspec = convert.K_to_keV(equal_boxsize_kpc_smaccube)
            if not residuals:
                if Xe_i == 2 and EA2_i == 5:
                    # Show the observation
                    kT = axes[3+6*EA2_i+Xe_i].imshow(temperature_smooth, vmin=3.5, vmax=12,
                        origin="lower", cmap=colorcet.cm["linear_kryw_5_100_c67"],
                        extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
                else:
                    # Display the cut-out, zoomed-in, correctly smoothed Smac Cube
                    kT = axes[3+6*EA2_i+Xe_i].imshow(tspec, vmin=3.5, vmax=12,
                        origin="lower", cmap=colorcet.cm["linear_kryw_5_100_c67"],
                        extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
            else:
                zoomx = float(ylen_obs_pix) / tspec.shape[0]
                zoomy = float(xlen_obs_pix) / tspec.shape[1]
                shape_matched = scipy.ndimage.zoom(tspec,  [zoomx, zoomy], order=3)
                kT = axes[3+6*EA2_i+Xe_i].imshow(
                    100*(temperature_smooth-shape_matched)/shape_matched,
                    vmin=-residuals_minmax, vmax=residuals_minmax, origin="lower",
                    # cmap=colorcet.cm["diverging_gwv_55_95_c39_r"],
                    cmap=colorcet.cm["diverging_bwr_40_95_c42"],
                    extent=[0, xlen_obs_pix, 0, ylen_obs_pix]
                )

                fig = pyplot.gcf().number

                if (Xe_i == 0 and EA2_i == 2) or (Xe_i == 0 and EA2_i == 3):
                    pyplot.figure()

                    ax = pyplot.gca()
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticklabels("")
                    ax.set_yticklabels("")

                    kT = pyplot.imshow(tspec, vmin=3.5, vmax=12,
                        origin="lower", cmap=colorcet.cm["linear_kryw_5_100_c67"],
                        extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
                    cax = pyplot.colorbar(kT, ax=ax, shrink=0.45, pad=0.03,
                        aspect=12, orientation="horizontal")
                    cax.ax.xaxis.set_ticks_position("both")
                    cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16, direction="in")
                    cax.ax.set_xlabel(r"Temperature [keV]", fontsize=18)
                    cax.ax.tick_params(which="minor", length=3, width=1, direction="in")
                    pyplot.tight_layout()
                    pyplot.savefig("out/kT_XE={0}_i={1}.pdf".format(0.25*(Xe_i+1), 15*EA2_i), dpi=300)
                    pyplot.close()

                    pyplot.figure()

                    ax = pyplot.gca()
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticklabels("")
                    ax.set_yticklabels("")

                    kT_residuals = pyplot.imshow(
                        100*(temperature_smooth-shape_matched)/shape_matched,
                        vmin=-residuals_minmax, vmax=residuals_minmax, origin="lower",
                        # cmap=colorcet.cm["diverging_gwv_55_95_c39_r"],
                        cmap=colorcet.cm["diverging_bwr_40_95_c42"],
                        extent=[0, xlen_obs_pix, 0, ylen_obs_pix]
                    )
                    cax = pyplot.colorbar(kT_residuals, ax=ax, shrink=0.35, pad=0.03,
                        aspect=12, orientation="horizontal")
                    cax.ax.xaxis.set_ticks_position("both")
                    cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16, direction="in")
                    cax.ax.set_xlabel(r"Temperature Residuals [\%]" , fontsize=18)
                    cax.ax.tick_params(which="minor", length=3, width=1, direction="in")
                    pyplot.tight_layout()
                    pyplot.savefig("out/kT_residuals_XE={0}_i={1}.pdf".format(0.25*(Xe_i+1), 15*EA2_i), dpi=300)
                    pyplot.close()

                kT_residuals = 100*(temperature_smooth-shape_matched)/shape_matched
                pyplot.figure()
                pyplot.hist(kT_residuals.flat, bins=512, normed=True)
                pyplot.xlim(-75, 75)
                pyplot.title(r"$X_E = {0} \quad i = {1}$".format(0.25*(Xe_i+1), 15*EA2_i))
                pyplot.savefig("out/temperature_residuals_histogram_XE={0}_i={1}.pdf".format(0.25*(Xe_i+1), 15*EA2_i), dpi=300)
                pyplot.close()

                pyplot.figure(fig)


            if Xe_i is 0 and EA2_i is 0:
                # Colorbar for X-ray Surface Brightness
                cax = pyplot.colorbar(Lx, cax=cax_left, orientation="horizontal")
                if not residuals:
                    cax.ax.xaxis.set_ticks_position("both")
                    cax.ax.tick_params(axis="both", which="major", length=6, width=1, labelsize=16, direction="in")
                    cax.ax.set_xlabel(r"X-ray Surface Brightness $\left[\frac{\mathrm{counts}}{\mathrm{cm}^2 \, \mathrm{s}} \right]$", fontsize=18)
                    cax.set_ticks([1e-9, 1e-8, 1e-7])
                    cax.set_ticklabels(["$10^{-9}$", "$10^{-8}$", "$10^{-7}$"])
                    minorticks = Lx.norm(numpy.hstack([numpy.arange(5, 10, 1)/1e10,
                        numpy.arange(2, 10, 1)/1e9, numpy.arange(2, 10, 1)/1e8]))
                    cax.ax.tick_params(which="minor", length=3, width=1, direction="in")
                    cax.ax.xaxis.set_ticks(minorticks, minor=True)
                else:
                    cax.ax.xaxis.set_ticks_position("both")
                    cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16, direction="in")
                    cax.ax.set_xlabel(r"X-ray Surface Brightness [\%]", fontsize=18)
                    cax.ax.tick_params(which="minor", length=3, width=1, direction="in")

                # Colorbar for Spectroscopic Temperature
                cax = pyplot.colorbar(kT, cax=cax_right, orientation="horizontal")
                cax_right.xaxis.set_ticks_position("both")
                cax_right.tick_params(axis="both", length=6, width=1, labelsize=16, direction="in")
                cax_right.set_xlabel(r"Temperature [keV]" if not residuals else r"Temperature [\%]" , fontsize=18)
                cax_right.tick_params(which="minor", length=3, width=1, direction="in")

    for xE, ax in zip([0.25, 0.50, 0.75, 0.25, 0.50, 0.75], axes[0:6]):
        ax.text(0.02, 0.95, "$X_E={0:.2f}$".format(xE), color="white", fontsize=22,
                ha="left", va="top", transform=ax.transAxes)

    for EA2, ax in zip([0, 15, 30, 45, 60, 75], axes[::6]):
        ax.text(0.02, 0.02, "$i={0:02d}$".format(EA2),
                color="white", fontsize=22, ha="left", va="bottom", transform=ax.transAxes)
        # axes[n].text(0.5, 0.5, str(n), transform=axes[n].transAxes)

    # Green border around bestfit
    for ax in [axes[18], axes[21]]:
        ax.set_zorder(10)  # 'bring to front'
        for s in ax.spines.values():
            s.set_color("lawngreen")
            s.set_linewidth(4.0)

    # White border around observation
    for ax in [axes[-1], axes[-4]]:
        ax.set_zorder(10)  # 'bring to front'
        for s in ax.spines.values():
            s.set_color("darkgreen")
            s.set_linewidth(4.0)

    pyplot.subplots_adjust(left=0., bottom=0.02, right=1., top=1., wspace=0., hspace=0.01)
    pyplot.savefig("out/matrix{0}.png".format("_residuals_"+str(residuals_minmax) if residuals else ""), pdi=6000)
    pyplot.savefig("out/matrix{0}.pdf".format("_residuals_"+str(residuals_minmax) if residuals else ""), pdi=6000)
    pyplot.close(fig)



def plot_residuals():
    # Open observation lss [counts/s/arcsec^2]
    obs = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    mosaic = fits.open(obs)

    # Convolve with 2D Gaussian, 9 pixel smoothing to ensure CygNW is visible
    mosaic_smooth = scipy.ndimage.filters.gaussian_filter(mosaic[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic[0].data.max()
    maxcounts_obs_index = mosaic[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic[0].header["NAXIS2"]
    pix2arcsec_obs = mosaic[0].header["CDELT2"]*3600  # Chandra size of pixel 0.492". Value in header is in degrees.
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)
    # Open simulation Lx Smac Cube [erg/cm^2/s/Hz]
    import parse
    sim = "/Volumes/Cygnus/timoh/runs/20170115T0905/analysis/20170115T0905_xray_45_best.fits.fz"
    header, data = parse.psmac2_fitsfile(sim)

    # Find the centroid of "CygA" to align simulation and observation later on
    maxcounts_sim = data.max()
    maxcounts_sim_index = data.argmax()  # of flattened array
    ylen_sim_pix, xlen_sim_pix = data[0].shape
    ylen_sim_kpc = xlen_sim_kpc = float(header["XYSize"])
    xcenter_sim = maxcounts_sim_index % xlen_sim_pix
    ycenter_sim = maxcounts_sim_index / xlen_sim_pix
    pix2kpc_sim = float(header["XYSize"])/int(header["XYPix"])

    print "Smac Cube"
    print "  Shape ({0},   {1})   pixels".format(xlen_sim_pix, ylen_sim_pix)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_sim_kpc, ylen_sim_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_sim, ycenter_sim, maxcounts_sim)

    # Cut relevant part from the simulation
    desired_xlen_sim_kpc = xlen_obs_kpc
    desired_ylen_sim_kpc = ylen_obs_kpc
    desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
    desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
    xoffset = int((xcenter_sim * pix2kpc_sim - xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
    yoffset = int((ycenter_sim * pix2kpc_sim - ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

    equal_boxsize_kpc_smaccube = data[0][yoffset:yoffset+desired_ylen_sim_pix,
                                         xoffset: xoffset+desired_xlen_sim_pix]

    # Convolve with 2D Gaussian, radius converted to kpc in simulation
    # from a 9 pixel radius in the Chandra observation
    smooth_obs_kpc = 9 * pix2kpc_obs * arcsec2kpc
    smooth_sim_kpc = smooth_obs_kpc / pix2kpc_sim
    smaccube_smooth = scipy.ndimage.filters.gaussian_filter(equal_boxsize_kpc_smaccube, smooth_sim_kpc)

    # central value of CygNW observation / central value of simulated CygNW in 0th snapshot
    magic = 5.82e-9 / 1.77e-5   # for 20170115T0905_xray_45_best.fits
    # magic = 5.82e-9 / 2.788e-5   # for 20170115T0905_xray_0_best.fits
    smaccube_smooth *= magic

    fig, (ax0, ax1, ax2) = pyplot.subplots(1, 3, figsize=(12, 9))

    im = ax0.imshow(mosaic_smooth, vmin=5.0e-10, vmax=1.0e-7,
                    norm=matplotlib.colors.LogNorm(), origin="lower",
                    cmap=colorcet.cm["linear_bmw_5_95_c86"])
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])

    # Display the cut-out, zoomed-in, correctly smoothed Smac Cube
    im = ax1.imshow(smaccube_smooth, vmin=5.0e-10, vmax=1.0e-7,
                    norm=matplotlib.colors.LogNorm(), origin="lower",
                    cmap=colorcet.cm["linear_bmw_5_95_c86"],
                    extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])

    # Create residuals plot
    zoomx = float(ylen_obs_pix) / smaccube_smooth.shape[0]
    zoomy = float(xlen_obs_pix) / smaccube_smooth.shape[1]
    shape_matched = scipy.ndimage.zoom(smaccube_smooth,  [zoomx, zoomy], order=3)

    im = ax2.imshow(100*(mosaic_smooth-shape_matched)/shape_matched,
                    vmin=-100, vmax=100, origin="lower",
                    # cmap=colorcet.cm["diverging_gwv_55_95_c39_r"],
                    cmap=colorcet.cm["diverging_bwr_40_95_c42"],
                    extent=[0, xlen_obs_pix, 0, ylen_obs_pix])
    im.axes.set_xticks([], [])
    im.axes.set_yticks([], [])

    # pyplot.switch_backend("Agg")
    # pyplot.savefig("out/Lx_45_with_residuals.pdf", dpi=300)


if __name__ == "__main__":
    to_plot = [ 6 ]

    # Coordinates of the CygA and CygNW centroids
    cygA = ( 299.8669, 40.734496 )
    cygNW = ( 299.7055, 40.884849 )

    radio = "../runs/RadioObservation/radio5GHz.fits"
    mosaic = "../runs/ChandraObservation/xray/cygnus_tot_flux.fits"
    lss = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    kT = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    lss_kT = "../runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck.dir/Frame2/working_spectra_kT_map.fits"

    if 1 in to_plot:
        pyplot.rcParams.update( { "text.usetex": False, "font.size": 18 } )
        plot_mosaic(lss, cygA, cygNW, is_lss=True)
        plot_mosaic(lss_kT, cygA, cygNW, is_lss=False, is_kT=True)
        pyplot.rcParams.update( { "text.usetex": True, "font.size": 28 } )

    if 2 in to_plot:
        a = main.new_argument_parser().parse_args()
        a.do_cut = False
        cygA_uncut, cygNW_uncut = main.set_observed_clusters(a)
        cygA_mask = cygA_uncut.avg.mask
        cygA_uncut.avg.mask = [False for i in range(len(cygA_uncut.avg.columns))]
        bestfit_betamodel(cygA_uncut)
        cygA_uncut.avg.mask = cygA_mask

    if 2 in to_plot:
        cygNW_mask = cygNW_uncut.avg.mask
        cygNW_uncut.avg.mask = [False for i in range(len(cygNW_uncut.avg.columns))]
        bestfit_betamodel(cygNW_uncut)
        cygNW_uncut.avg.mask = cygNW_mask

    if 3 in to_plot:
        a = main.new_argument_parser().parse_args()
        a.do_cut = False
        cygA_uncut, cygNW_uncut = main.set_observed_clusters(a)
        a.do_cut = True
        cygA_cut, cygNW_cut = main.set_observed_clusters(a)
        plot_mass_ratio(cygA_uncut, cygNW_uncut, cut=False)
        plot_mass_ratio(cygA_cut, cygNW_cut, cut=True)

    if 5 in to_plot:
        show_puffup()

    if 1338 in to_plot:
        find_bestfit_snapshots()

    if 6 in to_plot:
        build_matrix()

    if 7 in to_plot:
        # build_matrix(residuals=True, residuals_minmax=25)
        build_matrix(residuals=True, residuals_minmax=50)
        # build_matrix(residuals=True, residuals_minmax=75)
        # build_matrix(residuals=True, residuals_minmax=100)
        # build_matrix(residuals=True, residuals_minmax=125)
        # build_matrix(residuals=True, residuals_minmax=150)

    if 8 in to_plot:
        plot_simulated_wedges()

    if 9 in to_plot:
        plot_compton_y()

    # Appendix
    if 1337 in to_plot:
        pyplot.switch_backend('agg')
        a, unknown = main.new_argument_parser().parse_known_args()
        a.do_cut = True
        a.basedir = "/Volumes/Cygnus/timoh/"
        a.timestamp = "20170115T0905"
        a.clustername = "both"
        cygA_cut, cygNW_cut = main.set_observed_clusters(a)
        sim = Simulation(base=a.basedir, name=a.clustername, timestamp=a.timestamp, set_data=False)
        sim.read_ics(verbose=True)

        sim.toy.halo0.name = "cygA"
        fignum = plot.donnert2014_figure1(cygA_cut, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, sim.toy.halo0, sim.outdir)

        sim.toy.halo1.name = "cygNW"
        fignum = plot.donnert2014_figure1(cygNW_cut, add_sim=True, verlinde=False)
        plot.add_sim_to_donnert2014_figure1(fignum, sim.toy.halo1, sim.outdir)

        # 30 MB down to ~3 MB, still looks vectorized
        # gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -sOutputFile=cygA_donnert2014figure1_reduced.pdf cygA_donnert2014figure1.pdf
        # gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -sOutputFile=cygNW_donnert2014figure1_reduced.pdf cygNW_donnert2014figure1.pdf
