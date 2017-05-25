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


def plot_mosaic(mosaic, cygA, cygNW, is_lss=False):
    """ Zoomin of the merger region with distance measure
        @param mosaic: path to the Chandra x-ray mosaic fits file
        @param cygA  : tuple with RA, dec of CygA centroid
        @param cygNW : tuple with RA, dec of CygNW centroid
        @param is_lss: bool to indicate full mosaic or zoom-in (lss) """


    gc = aplpy.FITSFigure(mosaic)

    # Add smoothed log-stretch of the entire mosaic
    # matplotlib.cm.register_cmap(name='BuPu_9_r', cmap=palettable.colorbrewer.diverging.PiYG_7_r.mpl_colormap)
    gc.show_colorscale(vmin=7.0e-10, vmax=1.0e-6, stretch="log",
        cmap="spectral", smooth=9)

    ax = pyplot.gca()

    # Find the pixels of the centroids
    cygA_x, cygA_y = gc.world2pixel(cygA[0], cygA[1])
    cygNW_x, cygNW_y = gc.world2pixel(cygNW[0], cygNW[1])

    # Add scale. Length is 500 kpc after unit conversions
    gc.add_scalebar(0.13227513)
    gc.scalebar.set_corner("bottom left" if is_lss else "bottom right")
    gc.scalebar.set_length(0.1)
    gc.scalebar.set_linewidth(4)
    gc.scalebar.set_font_size(22 if is_lss else 18)
    gc.scalebar.set_label("500 kpc")
    gc.scalebar.set_color("white")

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm")
    gc.tick_labels.set_yformat("dd:mm")
    gc.frame.set_color("white")

    ax.tick_params(axis="both", which="both", reset=True, color="w", labelcolor="k",
        pad=8, width=2, size=4, direction="in", top="on", right="on")
    ax.tick_params(axis="both", which="major", size=8)

    # CygA and CygNW label
    ax.text(cygA_x, 0.80*cygA_y if not is_lss else 0.60*cygA_y, "CygA",
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
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels("")
        ax.set_yticklabels("")

        cax = pyplot.colorbar(gc.image, ax=ax, shrink=0.45, pad=0.03,
                              aspect=12, orientation="horizontal")
        cax.ax.xaxis.set_ticks_position("both")
        cax.ax.tick_params(axis="both", length=6, width=1, labelsize=16)
        cax.ax.set_xlabel(r"X-ray Surface Brightness $\left[\frac{\mathrm{counts}}{\mathrm{cm}^2 \, \mathrm{s}} \right]$", fontsize=16)

        # Yes we must set ticks manually... +_+
        cax.set_ticks([1e-9, 1e-8, 1e-7, 1e-6])
        cax.set_ticklabels(["$10^{-9}$", "$10^{-8}$", "$10^{-7}$", "$10^{-6}$"])
        cax.ax.minorticks_on()
        minorticks = gc.image.norm(numpy.hstack([numpy.arange(2, 10, 1)/1e9,
            numpy.arange(2, 10, 1)/1e8, numpy.arange(2, 10, 1)/1e7]))
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
    else:
        # Annotate top and low left corner with observation details
        ax.text(0.5, 0.98, "Chandra X-ray Surface Brightness", weight="bold",
                   fontsize=22, color="white", ha="center", va="top", transform=ax.transAxes)
        ax.text(0.05, 0.05, "ACIS-I Mosaic\n0.5-7.0 keV\n1.03 Msec total exposure", color="white",
                   fontsize=18, ha="left", va="bottom", transform=ax.transAxes)
        for label in ax.get_ymajorticklabels() + ax.get_yminorticklabels():
            label.set_rotation_mode("anchor")
            label.set_rotation(90)
            label.set_horizontalalignment("center")

    # For some reason shows up double...
    ax.tick_params(axis="both", which="both", top="off", right="off")

    pyplot.tight_layout()
    gc.save("out/mosaic_xray{0}.png".format("_lss" if is_lss else ""), dpi=300)
    gc.save("out/mosaic_xray{0}.pdf".format("_lss" if is_lss else ""), dpi=300)


def bestfit_betamodel(c):
    """ Plot best-fit betamodel with residuals """

    # Define kwargs for pyplot to set up style of the plot
    avg = { "marker": "o", "ls": "", "c": "b" if c.name == "cygA" else "b",
            "ms": 4, "alpha": 1, "elinewidth": 2,
            "label": "1.03 Msec Chandra\n(Wise+ in prep)" }
    fit = { "color": "k", "lw": 4, "linestyle": "solid" }

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
    pyplot.plot(cygA.HE_radii * convert.cm2kpc, hydrostatic, c="k", ls="--", label="hydrostatic")

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
            fontsize=22, transform=trans)
    ax.text(0.97*cygNW.halo["r500"], 0.98, r"$r_{500}$", ha="right", va="top",
            fontsize=22, transform=trans)

    pyplot.xscale("log")
    # pyplot.xlim(200, 1e4)
    # pyplot.ylim(0.5, 2.5)
    pyplot.xlim(60, 1.1*cygA.halo["r200"])
    pyplot.ylim(0, 4)
    pyplot.xlabel("Radius [kpc]")
    pyplot.ylabel("Mass Ratio [CygA/CygNW]")
    pyplot.legend(loc="lower left", fontsize=22)

    ax.tick_params(axis="both", which="both", top="on", right="on")
    pyplot.tight_layout()
    pyplot.savefig("out/mass_ratio_{0}cut.pdf".format("" if cut else "un"), dpi=600)


if __name__ == "__main__":
    to_plot = [ 3, 4, 5 ]

    # Coordinates of the CygA and CygNW centroids
    cygA = ( 299.8669, 40.734496 )
    cygNW = ( 299.7055, 40.884849 )

    radio = "../runs/RadioObservation/radio5GHz.fits"
    mosaic = "../runs/ChandraObservation/xray/cygnus_tot_flux.fits"
    lss = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"

    adjustable_cmap(lss)
    import sys; sys.exit(0)

    if 1 in to_plot:
        pyplot.rcParams.update( { "text.usetex": False, "font.size": 18 } )
        plot_mosaic(mosaic, cygA, cygNW, is_lss=False)
        pyplot.rcParams.update( { "text.usetex": True, "font.size": 28 } )

    if 2 in to_plot:
        pyplot.rcParams.update( { "text.usetex": True, "font.size": 18 } )
        # pyplot.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
        plot_mosaic(lss, cygA, cygNW, is_lss=True)
        pyplot.rcParams.update( { "text.usetex": True, "font.size": 28 } )

    if 3 in to_plot:
        a = main.new_argument_parser().parse_args()
        a.do_cut = False
        cygA_uncut, cygNW_uncut = main.set_observed_clusters(a)
        cygA_mask = cygA_uncut.avg.mask
        cygA_uncut.avg.mask = [False for i in range(len(cygA_uncut.avg.columns))]
        bestfit_betamodel(cygA_uncut)
        cygA_uncut.avg.mask = cygA_mask

    if 4 in to_plot:
        cygNW_mask = cygNW_uncut.avg.mask
        cygNW_uncut.avg.mask = [False for i in range(len(cygNW_uncut.avg.columns))]
        bestfit_betamodel(cygNW_uncut)
        cygNW_uncut.avg.mask = cygNW_mask

    if 5 in to_plot:
        a = main.new_argument_parser().parse_args()
        a.do_cut = False
        cygA_uncut, cygNW_uncut = main.set_observed_clusters(a)
        a.do_cut = True
        cygA_cut, cygNW_cut = main.set_observed_clusters(a)
        plot_mass_ratio(cygA_uncut, cygNW_uncut, cut=False)
        plot_mass_ratio(cygA_cut, cygNW_cut, cut=True)

    if 6 in to_plot:
        a = main.new_argument_parser().parse_args()
        a.do_cut = True
        a.basedir = "/media/SURFlisa/"
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


