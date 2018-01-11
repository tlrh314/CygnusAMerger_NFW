# -*- coding: utf-8 -*-

import numpy
import scipy
from scipy import ndimage
import aplpy
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

import parse
from simulation import Simulation

from plotsettings import PlotSettings
style = PlotSettings()


def plot_xray_simulation(Lx, obs=False):
    pyplot.style.use(["dark_background"])

    sigma = 9  # pixel radius

    if not obs:
        # deg per pixel, strange to use this factor. TODO: get correct units
        obscure_observer_unit = 1.366e-4
    else:
        obscure_observer_unit = 1
    Lx_smoothed = scipy.ndimage.filters.gaussian_filter(obscure_observer_unit*Lx, sigma)

    fig, ax = pyplot.subplots(figsize=(16, 16))
    # https://stackoverflow.com/questions/32462881
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    im = ax.imshow(numpy.log10(Lx_smoothed), origin="lower", cmap="spectral",
                   vmin=numpy.log10(7.0e-10), vmax=numpy.log10(1.0e-6))
    fig.colorbar(im, cax=cax, orientation="vertical")

    if not obs:
        snapnr = 119
        TimeBetSnap = 0.01
        fig.suptitle("T = {0:04.2f} Gyr. snapnr = {1:03d}. distance = 639"
            .format(snapnr*TimeBetSnap, snapnr), color="white", size=26, y=0.9)

        xlen = ylen = 2048
        xstart = ystart = 750
        pixelscale = 3.02
        scale = int(xlen - xstart)*pixelscale
        scale = "[{0:.1f} Mpc]^2".format(float(scale)/1000)
        pad = 16
        ax.text(xstart + 2*pad, ystart + pad, scale, color="white",  size=18,
                horizontalalignment="left", verticalalignment="bottom")

        # ax.set_xlim(xstart, xlen - xstart)
        # ax.set_ylim(ystart, ylen - ystart)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect("equal")
    fig.tight_layout()
    # fig.savefig(outdir+"tspec_{0:03d}.png".format(i), dpi=300)
    # pyplot.close(fig)
    return fig


if __name__ == "__main__":
    mosaic = "../runs/ChandraObservation/xray/cygnus_tot_flux.fits"
    lss = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    xraysim = "../runs/ChandraObservation/lss/20170115T0905_xray_0_best.fits.fz"

    # World coordinates, eyeballed from ds9
    cxA, cyA = 299.8669, 40.734496
    cxNW, cyNW = 299.7055, 40.884849

    # gc = aplpy.FITSFigure(lss)
    # cxA, cyA = gc.world2pixel(cxA, cyA)
    # cxNW, cyNW = gc.world2pixel(cxNW, cyNW)
    cxA, cyA = 672.630553335, 581.29563288
    cxNW, cyNW = 1565.53569891, 1565.53569891

    simheader, simdata = parse.psmac2_fitsfile(xraysim)
    simLx = simdata[0]  # P-Smac Fits cube of bestfit snap has shape (1, Xpix, Ypix)
    lssheader, lssdata = parse.psmac2_fitsfile(lss)

    print "lss dimensions [pixel]", lssdata.shape
    print "centroid CygA [pixel]", cxA, cyA
    print "centroid CygNW [pixel]", cxNW, cxNW

    figsim = plot_xray_simulation(simLx)
    figobs = plot_xray_simulation(lssdata, obs=True)
    pyplot.show()

