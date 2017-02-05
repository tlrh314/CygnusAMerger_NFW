# -*- coding: utf-8 -*-

import glob
import numpy
import scipy
import astropy
from astropy.io import ascii
import astropy.units as u
from scipy.stats import kde
from matplotlib import pyplot
import aplpy
import sphviewer
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = None


def plot_zoomin_of_core(mosaic, radio, cygA):
    """ Zoomin of the AGN feedback core region where FR-II interacts with gas
        @param mosaic: path to the Chandra x-ray mosaic fits file
        @param radio :  path to JVLA 5GHz fits (1984ApJ...285L..35P)
        @param cygA  : tuple with RA, dec of CygA centroid """
    gc = aplpy.FITSFigure(mosaic)

    # Show xray observation with stretch to highlight the core
    gc.show_colorscale(vmin=8.0e-10, vmax=2.0e-6,
                       stretch="log", cmap="spectral")

    # Add the famous 5GHz radio contours (Perley, Dreher & Cowan 1984)
    gc.show_contour(radio, vmin=0.002, vmax=0.1, levels=15, smooth=1,
                    colors="black", lw=8)

    # Show a scale bar of 30 kpc after unit conversions
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    kpc2arcsec = 1/cc.kpc_DA
    gc.add_scalebar(30 * kpc2arcsec * u.arcsecond)
    gc.scalebar.set_corner("bottom right")
    gc.scalebar.set_linewidth(4)
    gc.scalebar.set_label("30 kpc")
    gc.scalebar.set_color("black")

    # Zoom in on the central region
    gc.recenter(cygA[0], cygA[1], width=0.037, height=0.018)
    pyplot.gca().tick_params(axis="both", which="both", colors="k", reset=True)

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm:ss")

    ax = pyplot.gca()
    ax.tick_params(axis="both", which="minor", colors="k",
                   pad=8, width=2, size=4, reset=True)
    ax.tick_params(axis="both", which="major", colors="k",
                   pad=8, width=2, size=8, reset=True)

    # gc.add_colorbar()
    # gc.colorbar.set_pad(0.1)

    pyplot.tight_layout()
    gc.save("out/CygA_Radio_5GHz.pdf", dpi=300)


def plot_mosaic_with_ruler(mosaic, cygA, cygNW):
    """ Zoomin of the merger region with distance measure
        @param mosaic: path to the Chandra x-ray mosaic fits file
        @param cygA  : tuple with RA, dec of CygA centroid
        @param cygNW : tuple with RA, dec of CygNW centroid """
    gc = aplpy.FITSFigure(mosaic)

    # Add smoothed log-stretch of the entire mosaic
    gc.show_colorscale(vmin=7.0e-10, vmax=1.0e-6, stretch="log",
                       cmap="spectral", smooth=9)

    # Add scale. Length is 500 kpc after unit conversions
    gc.add_scalebar(0.13227513)
    gc.scalebar.set_corner("bottom right")
    gc.scalebar.set_length(0.1)
    gc.scalebar.set_linewidth(4)
    gc.scalebar.set_label("500 kpc")
    gc.scalebar.set_color("white")

    # Find the pixels of the centroids
    cygA_x, cygA_y = gc.world2pixel(cygA[0], cygA[1])
    cygNW_x, cygNW_y = gc.world2pixel(cygNW[0], cygNW[1])

    ax = pyplot.gca()
    ax.plot([cygA_x, cygNW_x], [cygA_y, cygNW_y], c="w", lw=1)

    # Eyeballed coordinates in ds9 :-) ...
    text_x, text_y = gc.world2pixel( 299.78952, 40.816273 )
    ax.text(text_x, text_y, '700.621"', ha="center", va="center", color="white",
            rotation=51, weight="bold", fontsize=22)

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm:ss")

    ax.tick_params(axis="both", which="minor", colors="k",
                   pad=8, width=2, size=4, reset=True)
    ax.tick_params(axis="both", which="major", colors="k",
                   pad=8, width=2, size=8, reset=True)

    # Zoom in a bit more on the merger region
    gc.recenter(299.78952, 40.81, width=0.185, height=0.185)

    pyplot.tight_layout()
    gc.save("out/mosaic_xray_ruler.pdf", dpi=300)


def plot_mosaic_with_wedges(mosaic, cygA):
    """ Plot the merger, hot, cold regions in smoothed mosaic
        @param mosaic: path to the Chandra x-ray mosaic fits file
        @param cygA  : tuple with RA, dec of CygA centroid """
    gc = aplpy.FITSFigure(mosaic)

    # Add smoothed log-stretch of the entire mosaic
    gc.show_colorscale(vmin=7.0e-10, vmax=4.0e-8, stretch="log",
                       cmap="spectral", smooth=9)

    # Add scale. Length is 500 kpc after unit conversions
    gc.add_scalebar(0.13227513)
    gc.scalebar.set_corner("bottom right")
    gc.scalebar.set_length(0.1)
    gc.scalebar.set_linewidth(4)
    gc.scalebar.set_label("500 kpc")
    gc.scalebar.set_color("white")

    # Find the pixels of the centroids
    x_pix, y_pix = gc.world2pixel(cygA[0], cygA[1])

    # Cut-out angles: 6, 96, 225 and 315 degrees.
    radii = numpy.linspace(0, 4500, 100)
    x6 = numpy.zeros(len(radii))
    x96 = numpy.zeros(len(radii))
    x225 = numpy.zeros(len(radii))
    x315 = numpy.zeros(len(radii))
    y6 = numpy.zeros(len(radii))
    y96 = numpy.zeros(len(radii))
    y225 = numpy.zeros(len(radii))
    y315 = numpy.zeros(len(radii))
    for i, r in enumerate(radii):
        x6[i] = r*numpy.cos(9*numpy.pi/180)
        y6[i] = r*numpy.sin(6*numpy.pi/180)

        x96[i] = r*numpy.cos(96*numpy.pi/180)
        y96[i] = r*numpy.sin(96*numpy.pi/180)

        x225[i] = r*numpy.cos(225*numpy.pi/180)
        y225[i] = r*numpy.sin(225*numpy.pi/180)

        x315[i] = r*numpy.cos(315*numpy.pi/180)
        y315[i] = r*numpy.sin(315*numpy.pi/180)

    ax = pyplot.gca()
    ax.plot(x6+x_pix, y6+y_pix, c="w", lw=2)
    ax.plot(x96+x_pix, y96+y_pix, c="w", lw=2)
    ax.plot(x225+x_pix, y225+y_pix, c="w", lw=2)
    ax.plot(x315+x_pix, y315+y_pix, c="w", lw=2)
    ax.text(0.65, 0.85, "MERGER", ha="center", va="center", color="white",
            bbox=dict(facecolor="green", edgecolor="green", pad=6),
            weight="bold", transform=pyplot.gca().transAxes, fontsize=16)
    ax.text(0.1, 0.5, "HOT", ha="center", va="center", color="white",
            bbox=dict(facecolor="red", edgecolor="red", pad=6),
            weight="bold", transform=pyplot.gca().transAxes, fontsize=16)
    ax.text(0.65, 0.4, "HOT", ha="center", va="center", color="white",
            bbox=dict(facecolor="red", edgecolor="red", pad=6),
            weight="bold", transform=pyplot.gca().transAxes, fontsize=16)
    ax.text(0.3, 0.05, "COLD", ha="center", va="center", color="white",
            bbox=dict(facecolor="purple", edgecolor="purple", pad=6),
            weight="bold", transform=pyplot.gca().transAxes, fontsize=16)

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm:ss")

    ax.tick_params(axis="both", which="minor", colors="k",
                   pad=8, width=2, size=4, reset=True)
    ax.tick_params(axis="both", which="major", colors="k",
                   pad=8, width=2, size=8, reset=True)

    pyplot.tight_layout()
    gc.save("out/mosaic_xray_wedges.pdf", dpi=300)


def plot_ledlow_positions(t, world2pixel):
    """ Plot galaxy positions of Ledlow+ (2005)
        @param t           : astropy.table.Table holding Ledlow data
        @param world2pixel : aplpy.FITSFigure world2pixel function to map
                             RA,dec coordinates to Chandra mosaic pixels """

    # Convert hour angle to degrees.
    ra = 15*astropy.coordinates.Angle(t['RAJ2000'].filled(numpy.nan), unit=u.degree)
    dec = astropy.coordinates.Angle(t['DEJ2000'].filled(numpy.nan), unit=u.degree)
    # convert ra, dec to xray mosaic pixel values
    ra, dec = world2pixel(ra, dec)

    # pyplot.gca().invert_xaxis()

    # So far I have been unable to reproduce the following --> all filled
    # Filled symbols show velocities within 1 sigma of the mean
    # Squares indicate velocity less than the mean, triangles greater

    # Biweight mean velocity adopted from text: 19008 km/s
    for ra_i, dec_i, v_i, ve_i in zip(ra, dec, t["HV"], t["e_HV"]):
        if v_i < 19008:
            if v_i < 16648:  # failed attempt... group1 mean velocity
                pyplot.plot(ra_i, dec_i, "s", mec="w", mfc="w", ms=4)
            else:
                pyplot.plot(ra_i, dec_i, "s", mec="w", mfc="w", ms=4)
        else:
            if v_i > 19428:
                pyplot.plot(ra_i, dec_i, "^", mec="w", mfc="w", ms=4)
            else:
                pyplot.plot(ra_i, dec_i, "^", mec="w", mfc="w", ms=4)


def plot_ledlow_contours(t, world2pixel, ngb=12, xsize=500, ysize=500):
    """ Smooth galaxy positions of Ledlow+ (2005) to get contour lines
        @param t           : astropy.table.Table holding Ledlow data
        @param world2pixel : aplpy.FITSFigure world2pixel function to map
                             RA,dec coordinates to Chandra mosaic pixels
        @param ngb         : number of neighbours to smooth over
        @param xsize, ysize: size of the image (resolution?)
        @return            : img, extent """
    ra = 15*astropy.coordinates.Angle(t['RAJ2000'].filled(numpy.nan), unit=u.degree)
    dec = astropy.coordinates.Angle(t['DEJ2000'].filled(numpy.nan), unit=u.degree)
    ra, dec = world2pixel(ra, dec)  # convert ra, dec to xray mosaic pixel values

    # Perform kernel density estimate
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.gaussian_kde.html
    # X, Y = numpy.mgrid[xmax:xmin:300j, ymin:ymax:3000j]
    # positions = numpy.vstack([X.ravel(), Y.ravel()])
    # values = numpy.vstack([ra, dec])
    # kernel = kde.gaussian_kde(values)
    # Z = numpy.reshape(kernel(positions).T, X.shape)
    # Z /= ((41.1-40.6)*60*(20.025-19.50)*60)

    # pyplot.imshow(numpy.rot90(Z), cmap=pyplot.cm.gist_earth_r,
    #               extent=[xmax, xmin, ymin, ymax])
    # cset = pyplot.contour(X, Y, Z, colors="w",
    #         levels=numpy.array([1,2,3,5,8,10])*Z.max()/10)
    # pyplot.clabel(cset, inline=1, fontsize=10)

    # https://stackoverflow.com/questions/2369492
    xmin, xmax = ra.min(), ra.max()
    ymin, ymax = dec.min(), dec.max()

    x0 = (xmin+xmax)/2.
    y0 = (ymin+ymax)/2.

    pos = numpy.zeros([3, len(ra)])
    pos[0,:] = ra
    pos[1,:] = dec
    w = numpy.ones(len(ra))

    P = sphviewer.Particles(pos, w, nb=ngb)
    S = sphviewer.Scene(P)
    S.update_camera(r="infinity", x=x0, y=y0, z=0, xsize=xsize, ysize=ysize)

    R = sphviewer.Render(S)
    img = R.get_image()
    extent = R.get_extent()
    for i, j in zip(xrange(4), [x0,x0,y0,y0]):
        extent[i] += j

    img = 10*img/numpy.max(img)

    return img, extent


def plot_mosaic_with_ledlow(mosaic, ledlow, ngb=12):
    """ Plot Chandra mosaic with smoothed Ledlow contour overlay
        @param mosaic : path to the Chandra x-ray mosaic fits file
        @param ledlow : astropy.table.Table holding Ledlow data
        @param ngb    : number of neighbours to smooth over """
    # Wise in prep X-ray mosaic
    gc = aplpy.FITSFigure(mosaic)

    # Add smoothed log-stretch of the entire mosaic
    gc.show_colorscale(vmin=7.0e-10, vmax=1.0e-6, stretch="log",
                       cmap="spectral", smooth=9)

    # Add scale. Length is 500 kpc after unit conversions
    gc.add_scalebar(0.13227513)
    gc.scalebar.set_corner("bottom right")
    gc.scalebar.set_length(0.1)
    gc.scalebar.set_linewidth(4)
    gc.scalebar.set_label("500 kpc")
    gc.scalebar.set_color("white")

    ax = pyplot.gca()

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm:ss")

    ax.tick_params(axis="both", which="minor", colors="k",
                   pad=8, width=2, size=4, reset=True)
    ax.tick_params(axis="both", which="major", colors="k",
                   pad=8, width=2, size=8, reset=True)

    plot_ledlow_positions(ledlow, gc.world2pixel)
    img, extent = plot_ledlow_contours(ledlow, gc.world2pixel, ngb=ngb)
    cset = pyplot.contour(img, extent=extent, origin="lower", aspect="auto",
           levels=[1,2,3,4,5,8,10], colors="w")
    # pyplot.clabel(cset, inline=1, fontsize=10)
    # pyplot.imshow(img, cmap="cubehelix", extent=extent,
    #               origin="lower", aspect="auto")

    gc.recenter(299.78952, 40.81, width=0.5, height=0.5)
    pyplot.tight_layout()
    gc.save("out/mosaic_xray_ledlow_{0}.pdf".format(ngb), dpi=300)


def plot_ledlow_with_chandra(mosaic, ledlow, confiles, ngb=12):
    gc = aplpy.FITSFigure(mosaic)  # Open for world2pixel, but dont plot.

    # Generated one ds9 confile b/c aplpy gc.show_contour too slow
    # Removed all blocks <6 lines (countour lines of point sources) from confile
    # Limits 3e-09 1e-07. Levels 10. Smoothness 32 (but method is block)
    for c in confiles:
        contours = ascii.read(c)
        xray_ra, xray_dec = gc.world2pixel(contours['col1'], contours["col2"])
        pyplot.plot(xray_ra, xray_dec, "w", lw=1)

    plot_ledlow_positions(ledlow, gc.world2pixel)
    img, extent = plot_ledlow_contours(ledlow, gc.world2pixel, ngb=ngb)
    # cset = pyplot.contour(img, extent=extent, origin="lower", aspect="auto",
    #        levels=[1,2,3,4,5,8,10], colors="w")
    # pyplot.clabel(cset, inline=1, fontsize=10)
    pyplot.imshow(img, cmap="cubehelix", extent=extent,
                  origin="lower", aspect="auto")

    gc.recenter(299.78952, 40.81, width=0.5, height=0.5)
    # pyplot.style.use("dark_background")
    pyplot.tight_layout()
    gc.save("out/mosaic_xray_ledlow_chandraContour_{0}.pdf".format(ngb), dpi=300)

if __name__ == "__main__":
    # Coordinates of the CygA centroid
    cygA = ( 299.8669, 40.734496 )
    cygNW = ( 299.7055, 40.884849 )

    # Data directory and radio/xray observation fits files
    radio = "../runs/RadioObservation/radio5GHz.fits"
    mosaic = "../runs/ChandraObservation/xray/cygnus_tot_flux.fits"
    lss = "../runs/ChandraObservation/lss/cygnus_lss_fill_flux.fits"
    confiles = glob.glob(obsdir+"StruisMosaics/xray_contours_*.con")
    ledlow = t = Vizier.get_catalogs("J/AJ/130/47")[0]  # table1

    # plot_zoomin_of_core(mosaic, radio, cygA)
    # plot_mosaic_with_wedges(mosaic, cygA)
    # plot_mosaic_with_ruler(mosaic, cygA, cygNW)
    for ngb in range(2, 32, 2):
        plot_mosaic_with_ledlow(mosaic, ledlow, ngb=ngb)
        plot_ledlow_with_chandra(mosaic, ledlow, confiles, ngb=ngb)
