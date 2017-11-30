import aplpy
import numpy
import scipy
import colorcet
import matplotlib
from scipy import ndimage
from matplotlib import pyplot
from astropy.io import fits
pyplot.rcParams.update( { "text.usetex": True, "font.size": 16 } )

import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.exceptions import AstropyUserWarning
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)
warnings.filterwarnings("ignore", category=AstropyUserWarning, append=True)

import convert
from parse import psmac2_fitsfile

from deco import concurrent, synchronized
threads=4


def build_observation_Lx(lss_Lx, set_the_stage=True):
    cygA = ( 299.8669, 40.734496 )
    cygNW = ( 299.7055, 40.884849 )

    gc = aplpy.FITSFigure(lss_Lx, figsize=(11, 12))
    gc.show_colorscale(vmin=5.0e-10, vmax=1.0e-7, stretch="log",
        cmap=colorcet.cm["linear_bmw_5_95_c86"],
        smooth=9 if not set_the_stage else None)
    cygA_x, cygA_y = gc.world2pixel(cygA[0], cygA[1])
    cygNW_x, cygNW_y = gc.world2pixel(cygNW[0], cygNW[1])

    if not set_the_stage:
        # Add scale. Length is 500 kpc after unit conversions
        gc.add_scalebar(0.13227513)
        gc.scalebar.set_corner("bottom right")
        gc.scalebar.set_length(0.1)
        gc.scalebar.set_linewidth(4)
        gc.scalebar.set_font_size(22)
        gc.scalebar.set_label("500 kpc")
        gc.scalebar.set_color("white")

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm")
    gc.frame.set_color("white")

    ax = pyplot.gca()
    ax.tick_params(axis="both", which="both", reset=True, color="w", labelcolor="k",
        pad=8, width=2, size=4, direction="in", top="on", right="on")
    ax.tick_params(axis="both", which="major", size=8)

    if not set_the_stage:
        # CygA and CygNW label
        ax.text(cygA_x, 0.60*cygA_y, "CygA", weight="bold",
                ha="center", va="center", color="white", fontsize=22)
        ax.text(cygNW_x, 1.12*cygNW_y, "CygNW", ha="center", va="center",
                color="white", weight="bold", fontsize=22)

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
    cax.ax.set_xlabel(r"X-ray Surface Brightness $\left[\frac{\mathrm{counts}}{\mathrm{cm}^2 \, \mathrm{s}} \right]$", fontsize=18)

    # Yes we must set ticks manually... +_+
    cax.set_ticks([1e-9, 1e-8, 1e-7])
    cax.set_ticklabels(["$10^{-9}$", "$10^{-8}$", "$10^{-7}$"])
    cax.ax.minorticks_on()
    minorticks = gc.image.norm(numpy.hstack([numpy.arange(5, 10, 1)/1e10,
        numpy.arange(2, 10, 1)/1e9, numpy.arange(2, 10, 1)/1e8]))
    cax.ax.tick_params(which="minor", length=3, width=1, direction="in")
    cax.ax.xaxis.set_ticks(minorticks, minor=True)
    cax.ax.tick_params(which="major", direction="in")

    for label in ax.get_ymajorticklabels() + ax.get_yminorticklabels():
        label.set_rotation_mode("anchor")
        label.set_rotation(90)
        label.set_horizontalalignment("center")

    # For some reason shows up double...
    # ax.tick_params(axis="both", which="both", top="off", right="off")

    if not set_the_stage:
        pyplot.subplots_adjust(left=-0.02, right=1.05, top=0.95, bottom=-0.05)
        pyplot.savefig("out/vid/obs_Lx.png", dpi=600)
    return pyplot.gcf(), ax, cax


def build_observation_kT(lss_kT, set_the_stage=True):
    cygA = ( 299.8669, 40.734496 )
    cygNW = ( 299.7055, 40.884849 )

    gc = aplpy.FITSFigure(lss_kT, figsize=(11, 12))
    gc.show_colorscale(vmin=3.5, vmax=12, stretch="linear",
        cmap=colorcet.cm["linear_kryw_5_100_c67"],
        smooth=9 if not set_the_stage else None)
    cygA_x, cygA_y = gc.world2pixel(cygA[0], cygA[1])
    cygNW_x, cygNW_y = gc.world2pixel(cygNW[0], cygNW[1])

    if not set_the_stage:
        # Add scale. Length is 500 kpc after unit conversions
        gc.add_scalebar(0.13227513)
        gc.scalebar.set_corner("bottom right")
        gc.scalebar.set_length(0.1)
        gc.scalebar.set_linewidth(4)
        gc.scalebar.set_font_size(22)
        gc.scalebar.set_label("500 kpc")
        gc.scalebar.set_color("white")

    # Pretty notation on the axes
    gc.tick_labels.set_xformat("hh:mm:ss")
    gc.tick_labels.set_yformat("dd:mm")
    gc.frame.set_color("white")

    ax = pyplot.gca()
    ax.tick_params(axis="both", which="both", reset=True, color="w", labelcolor="k",
        pad=8, width=2, size=4, direction="in", top="on", right="on")
    ax.tick_params(axis="both", which="major", size=8)

    if not set_the_stage:
        # CygA and CygNW label
        ax.text(cygA_x, 0.60*cygA_y, "CygA", weight="bold",
                ha="center", va="center", color="white", fontsize=22)
        ax.text(cygNW_x, 1.12*cygNW_y, "CygNW", ha="center", va="center",
                color="white", weight="bold", fontsize=22)

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
    cax.ax.tick_params(which="major", direction="in")

    for label in ax.get_ymajorticklabels() + ax.get_yminorticklabels():
        label.set_rotation_mode("anchor")
        label.set_rotation(90)
        label.set_horizontalalignment("center")

    # For some reason shows up double...
    # ax.tick_params(axis="both", which="both", top="off", right="off")

    if not set_the_stage:
        pyplot.subplots_adjust(left=-0.02, right=1.05, top=0.95, bottom=-0.05)
        pyplot.savefig("out/vid/obs_kT.png", dpi=600)
    return pyplot.gcf(), ax, cax


def get_core_separation(H, pixelscale, i, verbose=False, do_plot=False):
    xhalf = H.shape[0]//2
    left, right = numpy.hsplit(H, 2)
    idx1 = list(left.flatten()).index(left.max())
    cygA = idx1 % left.shape[1], idx1 / left.shape[1]

    idx2 = list(right.flatten()).index(right.max())
    cygNW = idx2 % right.shape[1] + xhalf, idx2 / right.shape[1]

    distance = numpy.sqrt((cygA[0]-cygNW[0])**2+(cygA[1]-cygNW[1])**2)
    distance *= pixelscale

    if verbose:
        print "Peaks for i = {0}".format(i)
        print "  cygA:  (x, y) = {0}".format(cygA)
        print "  cygNW: (x, y) = {0}".format(cygNW)
        print "  distance      = {0:.2f} kpc\n".format(distance)

    if do_plot:
        pyplot.figure()
        pyplot.imshow(numpy.log10(H), origin="lower")
        pyplot.plot(cygA[0], cygA[1], "rX", ms=5)
        pyplot.plot(cygNW[0], cygNW[1], "rX", ms=5)
        pyplot.xticks([],[]); pyplot.yticks([],[])
        pyplot.tight_layout()
        pyplot.savefig("out/vid/peakfind_{0:03d}.pdf".format(i), bbox_inches="tight")
        pyplot.close()

    return cygA, cygNW, distance


def zoom_into_box(H, i, nsteps,
        xlen, xoffset, desired_xlen_sim_pix,
        ylen, yoffset, desired_ylen_sim_pix, verbose=False):
    x1 = (i*xoffset/nsteps)
    y1 = (i*yoffset/nsteps)
    x2 = (xlen-i*(xlen-xoffset-desired_xlen_sim_pix)/nsteps)
    y2 = (ylen-i*(ylen-yoffset-desired_ylen_sim_pix)/nsteps)

    zoomstep = H[y1:y2, x1:x2]
    zoomx = float(xlen) / numpy.abs(y1-y2)
    zoomy = float(ylen) / numpy.abs(x2-x1)

    if verbose:
        print("\n    x1   : {0}\n    x2   : {1}\n    y1   : {2}\n    y2   : {3}"\
            .format(x1, x2, y1, y2))
        print("    zoomx: {0}\n    zoomy: {1}\n".format(zoomx, zoomy))

    return scipy.ndimage.zoom(zoomstep,  [zoomx, zoomy], order=3), zoomx


@concurrent(processes=threads)
def build_bestfit(i, Lx, kT, xlen, ylen, pix2kpc_sim, time, outdir, pix2kpc_obs,
                  arcsec2kpc, EA1=0, EA2=0, zoom=None):
    """ Stage 1: snapshot_000 snapshot_147 3
        Stage 2: snapshot_147_000 snapshot_147_010 1 [end at 9!]
        Stage 3: Euler_Angle_1 0 51 3
        Stage 3: Euler_Angle_2 0 45 3 """

    print("    ... building {0:03d}: {1:04.6f} --> {2}, {3}; {4}".format(i, time, EA1, EA2, zoom))

    #### Build Lx ####
    lss_Lx = "/usr/local/mscproj/runs/ChandraObservation/lss/cygnus_lss_fill_flux_2Msec.fits"
    fig_Lx, ax_Lx, cax_Lx = build_observation_Lx(lss_Lx)

    # First, get some parameters from the observation to ensure
    # equal smoothing and limits to the imshow of Smac Lx/kT
    magic = 5.82e-9 / 1.77e-5
    smooth_obs_kpc = 9 * pix2kpc_obs * arcsec2kpc
    smooth_sim_kpc = smooth_obs_kpc / pix2kpc_sim

    (xmin, xmax), (ymin, ymax) = ax_Lx.get_xlim(), ax_Lx.get_ylim()
    Lx_smooth = magic*scipy.ndimage.filters.gaussian_filter(
        Lx, smooth_sim_kpc)
    im = ax_Lx.imshow(#numpy.ma.masked_less_equal(Lx_smooth, 5.0e-10),
        Lx_smooth, extent=[xmin, xmax, ymin, ymax],
        origin="lower", vmin=5.0e-10, vmax=1.0e-7,
        cmap=colorcet.cm["linear_bmw_5_95_c86"],
        norm=matplotlib.colors.LogNorm())

    # Hide observer units on axes
    ax_Lx.tick_params(axis="both", colors="white")
    ax_Lx.xaxis.label.set_color("white")
    ax_Lx.yaxis.label.set_color("white")

    # Toss in some simulation numbers
    cygA, cygNW, distance = get_core_separation(Lx, pix2kpc_sim, i)

    # Set Time + Distance
    distance_ph = distance
    if EA1 is not 0:
        distance_ph = 1076.43
        distance_obs = distance
    if zoom:
        distance_ph = 1076.43
        distance_obs = 761.22
    timedistance = r"\begin{tabular}{p{2.0cm}ll}"
    timedistance += r" T$_{{\rm simulation}}$ & = & {0:<07.3f} Gyr \\".format(time)
    if EA2 is 0:
        timedistance += r" d$_{{\rm physical}}$ & = & {0:<06.2f} kpc \\".format(distance_ph)
    else:
        timedistance += r" d$_{{\rm physical}}$ & = & {0:<06.2f} kpc \\".format(distance_ph)
        timedistance += r" d$_{{\rm observed}}$ & = & {0:<06.2f} kpc \\".format(distance_obs)
    timedistance += (" \end{tabular}")
    ax_Lx.text(0.5, 0.98, timedistance, size=18, color="white",
        ha="center", va="top", transform=ax_Lx.transAxes)

    # Set Scale indicator
    scale = xlen*pix2kpc_sim
    if zoom:
        scale /= zoom
    scale = "[{0:.1f} Mpc]$^2$".format(float(scale)/1000)
    pad = 50
    ax_Lx.text(2*pad, pad, scale, size=16, color="white", ha="left", va="bottom")

    # Set Euler Angles
    EA0 = 90
    angles = r"\begin{tabular}{p{1.25cm}ll}"
    angles += r" EA0 & = & {0:03d} \\".format(EA0)
    angles += " EA1 & = & {0:03d} \\\\".format(EA1)
    angles += " EA2 & = & {0:03d} \\\\".format(EA2)
    angles += (" \end{tabular}")
    ax_Lx.text(xmax-2*pad, pad, angles, size=16, color="white", ha="right", va="bottom")

    fig_Lx.subplots_adjust(left=-0.02, right=1.05, top=0.95, bottom=-0.05)
    fig_Lx.savefig("out/vid/sim_Lx_{0:03d}.png".format(i), dpi=600)
    pyplot.close(fig_Lx)


    #### Build kT ####
    lss_kT = "/usr/local/mscproj/runs/ChandraObservation/lss/working_spectra_kT_map_2Msec.fits"
    fig_kT, ax_kT, cax_kT = build_observation_kT(lss_kT)

    (xmin, xmax), (ymin, ymax) = ax_kT.get_xlim(), ax_kT.get_ylim()
    kT_smooth = scipy.ndimage.filters.gaussian_filter(
        convert.K_to_keV(kT), smooth_sim_kpc)
    im = ax_kT.imshow(#numpy.ma.masked_less_equal(kT_smooth, 3.5),
        kT_smooth, extent=[xmin, xmax, ymin, ymax],
        origin="lower", vmin=3.5, vmax=12,
        cmap=colorcet.cm["linear_kryw_5_100_c67"])
    # Hide observer units on axes
    ax_kT.tick_params(axis="both", colors="white")
    ax_kT.xaxis.label.set_color("white")
    ax_kT.yaxis.label.set_color("white")

    # Set Time + Distance
    ax_kT.text(0.5, 0.98, timedistance, size=18, color="white",
        ha="center", va="top", transform=ax_kT.transAxes)

    # Set Scale indicator
    ax_kT.text(2*pad, pad, scale, size=16, color="white", ha="left", va="bottom")

    # Set Euler Angles
    ax_kT.text(xmax-2*pad, pad, angles, size=16, color="white", ha="right", va="bottom")

    fig_kT.subplots_adjust(left=-0.02, right=1.05, top=0.95, bottom=-0.05)
    fig_kT.savefig("out/vid/sim_kT_{0:03d}.png".format(i), dpi=600)
    pyplot.close(fig_kT)


def set_observation():
    print("\nSetting observed boxsize and CygA Centroid")
    # First we find observed CygA centroid, and 'boxsize'
    lss_Lx = "/usr/local/mscproj/runs/ChandraObservation/lss/cygnus_lss_fill_flux_2Msec.fits"
    lss_kT = "/usr/local/mscproj/runs/ChandraObservation/lss/working_spectra_kT_map_2Msec.fits"
    mosaic_Lx = fits.open(lss_Lx)
    mosaic_kT = fits.open(lss_kT)
    contour_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_Lx[0].data, 25)
    kT_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_kT[0].data, 9)

    # Find the centroid of CygA to align simulation and observation later on
    maxcounts_obs = mosaic_Lx[0].data.max()
    maxcounts_obs_index = mosaic_Lx[0].data.argmax()  # of flattened array
    ylen_obs_pix, xlen_obs_pix = mosaic_Lx[0].data.shape
    xcenter_obs = maxcounts_obs_index % xlen_obs_pix
    ycenter_obs = maxcounts_obs_index / xlen_obs_pix

    # Find the dimensions of the Chandra image in pix, arcsec and kpc
    xlen_obs_pix = mosaic_Lx[0].header["NAXIS1"]  # same as using mosaic_smooth.shape
    ylen_obs_pix = mosaic_Lx[0].header["NAXIS2"]
    # Chandra size of pixel 0.492". Value in header is in degrees.
    pix2arcsec_obs = mosaic_Lx[0].header["CDELT2"]*3600
    xlen_obs_arcsec = xlen_obs_pix * pix2arcsec_obs
    ylen_obs_arcsec = ylen_obs_pix * pix2arcsec_obs
    from cosmology import CosmologyCalculator
    cc = CosmologyCalculator(0.0562)
    arcsec2kpc = cc.kpc_DA
    pix2kpc_obs = pix2arcsec_obs * arcsec2kpc
    xlen_obs_kpc = xlen_obs_arcsec * arcsec2kpc
    ylen_obs_kpc = ylen_obs_arcsec * arcsec2kpc
    zlen_obs_kpc = ylen_obs_kpc

    mosaic_Lx.close()
    mosaic_kT.close()

    print "  Chandra Observation [lss_fill_flux]"
    print "    Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "    Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "    Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "    CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    return (xlen_obs_kpc, ylen_obs_kpc, xcenter_obs, ycenter_obs,
            pix2kpc_obs, arcsec2kpc)


@synchronized
def build_varying_time(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False, bestfit=True):
    print("\nBuilding frames of varying time")

    Lx_stage1 = smacdir + "BestFitSimulation_Stage1_BestTime_Lx.fits.fz"
    kT_stage1 = smacdir + "BestFitSimulation_Stage1_BestTime_kT.fits.fz"

    header, Lx = psmac2_fitsfile(Lx_stage1)
    header2, kT = psmac2_fitsfile(kT_stage1)
    nsnap, xlen, ylen = kT.shape
    pix2kpc_sim = float(header["XYSize"])/int(xlen)

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    print("  nsnap      : {0}\n  xlen       : {1}\n  ylen       : {2}"\
          .format(nsnap, xlen, ylen))
    print("  dt         : {0}".format(dt))
    print("  pix2kpc_sim: {0:.3f}".format(pix2kpc_sim))
    print("  pix2kpc_obs: {0:.3f}".format(pix2kpc_obs))
    print("  arcsec2kpc : {0:.3f}".format(arcsec2kpc))
    print("")

    for i in range(147 if bestfit else nsnap):
        time = 0 + 3*i * dt
        print("  {0:03d}: {1:04.6f}".format(i, time))
        if skip: continue
        # if i > 1 and i < nsnap-2: continue
        build_bestfit(i, Lx[i], kT[i], xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc)

    return frame_number + i


@synchronized
def build_varying_time_finer_interpolation(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False):
    print("\nBuilding frames of varying time with finer interpolation")

    Lx_stage2 = smacdir + "BestFitSimulation_Stage1_BestTime_Lx_finer.fits.fz"
    kT_stage2 = smacdir + "BestFitSimulation_Stage1_BestTime_kT_finer.fits.fz"

    header, Lx = psmac2_fitsfile(Lx_stage2)
    header2, kT = psmac2_fitsfile(kT_stage2)
    nsnap, xlen, ylen = Lx.shape
    pix2kpc_sim = float(header["XYSize"])/int(xlen)

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    t0 = 147 * dt
    for i in range(1, nsnap-1):
        time = t0 + i * dt_finer
        print("  {0:03d}: {1:04.6f}".format(i+frame_number, time))
        if skip: continue
        # if i > 1 and i < nsnap-3: continue
        build_bestfit(i+frame_number, Lx[i], kT[i], xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc)

    return frame_number + i


@synchronized
def build_varying_euler_angle_1(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False):
    print("\nBuilding frames of varying Euler Angle 1 (Angle in Sky Plane)")

    Lx_stage3 = smacdir + "BestFitSimulation_Stage1_BestTime_Lx_EA1.fits.fz"
    kT_stage3 = smacdir + "BestFitSimulation_Stage1_BestTime_kT_EA1.fits.fz"

    header, Lx = psmac2_fitsfile(Lx_stage3)
    header2, kT = psmac2_fitsfile(kT_stage3)
    nsnap, xlen, ylen = Lx.shape
    pix2kpc_sim = float(header["XYSize"])/int(xlen)

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    t0 = 147 * dt
    time = t0 + 9 * dt_finer
    for i in range(1, nsnap):
        EA1 = i
        print("  {0:03d}: {1:04.6f} --> {2}".format(i+frame_number, time, EA1))
        if skip: continue
        # if i > 1 and i < nsnap-2: continue
        build_bestfit(i+frame_number, Lx[i], kT[i], xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc, EA1=EA1)

    return frame_number + i


@synchronized
def build_varying_euler_angle_2(smacdir, outdir, frame_number, pix2kpc_obs,
        arcsec2kpc, skip=False):
    print("\nBuilding frames of varying Euler Angle 2 (Inclination)")
    print("Going from 1 - 80")

    Lx_stage4 = smacdir + "BestFitSimulation_Stage1_BestTime_Lx_EA2.fits.fz"
    kT_stage4 = smacdir + "BestFitSimulation_Stage1_BestTime_kT_EA2.fits.fz"

    header, Lx = psmac2_fitsfile(Lx_stage4)
    header2, kT = psmac2_fitsfile(kT_stage4)
    nsnap, xlen, ylen = Lx.shape
    pix2kpc_sim = float(header["XYSize"])/int(xlen)

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    t0 = 147 * dt
    time = t0 + 9 * dt_finer
    EA1 = 51
    for i in range(1, 81):
        EA2 = i
        print("  {0:03d}: {1:04.6f} --> {2}, {3}".format(i+frame_number, time, EA1, EA2))
        if skip: continue
        # if i > 1 and i < nsnap-2: continue
        build_bestfit(i+frame_number, Lx[i], kT[i], xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc, EA1=EA1, EA2=EA2)

    return frame_number + i


@synchronized
def build_varying_euler_angle_2_goback(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False):
    print("\nBuilding frames of varying Euler Angle 2 (Inclination)")
    print("Going from 80 - 0")

    Lx_stage4 = smacdir + "BestFitSimulation_Stage1_BestTime_Lx_EA2.fits.fz"
    kT_stage4 = smacdir + "BestFitSimulation_Stage1_BestTime_kT_EA2.fits.fz"

    header, Lx = psmac2_fitsfile(Lx_stage4)
    header2, kT = psmac2_fitsfile(kT_stage4)
    nsnap, xlen, ylen = Lx.shape
    pix2kpc_sim = float(header["XYSize"])/int(xlen)

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    t0 = 147 * dt
    time = t0 + 9 * dt_finer
    EA1 = 51
    for j in range(79, -1, -1):
        EA2 = j
        # if i > 1 and i < nsnap-2: continue
        print("  {0:03d}: {1:04.6f} --> {2}, {3}".format(frame_number+80-j, time, EA1, EA2))
        if j > 46: continue
        if skip: continue
        build_bestfit(frame_number+80-j, Lx[j], kT[j], xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc, EA1=EA1, EA2=EA2)

    return frame_number+80-j


@synchronized
def build_varying_euler_angle_2_goforth(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False):
    print("\nBuilding frames of varying Euler Angle 2 (Inclination)")
    print("Going from 1 - 45")

    Lx_stage4 = smacdir + "BestFitSimulation_Stage1_BestTime_Lx_EA2.fits.fz"
    kT_stage4 = smacdir + "BestFitSimulation_Stage1_BestTime_kT_EA2.fits.fz"

    header, Lx = psmac2_fitsfile(Lx_stage4)
    header2, kT = psmac2_fitsfile(kT_stage4)
    nsnap, xlen, ylen = Lx.shape
    pix2kpc_sim = float(header["XYSize"])/int(xlen)

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    t0 = 147 * dt
    time = t0 + 9 * dt_finer
    EA1 = 51
    for i in range(1, 45):
        EA2 = i
        # if i > 1 and i < nsnap-2: continue
        print("  {0:03d}: {1:04.6f} --> {2}, {3}".format(frame_number+i, time, EA1, EA2))
        if skip: continue
        build_bestfit(frame_number+i, Lx[i], kT[i], xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc, EA1=EA1, EA2=EA2)

    return frame_number + i , Lx[i], kT[i], pix2kpc_sim


def increase_time_resolution_of_ea2(frame_number):
    print("\nIncreasing Time Resolution of Varying Inclination")

    increase_time_resolution_EA2 = False
    if increase_time_resolution_EA2:
        import os
        for i in range(1, 16):
            # print("sim_Lx_{0:03d}.png".format(i+frame_number))
            # print("sim_kT_{0:03d}.png".format(i+frame_number))
            os.system("cp out/sim_Lx_{0:03d}.png out/sim_Lx_f_{1:03d}.png".format(i+frame_number, i+frame_number + 2*i-2))
            os.system("cp out/sim_Lx_{0:03d}.png out/sim_Lx_f_{1:03d}.png".format(i+frame_number, i+frame_number + 2*i-1))
            os.system("cp out/sim_Lx_{0:03d}.png out/sim_Lx_f_{1:03d}.png".format(i+frame_number, i+frame_number + 2*i))
            os.system("cp out/sim_kT_{0:03d}.png out/sim_kT_f_{1:03d}.png".format(i+frame_number, i+frame_number + 2*i-2))
            os.system("cp out/sim_kT_{0:03d}.png out/sim_kT_f_{1:03d}.png".format(i+frame_number, i+frame_number + 2*i-1))
            os.system("cp out/sim_kT_{0:03d}.png out/sim_kT_f_{1:03d}.png".format(i+frame_number, i+frame_number + 2*i))
            # os.system("echo out/sim_Lx_{0:03d}.png out/sim_Lx_{0:03d}_.png".format(i+frame_number))

        for i in range(1, 16*3-2):
            print("{0:03d}".format(i+49+9+17))
            os.system("mv out/sim_Lx_f_{0:03d}.png out/sim_Lx_{0:03d}.png".format(i+frame_number))
            os.system("mv out/sim_kT_f_{0:03d}.png out/sim_kT_{0:03d}.png".format(i+frame_number))

        return frame_number + i


@synchronized
def build_zoom_into_simulation_box(outdir, Lx, kT, frame_number, pix2kpc_sim,
        pix2kpc_obs, xcenter_obs, ycenter_obs, arcsec2kpc, skip=False):
    print("\nBuilding frames of zooming into the simulation box")

    # We have now reached best time, best EA1, best EA2, so Lx[-1]
    # will tell us where the centroid sits. We can then place simulated
    # centroid on top of observed centroid (which is already defined).

    dt = 0.01         # For regular snapshots
    dt_finer = dt/40  # For finer interpolation

    t0 = 147 * dt
    time = t0 + 9 * dt_finer

    xlen, ylen = Lx.shape

    cygA, cygNW, distance = get_core_separation(Lx, pix2kpc_sim, 90)
    xcenter_sim = cygA[0]
    ycenter_sim = cygA[1]

    desired_xlen_sim_kpc = xlen_obs_kpc
    desired_ylen_sim_kpc = ylen_obs_kpc
    desired_xlen_sim_pix = int(desired_xlen_sim_kpc / pix2kpc_sim + 0.5)
    desired_ylen_sim_pix = int(desired_ylen_sim_kpc / pix2kpc_sim + 0.5)
    xoffset = int((xcenter_sim * pix2kpc_sim - xcenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)
    yoffset = int((ycenter_sim * pix2kpc_sim - ycenter_obs * pix2kpc_obs) / pix2kpc_sim + 0.5)

    print(xoffset)
    print(yoffset)

    nsteps = 50
    EA1, EA2 = 51, 45
    for i in range(1, nsteps+1):
        print("  {0:03d}: {1:04.6f} --> {2}, {3}".format(i+frame_number, time, EA1, EA2))
        if skip: continue
        Lxzoom, zoomx = zoom_into_box(Lx, i, nsteps,
            xlen, xoffset, desired_xlen_sim_pix,
            ylen, yoffset, desired_ylen_sim_pix )
        kTzoom, zoomx = zoom_into_box(kT, i, nsteps,
          xlen, xoffset, desired_xlen_sim_pix,
            ylen, yoffset, desired_ylen_sim_pix )

        # if i > 1 and i < nsnap-2: continue
        build_bestfit(i+frame_number, Lxzoom, kTzoom, xlen, ylen, pix2kpc_sim,
            time, outdir, pix2kpc_obs, arcsec2kpc, EA1=EA1, EA2=EA2, zoom=zoomx)

    return frame_number + i


if __name__ == "__main__":
    (xlen_obs_kpc, ylen_obs_kpc, xcenter_obs, ycenter_obs,
        pix2kpc_obs, arcsec2kpc) = set_observation()

    # lss_Lx = "/usr/local/mscproj/runs/ChandraObservation/lss/cygnus_lss_fill_flux_2Msec.fits"
    # lss_kT = "/usr/local/mscproj/runs/ChandraObservation/lss/working_spectra_kT_map_2Msec.fits"
    # fig_Lx, ax_Lx, cax_Lx = build_observation_Lx(lss_Lx, set_the_stage=False)
    # fig_kT, ax_kT, cax_kT = build_observation_kT(lss_kT, set_the_stage=False)

    simdir = "/Volumes/Cygnus/timoh/runs/20170115T0907/"
    smacdir = simdir + "analysis/"
    outdir = simdir + "out/"

    # import sys
    # old_stdout, old_stderr = sys.stdout, sys.stderr
    # sys.stderr = open("/dev/null", "w")

    frame_number = 0   # Global Counter to keep track of 'offset'
    frame_number = build_varying_time(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False, bestfit=False)

    frame_number = build_varying_time(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False, bestfit=True)

    frame_number = build_varying_time_finer_interpolation(smacdir, outdir,
        frame_number, pix2kpc_obs, arcsec2kpc, skip=True)

    frame_number = build_varying_euler_angle_1(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False)

    frame_number = build_varying_euler_angle_2(smacdir, outdir, frame_number,
        pix2kpc_obs, arcsec2kpc, skip=False)

    frame_number = build_varying_euler_angle_2_goback(smacdir, outdir,
        frame_number, pix2kpc_obs, arcsec2kpc, skip=False)

    frame_number, Lx, kT, pix2kpc_sim = build_varying_euler_angle_2_goforth(
        smacdir, outdir, frame_number, pix2kpc_obs, arcsec2kpc, skip=False)

    frame_number = build_zoom_into_simulation_box(outdir, Lx, kT, frame_number,
        pix2kpc_sim, pix2kpc_obs, xcenter_obs, ycenter_obs, arcsec2kpc, skip=False)

    import os

    os.system('ffmpeg -y -r 10 -i "out/vid/sim_kT_%3d.png" -profile:v high444 -level 4.1 \
        -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -s "2000:2000" \
        -an "out/vid/sim_kT.mp4"')

    os.system('ffmpeg -y -r 10 -i "out/vid/sim_Lx_%3d.png" -profile:v high444 -level 4.1 \
        -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -s "2000:2000" \
        -an "out/vid/sim_Lx.mp4"')
