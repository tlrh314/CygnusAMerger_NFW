import numpy
import glob
import copy
import scipy
from scipy import ndimage
import matplotlib
from matplotlib import pyplot
pyplot.switch_backend("Qt4Agg")
from astropy.io import ascii, fits

import parse
from cluster import Cluster
from simulation import Simulation
from parse import toycluster_icfile
from main import new_argument_parser


def apply_projection(EulAng, gas, dm):
    """ all vector quantities have to be rotated

        This method has been shamelessly copy-pasted from P-Smac2,
            see src/setup.c#L107-L208

        @param EulAng: three-vector with psi, theta, phi
        @param gas: AstroPy Table containing gas properties
        @param dm: AstroPy Table containing dm properties

        @return: gas, dm where the vectors have been rotated """

    YAWPITCHROLL = True
    deg2rad = numpy.pi / 180.0
    psi = EulAng[0] * deg2rad
    theta = EulAng[1] * deg2rad
    phi = EulAng[2] * deg2rad

    if not phi and not theta and not psi:
        return  # nothing to do

    A = numpy.zeros((3, 3))  # Rotation Matrix

    # Define rotation matrix
    if YAWPITCHROLL:  # Luftfahrtnorm (DIN 9300) (Yaw-Pitch-Roll, Z, Y', X'')
        A = [
                [
                    numpy.cos(theta) * numpy.cos(phi),
                    numpy.cos(theta) * numpy.sin(phi),
                    -numpy.sin(theta)
                ], [
                    numpy.sin(psi) * numpy.sin(theta) * numpy.cos(phi) - numpy.cos(psi) * numpy.sin(phi),
                    numpy.sin(psi) * numpy.sin(theta) * numpy.sin(phi) + numpy.cos(psi) * numpy.cos(phi),
                    numpy.sin(psi) * numpy.cos(theta)
                ], [
                    numpy.cos(psi) * numpy.sin(theta) * numpy.cos(phi) + numpy.sin(psi) * numpy.sin(phi),
                    numpy.cos(psi) * numpy.sin(theta) * numpy.sin(phi) - numpy.sin(psi) * numpy.cos(phi),
                    numpy.cos(psi) * numpy.cos(phi)
                ]
        ]
    else:  # Euler Matrix, y-Convention
       A = [
               [
                    -numpy.sin(psi) * numpy.sin(phi) + numpy.cos(psi) * numpy.cos(theta) * numpy.cos(phi),
                    -numpy.sin(psi) * numpy.cos(phi) - numpy.cos(psi) * numpy.cos(theta) * numpy.sin(phi),
                    numpy.cos(psi) * numpy.sin(theta)
                ], [
                    numpy.cos(psi) * numpy.sin(phi) + numpy.sin(psi) * numpy.cos(theta) * numpy.cos(phi),
                    numpy.cos(psi) * numpy.cos(phi) - numpy.sin(psi) * numpy.cos(theta) * numpy.sin(phi),
                    numpy.sin(psi) * numpy.sin(theta)
                ], [
                    -numpy.sin(theta) * numpy.cos(phi),
                    numpy.sin(theta) * numpy.sin(phi),
                    numpy.cos(theta)
                ]
            ]

    # Apply transformations to dm/gas positions
    x, y, z = dm["x"], dm["y"], dm["z"]
    dm["x"] = A[0][0] * x + A[0][1] * y + A[0][2] * z
    dm["y"] = A[1][0] * x + A[1][1] * y + A[1][2] * z
    dm["z"] = A[2][0] * x + A[2][1] * y + A[2][2] * z

    x, y, z = gas["x"], gas["y"], gas["z"]
    gas["x"] = A[0][0] * x + A[0][1] * y + A[0][2] * z
    gas["y"] = A[1][0] * x + A[1][1] * y + A[1][2] * z
    gas["z"] = A[2][0] * x + A[2][1] * y + A[2][2] * z

    # Apply transformations to dm/gas velocities
    vx, vy, vz = dm["vx"], dm["vy"], dm["vz"]
    dm["vx"] = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz
    dm["vy"] = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz
    dm["vz"] = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz

    vx, vy, vz = gas["vx"], gas["vy"], gas["vz"]
    gas["vx"] = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz
    gas["vy"] = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz
    gas["vz"] = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz

    # Apply transformations to magnetic field (gas only)
    if "Bx" in gas.keys():
        Bx, By, Bz = gas["Bx"], gas["By"], gas["Bz"]

        gas["Bx"] = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2];
        gas["By"] = A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2];
        gas["Bz"] = A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2];

    # Apply transformations to bulk velocity (gas only; only ifdef VTURB)
    if "VBulkx" in gas.keys():
        vx, vz, vz = gas["VBulkx"], gas["VBulky"], gas["VBulkz"]
        gas["VBulkx"] = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz
        gas["VBulky"] = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz
        gas["VBulkz"] = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz


    return gas, dm


def project_gadget_snap_and_set_boxsize_equal_to_observed_lss(
        basedir, timestamp, snapname ):
    # First we find observed CygA centroid, and 'boxsize'
    lss_Lx = "/usr/local/mscproj/runs/ChandraObservation/lss/cygnus_lss_fill_flux_2Msec.fits"
    mosaic_Lx = fits.open(lss_Lx)
    contour_smooth = scipy.ndimage.filters.gaussian_filter(mosaic_Lx[0].data, 25)

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
    zlen_obs_kpc = ylen_obs_kpc

    print "Chandra Observation [lss_fill_flux]"
    print "  Shape ({0},   {1})   pixels".format(xlen_obs_pix, ylen_obs_pix)
    print "  Shape ({0:.1f}, {1:.1f}) arcsec".format(xlen_obs_arcsec, ylen_obs_arcsec)
    print "  Shape ({0:.1f}, {1:.1f}) kpc".format(xlen_obs_kpc, ylen_obs_kpc)
    print "  CygA at ({0}, {1}) pixels. Value = {2:2.2g}".format(xcenter_obs, ycenter_obs, maxcounts_obs)

    # Set up simulation paths
    basedir = "/Volumes/Cygnus/timoh/"
    timestamp = "20170115T0907"
    simdir = "{0}/runs/{1}/snaps/".format(basedir, timestamp)

    # Read in snapshot
    snapname = "snapshot_147_010"
    header, gas, dm = toycluster_icfile(simdir+snapname)
    boxsize = header["boxSize"]
    boxhalf = boxsize/2

    # Use Cluster instance to hold data. Toycluster parms needed for find_dm_centroid
    c = Cluster(header)
    c.set_header_properties()
    c.parms = parse.read_toycluster_parameterfile(glob.glob(simdir+"../ICs/*.par")[0])

    # !! Domain [-boxhalf, boxhalf] for rotation matrices !!
    gas["x"] -= boxhalf
    gas["y"] -= boxhalf
    gas["z"] -= boxhalf
    dm["x"]  -= boxhalf
    dm["y"]  -= boxhalf
    dm["z"]  -= boxhalf

    # This seems best-fit rotation angles
    EulAng = numpy.array([90, 51, 45])
    gas, dm = apply_projection(EulAng, gas, dm)

    # Now find centroids in rotated image to place cygA and fidicual
    # cygA at same location in plot. !! Domain find_dm_centroid [0, boxSize] !!
    gas["x"] += boxhalf
    gas["y"] += boxhalf
    gas["z"] += boxhalf
    dm["x"]  += boxhalf
    dm["y"]  += boxhalf
    dm["z"]  += boxhalf

    c.dm, c.gas = dm, gas
    ImpactParam = c.parms["ImpactParam"]
    c.parms["ImpactParam"] = 1337  # not 0.0
    c.find_dm_centroid(single=False)
    c.parms["ImpactParam"] = ImpactParam  # put back

    xcenter_sim = c.centroid0[0]
    ycenter_sim = c.centroid0[1]
    zcenter_sim = c.centroid0[2]

    print "Gadget3 Snapshot"
    print "  CygA at ({0}, {1}, {2}).".format(xcenter_sim, ycenter_sim, zcenter_sim)

    gas["x"] = gas["x"] - xcenter_sim + xcenter_obs*pix2kpc_obs
    gas["y"] = gas["y"] - ycenter_sim + ycenter_obs*pix2kpc_obs
    gas["z"] = gas["z"] - zcenter_sim + ycenter_obs*pix2kpc_obs
    dm["x"] = dm["x"] - xcenter_sim + xcenter_obs*pix2kpc_obs
    dm["y"] = dm["y"] - ycenter_sim + ycenter_obs*pix2kpc_obs
    dm["z"] = dm["z"] - zcenter_sim + ycenter_obs*pix2kpc_obs

    zoomx = float(ylen_obs_pix) / xlen_obs_kpc
    zoomy = float(xlen_obs_pix) / ylen_obs_kpc
    shape_matched = scipy.ndimage.zoom(contour_smooth,  [1/zoomx, 1/zoomy], order=3)

    pyplot.figure()
    ax = pyplot.gca()
    delta = 1
    x = numpy.arange(0, xlen_obs_kpc, delta)
    y = numpy.arange(0, int(ylen_obs_kpc), delta)
    X, Y = numpy.meshgrid(y, x)
    CS = pyplot.contour(X, Y, numpy.log10(shape_matched.clip(10**-8.8)), 7,
        colors="black", linestyles="solid", ls=4)
    # pyplot.clabel(CS, fontsize=16, inline=1, colors="black")

    norm = matplotlib.colors.LogNorm()
    cax = pyplot.hist2d(gas["x"], gas["y"], bins=1024)[-1]
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.xlim(0, xlen_obs_kpc)
    pyplot.ylim(0, ylen_obs_kpc)
    pyplot.colorbar(cax)
    pyplot.show()

    pyplot.figure()
    norm = matplotlib.colors.LogNorm()
    cax = pyplot.hist2d(dm["x"], dm["z"], bins=256, norm=norm)[-1]
    pyplot.xlabel("x")
    pyplot.ylabel("z")
    pyplot.colorbar(cax)
    pyplot.show()

    imatch, = numpy.where(
          ((gas["x"] > 0.) & (gas["x"] < xlen_obs_kpc))
        & ((gas["y"] > 0.) & (gas["y"] < ylen_obs_kpc))
        & ((gas["z"] > 0.) & (gas["z"] < ylen_obs_kpc))
    )

    return header, gas, dm
