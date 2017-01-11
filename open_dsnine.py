#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import argparse
import logging

from parse import psmac2_fitsfile

try:
    import pyds9
except ImportError:
    print("Program requires installation of ds9")
    exit(1)

try:
    import IPython
except ImportError:
    print("Program requires IPython installation")
    exit(1)

try:
    from astropy.io import fits as pyfits
except ImportError:
    try:
        import pyfits
    except ImportError:
        has_fits = False
    else:
        has_fits = True
else:
    has_fits = True


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s|%(name)s|%(levelname)s|%(message)s")
logger = logging.getLogger("open_ds9")


def open_file(d, fname):
    try:
        d.set("frame new")
        d.set("smooth no")
        d.set("file {}".format(fname))

        if "temperature" in fname:
            # d.set("scale linear")
            # d.set("scale limits 2e6 2e8")
            # d.set("cmap bb")
            a = d.get_arr2np()
            kB = 8.6173427909e-08  # keV/K
            d.set_np2arr(a*kB)
            d.set("scale linear")
            d.set("scale limits 0.1 9")
            d.set("cmap bb")
            # time.sleep(0.5)
        if "xray" in fname:
            d.set("scale log")
            d.set("scale limits 1e-8 2e-4")
            d.set("cmap sls")
        if "density" in fname:
            d.set("scale log")
            d.set("scale limits 2e-16 0.02")
            d.set("cmap sls")
        if "velocity" in fname:
            d.set("scale log")
            d.set("scale limits 1.1e7 2.2e8")
            d.set("cmap bb")
    except ValueError as err:
        print str(err)
        print has_fits
        if "XPA$ERROR Unable to load fits" in str(err) and has_fits:
            logger.exception(
                "Cannot load fits file natively, using pyfits")
            d.set_np2arr(pyfits.getdata(fname).astype(float))
        else:
            raise


def setup_ds9_connection(reconnect):
    """ Open a DS9 window. If DS9 already opened spawn a new instance """

    # ds9_targets returns None if no ds9 window is opened
    ds9_running = pyds9.ds9_targets()
    if ds9_running:
        if reconnect and len(ds9_running) == 1:
            connectionstring = (ds9_running[0]).replace("DS9:ds9 ", "")
            d = pyds9.DS9(connectionstring)
            time.sleep(0.5)
            return d, connectionstring
        elif reconnect and len(ds9_running) != 1:
            print "Error: multiple ds9 instances running. Unknown which to connect to."
            exit(1)

        print "We see that ds9 is already running."
        sys.stdout.write("Opening a new ds9 window...")
        sys.stdout.flush()
        import os
        launch = os.system("/Applications/SAOImage\ DS9.app/Contents/MacOS/ds9 &")
        del(os)
        time.sleep(3)
        if launch is 0:
            print " done"
            ds9_running_new = pyds9.ds9_targets()
            unique_entries = list(set(ds9_running_new) - set(ds9_running))
            if len(unique_entries) == 1:
                connectionstring = unique_entries[0]
                connectionstring = connectionstring.replace("DS9:ds9 ", "")
                d = pyds9.DS9(connectionstring)
                time.sleep(0.5)
            else:
                exit(1)
        else:
            exit(1)
    else:
        print "No ds9 instance is running."
        sys.stdout.write("Opening a new ds9 window ...")
        sys.stdout.flush()
        d = pyds9.DS9()
        time.sleep(0.5)
        connectionstring = (pyds9.ds9_targets()[0]).replace("DS9:ds9 ", "")
        print " done"

    return d, connectionstring

def cmd():
    import os
    os.system("open /Applications/Firefox.app/ 'http://ds9.si.edu/doc/ref/command.html'")
    del(os)

def print_fitsheader(fits):
    for line in repr(fits[0].header).split("\n"):
        print line.strip()


def main(args):
    # List of commands: http://ds9.si.edu/doc/ref/command.html

    d, connectionstring = setup_ds9_connection(args.reconnect)

    header = "DS9 instance in object `d`:"
    header += " Connectionstring = {0}".format(connectionstring)

    if args.chandra:
        sys.stdout.write("\nRestoring ds9 backup of Lx lss and kT lss ...")
        sys.stdout.flush()
        lss = "/usr/local/mscproj/runs/ChandraObservation/ds9bck_Lx-lss_kT-lss/ds9.bck"
        lss_Lx = lss+".dir/Frame1/cygnus_lss_fill_flux.fits"
        lss_kT = lss+".dir/Frame2/working_spectra_kT_map.fits"
        Lx = pyfits.open(lss_Lx)
        kT = pyfits.open(lss_kT)
        header += "\nlss Lx instance in object 'Lx'"
        header += "\nlss kT instance in object 'kT'"
        header += ""
        if not args.reconnect: d.set("restore {0}".format(lss))
        print " done"

    if args.filename:
        for i, fname in enumerate(args.filename):
            sys.stdout.write("\nOpening file '{0}' ...".format(fname))
            sys.stdout.flush()
            if not args.reconnect: open_file(d, fname)
            if has_fits :
                f = "f{0}".format(i)
                exec("{0} = pyfits.open(fname)".format(f))
                if "fits.fz" in fname:
                    exec("{0}_header, {0}_data = psmac2_fitsfile(fname)".format(f))
                    exec("{0}_pix2kcp = float({0}_header['XYSize'])/float({0}_header['XYPix'])".format(f))
                header += "\npyfits instance in object `{0}`".format(f)

            print " done"

    IPython.embed(banner1="", header=header)

    # Close the ds9 window if it is still open
    if pyds9.ds9_targets() and "DS9:ds9 {0}".format(connectionstring) in pyds9.ds9_targets():
        d.set("exit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Open SAOImage DS9, set up XPA"+
        "connection, and embed into iPython session")
    parser.add_argument("-f", "--filename", dest="filename", default=None,
                        nargs='*')
    parser.add_argument("-c", "--chandra", dest="chandra", action="store_true")
    parser.add_argument("-r", "--reconnect", dest="reconnect", action="store_true")

    main(parser.parse_args())
