import astropy
from astropy.io import ascii


# ----------------------------------------------------------------------------
# Parse Chandra observation: i) quiescent/non-merger; ii) sector analysis
# ----------------------------------------------------------------------------
def parse_chandra_quiescent(name):
    """ `quiescent', or average profile (data copied at 20161108) """
    datadir = "/usr/local/mscproj/CygnusAMerger_NFW/data/20161108/"

    # /scratch/martyndv/cygnus/combined/spectral/maps/radial/sn100/cygA_plots
    # Last edit: Oct 18 09:27 (CygA), and Oct 18 11:37 (CygNW).
    # Edit by TLRH after copy:
        # header of datafile: i) removed spaces, ii) renamed Error to avoid double
    # 252 bins (CygA). Radius1, Radius2, SB, SBError, BGRD, BGRDError, AREA
    # 36 bins (CygNW)
    sb_file = datadir+"{0}_sb_sn100.dat".format(name)
    sbresults = ascii.read(sb_file)

    # /scratch/martyndv/cygnus/combined/spectral/maps/radial/pressure_sn100
    # Last edit: Nov  2 14:16 (CygA), and Nov  2 14:21 (CygNW).
    # 252 bins (CygA). Volume, Temperature, number density, Pressure, Compton-Y
    # Edit by TLRH after copy: removed '|' at beginning and end of each line
    # Override because datafile has a messy header
    ne_file = datadir+"{0}_sn100_therm_profile.dat".format(name)
    header = ["Bin", "V", "kT", "fkT", "n", "fn", "P", "fP", "Yparm"]
    neresults = ascii.read(ne_file, names=header, data_start=1)

    avg = astropy.table.hstack([sbresults, neresults])
    return avg

def parse_chandra_sectors():
    """ hot/cold/merger profiles (data copied at 20161108) """
    datadir = "/usr/local/mscproj/CygnusAMerger_NFW/data/20161108/"

    # /scratch/martyndv/cygnus/combined/spectral/maps/sector/plots
    # Last edit: Oct 18 12:33
    # fitresults = datadir+"cygnus_sector_fitresults.dat"

    # /scratch/martyndv/cygnus/combined/spectral/maps/sector/pressure/
    # Last edit: Oct 18 12:26
    # Edit by TLRH after copy: removed '|' at beginning and end of each line
        # Also cleaned up the header
    sb_file = datadir+"cygnus_sector_sn100_sbprofile.dat"
    sbresults = ascii.read(sb_file)

    # /scratch/martyndv/cygnus/combined/spectral/maps/sector/pressure
    # Last edit:  Nov  2 14:35
    # Edit by TLRH after copy: removed '|' at beginning and end of each line
    # Override because datafile has a messy header
    ne_file = datadir+"cygnus_sector_therm_profile.dat"
    header = ["Bin", "V", "kT", "fkT", "n", "fn", "P", "fP", "Yparm"]
    neresults = ascii.read(ne_file, names=header, data_start=1)

    sector = astropy.table.hstack([sbresults, neresults])
    merger = sector[0:164]  # careful with indices when looking at raw data
    hot = sector[164:364]   # astropy Table removes header, so off-by-one
    cold = sector[364:]

    return merger, hot, cold
# ----------------------------------------------------------------------------
