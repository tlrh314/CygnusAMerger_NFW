import astropy
from matplotlib import pyplot

import convert
from cosmology import CosmologyCalculator


class ObservedCluster(object):
    """ Parse and store Chandra XVP (PI Wise) observation """
    def __init__(self, name):
        """ Read in the quiescent radial profiles of CygA/CygNW afer 1.03 Msec
            Chandra XVP observations (PI Wise). Data courtesy of M.N. de Vries.
            Files are copied over from Struis account martyndv.

            Available profiles: density, metallicity, pressure, temperature, Y
            Both average sector, and hot/cold/merger sectors available
        """

        self.name = name

        # Redshift of Cygnus cluster Owen+ 1997. CygNW might have different z.
        # We adopt concordance cosmology with generic cosmological parameters
        self.cc = CosmologyCalculator(z=0.0562, H0=70, WM=0.3, WV=0.7)

        self.parse_quiescent()
        if self.name == "cygA":
            # no have sectoranalysis for CygNW
            self.parse_sectors()

    def __str__(self):
        return str(self.avg)

    def parse_quiescent(self):
        """ `quiescent', or average profile (data copied at 20161108) """
        datadir = "/usr/local/mscproj/CygnusAMerger_NFW/data/20161108/"

        # /scratch/martyndv/cygnus/combined/spectral/maps/radial/sn100/cygA_plots
        # Last edit: Oct 18 09:27 (CygA), and Oct 18 11:37 (CygNW).
        # Edit by TLRH after copy:
            # header of datafile: i) removed spaces, ii) renamed Error to avoid double
        # 252 bins (CygA). Radius1, Radius2, SB, SBError, BGRD, BGRDError, AREA
        # 36 bins (CygNW)
        sb_file = datadir+"{0}_sb_sn100.dat".format(self.name)
        sbresults = astropy.io.ascii.read(sb_file)

        # /scratch/martyndv/cygnus/combined/spectral/maps/radial/pressure_sn100
        # Last edit: Nov  2 14:16 (CygA), and Nov  2 14:21 (CygNW).
        # 252 bins (CygA). Volume, Temperature, number density, Pressure, Compton-Y
        # Edit by TLRH after copy: removed '|' at beginning and end of each line
        # Override because datafile has a messy header
        ne_file = datadir+"{0}_sn100_therm_profile.dat".format(self.name)
        header = ["Bin", "V", "kT", "fkT", "n", "fn", "P", "fP", "Yparm"]
        neresults = astropy.io.ascii.read(ne_file, names=header, data_start=1)

        self.avg = astropy.table.hstack([sbresults, neresults])
        self.set_radius(self.avg)

    def set_radius(self, t):
        # An adaptive binning routine is used for data extraction to ensure
        # SNR==100. Therefore binsizes vary, but error bars are equal.
        arcsec2kpc = self.cc.kpc_DA  # kpc
        t["radius"] = (t["Radius1"] + t["Radius2"])/2 * arcsec2kpc
        t["binsize"] = (t["Radius2"] - t["Radius1"]) * arcsec2kpc

    def parse_sectors(self):
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
        sbresults = astropy.io.ascii.read(sb_file)

        # /scratch/martyndv/cygnus/combined/spectral/maps/sector/pressure
        # Last edit:  Nov  2 14:35
        # Edit by TLRH after copy: removed '|' at beginning and end of each line
        # Override because datafile has a messy header
        ne_file = datadir+"cygnus_sector_therm_profile.dat"
        header = ["Bin", "V", "kT", "fkT", "n", "fn", "P", "fP", "Yparm"]
        neresults = astropy.io.ascii.read(ne_file, names=header, data_start=1)

        sector = astropy.table.hstack([sbresults, neresults])
        self.set_radius(sector)
        self.merger = sector[0:166]
        self.hot = sector[166:366]
        self.cold = sector[366:439]

    def plot_chandra_average(self, parm="kT", alpha=0.2):
        """ plot of observed average profile of parm """
        pyplot.errorbar(self.avg["radius"], self.avg[parm],
                        xerr=self.avg["binsize"]/2,
                        yerr=[self.avg["f"+parm], self.avg["f"+parm]],
                        marker="o", ls="", c="b", ms=4,
                        alpha=alpha, elinewidth=2, label="Average "+parm)

    def plot_chandra_sector(self, parm="kT", merger=True, hot=True, cold=True, alpha=0.2):
        if merger:
            pyplot.errorbar(self.merger["radius"], self.merger[parm],
                            xerr=self.merger["binsize"]/2,
                            yerr=[self.merger["f"+parm], self.merger["f"+parm]],
                            marker="o", ls="", c="g", ms=4, alpha=alpha,
                            elinewidth=2, label="Merger "+parm)
        if hot:
            pyplot.errorbar(self.hot["radius"], self.hot[parm],
                            xerr=self.hot["binsize"]/2,
                            yerr=[self.hot["f"+parm], self.hot["f"+parm]],
                            marker="o", ls="", c="r", ms=6, alpha=alpha,
                            elinewidth=2, label="Hot "+parm)
        if cold:
            pyplot.errorbar(self.cold["radius"], self.cold[parm],
                            xerr=self.cold["binsize"]/2,
                            yerr=[self.cold["f"+parm], self.cold["f"+parm]],
                            marker="o", ls="", c="purple", ms=6, alpha=alpha,
                            elinewidth=2, label="Cold "+parm)

if __name__ == "__main__":
    cygA = ObservedCluster("cygA")
    pyplot.figure(figsize=(12, 9))
    cygA.plot_chandra_average(alpha=1)
    cygA.plot_chandra_sector(alpha=1)
    pyplot.xlabel("Radius [kpc]")
    pyplot.xscale("log")
    pyplot.xlim(1, 1000)
    pyplot.ylabel("kT [keV]")
    pyplot.ylim(2, 11)
    pyplot.legend(loc="upper left")
    pyplot.show()
    cygNW = ObservedCluster("cygNW")
