# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import astropy
from astropy.io import ascii
from astropy.io import fits

from macro import p2


# ----------------------------------------------------------------------------
# Parse Chandra observation: i) quiescent/non-merger; ii) sector analysis
# ----------------------------------------------------------------------------
def chandra_quiescent(basedir, name, data="1Msec"):
    """ `quiescent', or average profile (data copied at 20161108, and 20171030) """
    if data == "2Msec":
        datadir = basedir+"/CygnusAMerger_NFW/data/20171030/"

        # /scratch/martyndv/cygnus/combined/spectral/maps/radial/cygA_sn125/results
        # /scratch/martyndv/cygnus/combined/spectral/maps/radial/cygB_sn75/results
        # Last edit: Oct 18 (2017!) 13:17 (CygA), and Oct 20 (2017!) 11:37 (CygNW).
        # Edit by TLRH after copy:
            # header of datafile: i) removed spaces, ii) renamed Error to avoid double
        # 313 bins (CygA). Radius1, Radius2, SB, SBError, BGRD, BGRDError, Bin
        # 105 bins (CygNW)
        if name == "cygA":
            sb_file = datadir+"cygA_radial_sn125_sbprofile.dat"
        if name == "cygNW":
            sb_file = datadir+"cygNW_sn75_sbprofile.dat"
        sbresults = ascii.read(sb_file)

        # /scratch/martyndv/cygnus/combined/spectral/maps/radial/cygA_sn125/results
        # /scratch/martyndv/cygnus/combined/spectral/maps/radial/cygB_sn75/results
        # Last edit: Oct 18 (2017!) 13:18 (CygA), and Oct 20 (2017!) 11:31 (CygNW).
        # 312 bins (CygA). Bin, Volume, Temperature, fkT, number density, fn, Pressure, fP, Compton-Y
        # Edit by TLRH after copy: removed '|' at beginning and end of each line
        # Override because datafile has a messy header
        if name == "cygA":
            ne_file = datadir+"cygA_sn125_therm_profile.dat"
        if name == "cygNW":
            ne_file = datadir+"cygNW_sn75_therm_profile.dat"
        header = ["Bin", "V", "kT", "fkT", "n", "fn", "P", "fP", "Yparm"]
        neresults = ascii.read(ne_file, names=header, data_start=1)

        avg = astropy.table.hstack([sbresults, neresults])
    else:  # data is 1.03 Msec
        datadir = basedir+"/CygnusAMerger_NFW/data/20161108/"

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

def chandra_sectors(basedir, data="1Msec"):
    """ hot/cold/merger profiles (data copied at 20161108, and 20171030) """

    if data == "2Msec":
        datadir = basedir+"/CygnusAMerger_NFW/data/20171030/"

        # /scratch/martyndv/cygnus/combined/spectral/maps/sector/profiles
        # Last edit:  Oct ~11-13 (2017!)
        # Edit by TLRH after copy: removed '|' at beginning and end of each line
            # Also cleaned up the header
        sb_file = datadir+"cygnus_sector_sn125_sbprofile.dat"
        sb_file = datadir+"cygnus_sector_sn100_sbprofile.dat"
        sbresults = ascii.read(sb_file)

        # /scratch/martyndv/cygnus/combined/spectral/maps/sector/profiles
        # Last edit:  Oct ~11-13 (2017!)
        ne_file = datadir+"cygnus_sector_sn125_fitresults"
        ne_file = datadir+"cygnus_sector_sn100_fitresults"
        # TODO!
        header = ["Bin", "V", "kT", "fkT", "n", "fn", "P", "fP", "Yparm"]
        neresults = ascii.read(ne_file, names=header, data_start=1)

        sector = astropy.table.hstack([sbresults, neresults])
        merger = sector[0:164]  # careful with indices when looking at raw data
        hot = sector[164:364]   # astropy Table removes header, so off-by-one
        cold = sector[364:]
    else:  # data is 1.03 Msec
        datadir = basedir+"/CygnusAMerger_NFW/data/20161108/"

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

# ----------------------------------------------------------------------------
# Write Toycluster parameter, eat Toycluster output
# ----------------------------------------------------------------------------
def toycluster_parameterfile(icparms):
    ics = """% % Toycluster Parameter File %%
% % {description:s} %%

Output_file ./IC_single_0   % Base name

Ntotal      2000000 % Total Number of Particles in R200
Mtotal      {Mtotal:05.0f}   % Total Mass in Code Units

Mass_Ratio  {Mass_Ratio:.4f}  % set =0 for single cluster

ImpactParam 0
ZeroEOrbitFrac 0

Cuspy       1        % Use cuspy model (rc /= 10)

beta_0      {beta_0:0.3f}
beta_1      {beta_1:0.3f}

Redshift    0.0562

Bfld_Norm   1e-6        % B(r) = B0 * normalised_density^eta
Bfld_Eta    0.5         % like Bonafede 2010. B0 /=2 if Mtotal<5d4
Bfld_Scale  1

%bf          0.17     % bf in r200, bf = 17% ~ 14% in r500
h_100       0.7      % HubbleConstant/100

%Units
UnitLength_in_cm             3.085678e21        %  1.0 kpc
UnitMass_in_g                1.989e43           %  1.0e10 solar masses
UnitVelocity_in_cm_per_s     1e5                %  1 km/sec

%% -DGIVEPARAMS Options
%% here some more merger parameters can be set by hand

% cluster 0 is {name_0:s}
c_nfw_0           {c_nfw_0:.2f}
v_com_0           0
rc_0              {rc_0:.3f}
bf_0              {bf_0:.5f}
rcut_r200_ratio_0 {rcut_r200_ratio_0:.5f}

% cluster 1 {name_1:s}
c_nfw_1           {c_nfw_1:.2f}
v_com_1           0
rc_1              {rc_1:.2f}
bf_1              {bf_1:.5f}
rcut_r200_ratio_1 {rcut_r200_ratio_1:.5f}

%% -DADD_THIRD_SUBHALO Options

%SubFirstMass 1e12

%SubFirstPos0 0
%SubFirstPos1 0
%SubFirstPos2 0

%SubFirstVel0 0
%SubFirstVel1 0
%SubFirstVel2 0

%% -DDOUBLE_BETA_COOL_CORES Options

%Rho0_Fac      50    % increase in Rho0
%Rc_Fac        40    % decrease in Rcore"""
    return ics.format(**icparms)


def write_toycluster_parameterfile(icparms):
    confirm = raw_input("Are you sure you want to overwrite"
            +" '{filename:s}'?\n[yY/nN]: ".format(**icparms))
    if confirm.lower() == "y":
        print "Alright then: overwriting..."
    else:
        print "Alright then: aborted!"
        return
    with open(icparms["filename"], "w") as f:
        ics = toycluster_parameterfile(icparms)
        f.write(ics)
        print "... and done!"


def read_toycluster_parameterfile(filename):
    """ Eat toycluster parameter file, return ordered dictionary """
    parameters = OrderedDict()

    with open(filename, "r") as f:
        for line in f:
            # Ignore commented lines
            if len(line) > 1 and not line.strip().startswith("%"):
                line = line.strip().split("%")[0]  # Ignore comments in lines
                keyvaluepair = line.split()
                if keyvaluepair[0] != "Output_file":
                    parameters[keyvaluepair[0]] = float(keyvaluepair[1])
                else:
                    parameters[keyvaluepair[0]] = keyvaluepair[1]

    return parameters


def read_toycluster_makefile(filename):
    options = OrderedDict()

    counter = 0
    with open(filename, "r") as f:
        for line in f:
            # Ignore all lines that do not start with "OPT"
            if len(line) > 1 and line.strip().startswith("OPT"):
                if "OPTIMIZE" in line: continue
                line = line.strip().split("#")[0]  # Ignore comments in lines
                keyvaluepair = line.split("+=")
                options[keyvaluepair[0].strip()+str(counter)] = \
                    keyvaluepair[1].strip()
                counter += 1

    return options


def toycluster_profiles(filename):
    with open(filename, "r") as f:
        s = f.readlines()
        s[0] = (s[0][1:]).replace(",", "")
        return ascii.read(s, header_start=0, data_start=1)


def set_header(f, verbose=False):
    if verbose: print "    Parsing block as header"
    blocklength = numpy.fromfile(f, dtype=numpy.uint32, count=1)

    header = dict()

    header["npart"] = numpy.fromfile(f, dtype=numpy.uint32, count=6)
    header["ngas"] = header["npart"][0]
    header["ndm"] = header["npart"][1]
    header["ntot"] = numpy.sum(header["npart"])
    header["massarr"] = numpy.fromfile(f, dtype=numpy.float64, count=6)
    header["time"] = numpy.fromfile(f, dtype=numpy.float64, count=1)[0]
    header["redshift"] = numpy.fromfile(f, dtype=numpy.float64, count=1)[0]
    # unused in public version of GADGET-2
    header["flag_sfr"] = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]
    header["flag_feedback"] = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]  # unused
    header["npartTotal"] = numpy.fromfile(f, dtype=numpy.int32, count=6)
    header["flag_cooling"] = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]
    header["numFiles"] = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]
    header["boxSize"] = numpy.fromfile(f, dtype=numpy.float64, count=1)[0]
    header["omega0"] = numpy.fromfile(f, dtype=numpy.float64, count=1)[0]
    header["omegalambda"] = numpy.fromfile(f, dtype=numpy.float64, count=1)[0]
    header["hubbleParam"] = numpy.fromfile(f, dtype=numpy.float64, count=1)[0]
    header["flag_age"] = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]  # unused
    header["flag_metals"] = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]  # unused
    header["numpart_total_hw"] = numpy.fromfile(f, dtype=numpy.int32, count=6)  # unused

    header["bytesleft"] = 256-6*4 - 6*8 - 8 - 8 - 2*4-6*4 - 4 - 4 - 8 - 8 - 8 - 8 - 4 - 4 - 6*4
    header["la"] = numpy.fromfile(f, dtype=numpy.uint16, count=header["bytesleft"]/2)

    blocklength_end = numpy.fromfile(f, dtype="uint32", count=1)
    if blocklength != blocklength_end:
        print "ERROR: blocklengths differ"

    if verbose:
        print "\n      Header"
        for k, v in header.iteritems(): print " "*8+"{0:<15}: {1}".format(k, v)
        print

    return header


def toycluster_icfile(filename, block="all", verbose=False):
    """ Eat Toycluster/Gadget-2/Gadget-3 output

        Inspired by https://stackoverflow.com/questions/23377274

        See Gadget-2 user guide for details. The data blocks are:
            Header
            Block 0 (Coordinates)
            Block 1 (Velocities)
            Block 2 (ParticleIDs)
            Block 3 (Density)         <-- gas only
            Block 4 (Model Density)   <-- gas only (Toycluster only; not Gadget2)
            Block 5 (SmoothingLength) <-- gas only
            Block 6 (InternalEnergy)  <-- gas only
            Block 7 (MagneticField)   <-- gas only (Toycluster only; not Gadget2)

        @param filename: path to the Binary Fortran 77 Unformatted file
        @param verbose : output block info (length/name)
        @return        : The function returns a 3-tuple with the following elements
                         header -> dict
                         gas    -> astropy.table.Table
                         dm     -> astropy.table.Table """

    if verbose: print "\nParsing ICs generated by Toycluster 2.0/NFW"

    def eat_block(f, dtype="int8"):
        """ Read Fortran 77 unformatted block
            @param f:     open file buffer, file
            @param dtype: data type (int8, float32, etc)
            @return:      list that contains the block content """
        length = numpy.fromfile(f, dtype=numpy.uint32, count=1)
        if not length: return None  # EOF reached
        if verbose: print "  Eating block of length:", length[0]
        # float32/uint32 is 4 bytes so count=length/4 in for non-byte dtype
        content = numpy.fromfile(f, dtype=dtype,
            count=length if dtype == "int8" else length/4)
        end_length = numpy.fromfile(f, dtype=numpy.uint32, count=1)

        if end_length != length:
            print "  ERROR: blocklengths differ"
            return None
        return content

    def name_block(block):
        """ Translate Fortran 77 unformatted block name from ascii values to str
            @param block: list of ascii values
            @return:      name of block, string """
        name = ""
        for i, char in enumerate(block):
            if i == 4:
                break
            if 65 <= char <= 90:
                name += chr(char)
            else:
                break
        if verbose and len(name) is not 0: print "    Block name:", name
        return name

    def set_pos_or_vel(f, name):
        """ float32; first gas (x,y,z) then dm (x,y,z) """
        name = "" if name == "POS" else "v"
        if verbose: print "    Parsing block as position"
        block = eat_block(f, dtype="float32")
        block = block.reshape((header["ntot"], 3))
        gaspart = block[0:header["ngas"]]
        dmpart = block[header["ngas"]:header["ntot"]]
        # NB radius has to be calculated after shifting back haloes (-boxsize/2)
        gas[name+"x"] = gaspart[:,0]
        gas[name+"y"] = gaspart[:,1]
        gas[name+"z"] = gaspart[:,2]
        dm[name+"x"] = dmpart[:,0]
        dm[name+"y"] = dmpart[:,1]
        dm[name+"z"] = dmpart[:,2]

    def set_id(f, name=None):
        """ uint32, first gas then dm """
        if verbose: print "    Parsing block as id"
        block = eat_block(f, dtype="uint32")
        gas["id"] = block[0:header["ngas"]]
        dm["id"] = block[header["ngas"]:header["ntot"]]

    def set_gas_float32(f, name):
        """ float 32, gas only: rho/hsml/u and rhom (Toycluster only) """
        if verbose: print "    Parsing block as gas-only float32"
        gas[name.lower()] = eat_block(f, dtype="float32")

    def set_magnetic_field(f, name=None):
        """ float32, gas only, Toycluster only (not Gadget-2) """
        if verbose: print "    Parsing block as magnetic field"
        block = eat_block(f, dtype="float32")
        block = block.reshape((header["ngas"], 3))
        gas["Bx"] = block[:,0]
        gas["By"] = block[:,1]
        gas["Bz"] = block[:,2]

    def void_block(f, name):
        """ read but not save block """
        if verbose: print "    Voiding block as gas-only float32"
        length = numpy.fromfile(f, dtype=numpy.uint32, count=1)
        if not length: return None  # EOF reached
        if verbose: print "    Blocklength:", length[0]
        f.seek(length[0], 1)  # 1 means relative to current position
        end_length = numpy.fromfile(f, dtype=numpy.uint32, count=1)

        if end_length != length:
            print "  ERROR: blocklengths differ"
            return None
        if verbose: print "    WARNING: block '{0}' is not saved!".format(name)

    header = dict()
    gas = astropy.table.Table()
    dm = astropy.table.Table()
    # Map name of block to a function pointer that handles parsing the data
    routines = { "HEAD": set_header, "POS": set_pos_or_vel, "VEL": set_pos_or_vel,
                 "ID": set_id, "RHO": set_gas_float32, "RHOM":  set_gas_float32,
                 "HSML": set_gas_float32,  "U": set_gas_float32, "BFLD": set_magnetic_field,
                 "DIVB": void_block, "ACVC": void_block, "AMDC": void_block,
                 "MACH": void_block, "SHSP": void_block, "SHCP": void_block,
                 "SHNR": void_block, "SHRH": void_block, "SHPR": void_block,
                 "SHVU": void_block, "VRMS": void_block, "VDIV": void_block,
                 "VROT": void_block, "INFO": void_block }

    with open(filename, "rb") as f:
        while True:
            block = eat_block(f)  # data is preceded by byte /w name of block
            if block is None: break  # EOF reached, or error reading block
            blockname = name_block(block)
            if blockname and blockname == "HEAD":
                header = set_header(f, verbose=verbose)
            elif blockname:
                routines.get(blockname, void_block)(f, blockname)

    if verbose:
        print "\nGas"
        print gas
        print "\nDM"
        print dm

    return header, gas, dm


def eat_f77(snap, blockname, verbose=False):
     with open(snap, "rb") as f:
         while True:
             # Eat name
             length = numpy.fromfile(f, dtype=numpy.uint32, count=1)
             if not length: break
             content = numpy.fromfile(f, dtype="int8", count=length)
             end_length = numpy.fromfile(f, dtype=numpy.uint32, count=1)

             if end_length != length:
                 print "  ERROR: blocklengths differ"
                 break

             name = "".join(chr(c) for c in content[:4] if 65 <= c <= 90)
             if verbose: print "Eating", name

             if blockname.lower() == "head":
                 return set_header(f, verbose=verbose)

             length = numpy.fromfile(f, dtype=numpy.uint32, count=1)
             if not length: break
             if name.strip().lower() != blockname.lower():
                 f.seek(length[0], 1)
             else:
                 content = numpy.fromfile(f, dtype="float32", count=length/4)

                 return content
             end_length = numpy.fromfile(f, dtype=numpy.uint32, count=1)

             if end_length != length:
                 print "  ERROR: blocklengths differ"
                 break
     return None
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Eat Gadget-2 output
# ----------------------------------------------------------------------------
def read_gadget2_parms(filename):
    parameters = OrderedDict()
    string_parms = ["InitCondFile", "OutputDir", "EnergyFile", "InfoFile",
                    "TimingsFile", "CpuFile", "RestartFile", "SnapshotFileBase",
                    "OutputListFilename", "ResubmitCommand"]

    with open(filename, "r") as f:
        for line in f:
            # Ignore commented lines
            if len(line) > 1 and not line.strip().startswith("%"):
                line = line.strip().split("%")[0]  # Ignore comments in lines
                keyvaluepair = line.split()
                if keyvaluepair[0] not in string_parms:
                    parameters[keyvaluepair[0]] = float(keyvaluepair[1])
                else:
                    parameters[keyvaluepair[0]] = keyvaluepair[1]

    return parameters


# ----------------------------------------------------------------------------
# Eat Gadget-3 output
# ----------------------------------------------------------------------------
def read_gadget3_parms(filename):
    parameters = OrderedDict()
    string_parms = ["InitCondFile", "OutputDir", "EnergyFile", "InfoFile",
                    "TimingsFile", "CpuFile", "TimebinFile", "RestartFile",
                    "SnapshotFileBase", "OutputListFilename", "ResubmitCommand"]

    with open(filename, "r") as f:
        for line in f:
            # Ignore commented lines
            if len(line) > 1 and not line.strip().startswith("%"):
                line = line.strip().split("%")[0]  # Ignore comments in lines
                keyvaluepair = line.split()
                if keyvaluepair[0] not in string_parms:
                    parameters[keyvaluepair[0]] = float(keyvaluepair[1])
                else:
                    parameters[keyvaluepair[0]] = keyvaluepair[1]

    return parameters


# ----------------------------------------------------------------------------
# Eat P-Smac2 output
# ----------------------------------------------------------------------------
def psmac2_fitsfile(filename):
    """ Read fits data cube. """
    with fits.open(filename) as f:
        header = f[0].header
        data = f[0].data

    cleaned_header = OrderedDict()
    for line in repr(header).split("\n"):
        if " = " in line and " / " in line and "SUM" not in line:
            value, key = line.strip().split(" = ")[-1].strip().split(" / ")
            cleaned_header[key] = value

    return cleaned_header, data
