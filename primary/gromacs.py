#!/usr/bin/python
desc = """MD_toolkit.primary.gromacs.py
    Functions for primary analysis of GROMACS trajectories
    Written by Karl Debiec on 12-11-30
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   MD_toolkit.standard_functions import is_num, month
################################################## ANALYSIS FUNCTIONS ##################################################
def log(segment, time_offset = 0.0, **kwargs):
    """ Parses log for <segment> """
    log         = segment[".log"]
    nsteps      = float(commands.getoutput("grep nsteps " + log).split()[2])
    nstlog      = float(commands.getoutput("grep nstlog " + log).split()[2])
    length      = nsteps / nstlog
    with open(log, "r") as log:
        raw     = [line.rstrip() for line in log.readlines()]
    seg_attrs   = {"decomposition": np.zeros((2, 3), dtype = np.int)}
    i           = 0
    data        = {}
    while i < len(raw):
        line    = raw[i]
        if   line.startswith("Domain decomposition grid"):
            line                            = line.split()
            seg_attrs["decomposition"][0]   = int(line[3]), int(line[5]), int(line[7][:-1])
        elif line.startswith("PME domain decomposition"):
            line                            = line.split()
            seg_attrs["decomposition"][1]   = int(line[3]), int(line[5]), int(line[7])
        elif line.startswith("       Time:"):
            seconds                         = int(float(line.split()[2]))
            seg_attrs["duration"]           = "{0}:{1:02d}:{2:02d}".format(seconds//60//60, seconds//60%60, seconds%60)
        elif line.startswith("Finished mdrun on"):
            line                            = line.split()
            seg_attrs["date"]               = "{0:02d}-{1:02d}-{2:02d}".format(int(line[9][-2:]), month(line[6]),
                                                                               int(line[7]))
            seg_attrs["time"]               = line[8]
            break
        elif line.startswith("           Step"):
            line2   = raw[i+1]
            k       = 0
            for k in range(6):
                field   = line[k*15:(k+1)*15].strip()
                if field == "": break
                value   = float(line2[k*15:(k+1)*15])
                if field in data:   data[field]    += [value]
                else:               data[field]     = [value]
            i      += 1
        elif line.startswith("   Energies (kJ/mol)"):
            i      += 1
            while True:
                line    = raw[i]
                line2   = raw[i+1]
                if line == "":  break
                for j in range(6):
                    field   = line[j*15:(j+1)*15].strip()
                    if field == "": break
                    value   = float(line2[j*15:(j+1)*15])
                    if field in data:   data[field]    += [value]
                    else:               data[field]     = [value]
                i      += 2
        i  += 1
    data["Time"]    = np.array(data["Time"])[1:]  / 1000. + time_offset
    dtype_line      = "np.dtype([('time', 'f4'),"
    log_line        = "[tuple(frame) for frame in np.column_stack((data['Time'],"
    attrs_line      = "{'time units': 'ns',"
    for field, dest in [("Total Energy",     "total"),
                        ("Potential",        "potential"),
                        ("Kinetic En.",      "kinetic"),
                        ("Conserved En.",    "conserved"),
                        ("Bond",             "bond"),
                        ("Constr. rmsd",     "constraint rmsd"),
                        ("Constr.2 rmsd",    "constraint 2 rmsd"),
                        ("Angle",            "angle"),
                        ("U-B",              "Urey-Bradley"),
                        ("Proper Dih.",      "proper dihedral"),
                        ("Improper Dih.",    "improper dihedral"),
                        ("CMAP Dih.",        "CMAP"),
                        ("Coulomb (SR)",     "coulomb short-range"),
                        ("Coul. recip.",     "coulomb long-range"),
                        ("Coulomb-14",       "coulomb 1-4"),
                        ("LJ (SR)",          "van der Waals short-range"),
                        ("LJ-14",            "van der Waals 1-4"),
                        ("Disper. corr.",    "van der Waals dispersion correction"),
                        ("Position Rest.",   "position restraint")]: 
        if not (field in data): continue
        data[field]             = np.array(data[field])[1:-1] * 0.239005736     # Convert from kJ to kcal
        dtype_line             += "('"     + dest  + "','f4'),"
        log_line               += "data['" + field + "'],"
        attrs_line             += "'"      + dest  + " units':'kcal mol-1',"
    if "Temperature" in data:
        data["Temperature"]     = np.array(data["Temperature"])[1:-1]
        dtype_line             += "('temperature','f4'),"
        log_line               += "data['Temperature'],"
        attrs_line             += "'temperature units':'K',"
    if "Pressure (bar)" in data:
        data["Pressure (bar)"]  = np.array(data["Pressure (bar)"])[1:-1]
        dtype_line             += "('pressure','f4'),"
        log_line               += "data['Pressure (bar)'],"
        attrs_line             += "'pressure units':'bar',"
    if "Pres. DC (bar)" in data:
        data["Pres. DC (bar)"]  = np.array(data["Pres. DC (bar)"])[1:-1]
        dtype_line             += "('pressure dispersion correction','f4'),"
        log_line               += "data['Pres. DC (bar)'],"
        attrs_line             += "'pressure dispersion correction units':'bar',"
    dtype_line  = dtype_line[:-1]   + "])"
    log_line    = log_line[:-1]     + "))]"
    attrs_line  = attrs_line[:-1]   + "}"
    log         = np.array(eval(log_line), eval(dtype_line))
    log_attrs   = eval(attrs_line)
    return  [(segment + "/log",   log),
             (segment + "/",      seg_attrs),
             (segment + "/log",   log_attrs)]
def _check_log(hdf5_file, segment, require_trajectory_files = False, force = False, **kwargs):
    if      (require_trajectory_files
    and not (segment.topology   and os.path.isfile(segment.topology)
    and      segment.trajectory and os.path.isfile(segment.trajectory))):
            return False
    if    (force
    or not segment + "/log" in hdf5_file):
            kwargs["time_offset"] = float(segment)
            return [(log, segment, kwargs)]
    else:   return False
