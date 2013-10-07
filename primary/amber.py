#!/usr/bin/python
desc = """amber.py
    Functions for primary analysis of AMBER trajectories
    Written by Karl Debiec on 12-12-01
    Last updated 13-10-07"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def log(segment, time_offset = 0.0, **kwargs):
    """ Parses log for <segment> """
    log         = segment[".out"]
    nstlim      = float(commands.getoutput("grep nstlim " + log).split()[2][:-1])
    ntpr        = float(commands.getoutput("grep ntpr   " + log).split()[2][:-1])
    dt          = float(commands.getoutput("grep dt     " + log).split()[2][:-1])
    length      = nstlim / ntpr
    with open(log, "r") as log:
        raw     = [line.strip() for line in log.readlines()]
    seg_attrs   = {}
    i           = 0
    data        = {}
    data_loaded = False
    
    while i < len(raw):
        line    = raw[i]
        if   line.startswith("A V E R A G E S"):
            data_loaded = True
        if   line.startswith("| Running AMBER/MPI version on"):
            line                    = line.split()
            seg_attrs["n_cores"]    = int(line[5])
        elif line.startswith("|  Master Total wall time"):
            seconds                 = int(line.split()[5])
            seg_attrs["duration"]   = "{0}:{1:02d}:{2:02d}".format(seconds//60//60, seconds//60%60, seconds%60)
        elif line.startswith("| Run on"):
            line                    = line.split()
            seg_attrs["date"]       = "{0:02d}-{1:02d}-{2:02d}".format(int(line[3][8:]), int(line[3][:2]),
                                                                       int(line[3][3:5]))
            seg_attrs["time"]       = line[5]
        elif line.startswith("NSTEP") and not data_loaded:
            while True:
                line    = raw[i]
                if line.startswith("------------------------------------------------------------------------------"):
                    break
                line    = line.split("=")
                for j in range(1,len(line)):
                    if j == 1:                  field   = line[j-1].strip()
                    else:
                        field                           = line[j-1].split()
                        if len(field)   == 1:   field   = field[0]
                        else:                   field   = " ".join(field[1:])
                    value   = float(line[j].split()[0])
                    if field in data:   data[field]    += [value]
                    else:               data[field]     = [value]
                i      += 1
        i  += 1
    data["TIME(PS)"]    = np.array(data["TIME(PS)"])  / 1000. + time_offset
    dtype_line      = "np.dtype([('time', 'f4'),"
    log_line        = "[tuple(frame) for frame in np.column_stack((data['TIME(PS)'],"
    attrs_line      = "{'time units': 'ns',"
    for field, dest in [("Etot",        "total"),
                        ("EPtot",       "potential"),
                        ("EKtot",       "kinetic"),
                        ("BOND",        "bond"),
                        ("ANGLE",       "angle"),
                        ("DIHED",       "dihedral"),
                        ("EELEC",       "coulomb"),
                        ("1-4 EEL",     "coulomb 1-4"),
                        ("VDWAALS",     "van der Waals"),
                        ("1-4 NB",      "van der Waals 1-4"),
                        ("EHBOND",      "hydrogen bond"),
                        ("RESTRAINT",   "position restraint"),
                        ("EKCMT",       "center of mass motion kinetic"),
                        ("VIRIAL",      "virial"),
                        ("EPOLZ",       "polarization")]:
        if not (field in data): continue
        data[field]     = np.array(data[field])
        dtype_line     += "('"     + dest  + "', 'f4'),"
        log_line       += "data['" + field + "'],"
        attrs_line     += "'"      + dest  + " units': 'kcal mol-1',"
    if "TEMP(K)" in data:
        data["TEMP(K)"] = np.array(data["TEMP(K)"])
        dtype_line     += "('temperature', 'f4'),"
        log_line       += "data['TEMP(K)'],"
        attrs_line     += "'temperature units': 'K',"
    if "PRESS" in data:
        data["PRESS"]   = np.array(data["PRESS"])
        dtype_line     += "('pressure', 'f4'),"
        log_line       += "data['PRESS'],"
        attrs_line     += "'pressure units': 'bar',"
    if "Dipole convergence: rms" in data:
        data["Dipole convergence: rms"] = np.array(data["Dipole convergence: rms"])
        dtype_line     += "('dipole converge rms', 'f4'),"
        log_line       += "data['Dipole convergence: rms'],"
    if "iters" in data:
        data["iters"]   = np.array(data["iters"])
        dtype_line     += "('iters', 'f4'),"
        log_line       += "data['iters'],"
    dtype_line  = dtype_line[:-1]   + "])"
    log_line    = log_line[:-1]     + "))]"
    attrs_line  = attrs_line[:-1]   + "}"
    log         = np.array(eval(log_line), eval(dtype_line))
    log_attrs   = eval(attrs_line)
    return [(segment + "/log",    log),
            (segment + "/",       seg_attrs),
            (segment + "/log",    log_attrs)]
def _check_log(hdf5_file, segment, require_trajectory_files = False, force = False, **kwargs):
    if      (require_trajectory_files
    and not (segment.topology   and os.path.isfile(segment.topology)
    and      segment.trajectory and os.path.isfile(segment.trajectory))):
            return False
    if    (force
    or not segment + "/log" in hdf5_file): return [(log, segment, kwargs)]
    else:                                  return False
