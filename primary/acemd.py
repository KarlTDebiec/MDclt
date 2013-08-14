#!/usr/bin/python
desc = """acemd.py
    Functions for primary analysis of ACEMD trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-08-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import month
################################################## ANALYSIS FUNCTIONS ##################################################
def log(segment, time_offset = 0.0, **kwargs):
    """ Parses log for <segment> """
    log         = segment[".log"]
    conf        = segment[".conf"]
    command     = "grep \"timestep                            \"  {0}".format(conf)
    timestep    = float(commands.getoutput(command).split()[1]) / 1000000
    dtype       = np.dtype([("time",     "f4"), ("bond",          "f4"), ("angle",       "f4"), ("dihedral", "f4"),
                            ("coulomb",  "f4"), ("van der Waals", "f4"), ("potential",   "f4"), ("kinetic",  "f4"),
                            ("external", "f4"), ("total",         "f4"), ("temperature", "f4"), ("pressure", "f4")])
    log          = np.genfromtxt(log, dtype, skip_header = 1, invalid_raise = False)
    log["time"]  = log["time"] * timestep + + timestep + time_offset
    log_attrs   = {"time units":        "ns",         "bond units":          "kcal mol-1",
                   "angle units":       "kcal mol-1", "dihedral units":      "kcal mol-1",
                   "coulomb units":     "kcal mol-1", "van der Waals units": "kcal mol-1",
                   "potential units":   "kcal mol-1", "kinetic units":       "kcal mol-1",
                   "external units":    "kcal mol-1", "total units":         "kcal mol-1",
                   "temperature units": "K",          "pressure units":      "Bar"}
    return [(segment + "/log",  log[1:]),
            (segment + "/log",  log_attrs)]
def _check_log(hdf5_file, segment, require_trajectory_files = False, force = False, **kwargs):
    if      (require_trajectory_files
    and not (segment.topology   and os.path.isfile(segment.topology)
    and      segment.trajectory and os.path.isfile(segment.trajectory))):
            return False
    if     (force
    or not (segment + "/log" in hdf5_file)):
            kwargs["time_offset"] = float(segment)
            return [(log, segment, kwargs)]
    else:   return False
