#!/usr/bin/python
desc = """desmond.py
    Functions for primary analysis of DESMOND trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-07-21"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import month
################################################## ANALYSIS FUNCTIONS ##################################################
def log(segment, **kwargs):
    """ Parses log for <segment> """
    log         = segment[".ene"]
    head        = commands.getoutput("head -n 3 {0}".format(log)).split("\n")[2].split()
    seg_attrs   = {"date": "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6])),
                   "time": head[7]}
    dtype       = np.dtype([("time",      "f4"), ("total",       "f4"),  ("potential",        "f4"), ("kinetic",  "f4"),
                            ("conserved", "f4"), ("exchange",    "f4"),  ("force correction", "f4"), ("pressure", "f4"),
                            ("volume",    "f4"), ("temperature", "f4")])
    log         = np.loadtxt(log, dtype, usecols = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    log["time"]/= 1000
    log_attrs   = {"time units":             "ns",         "total units":       "kcal mol-1",
                   "potential units":        "kcal mol-1", "kinetic units":     "kcal mol-1",
                   "conserved units":        "kcal mol-1", "exchange units":    "kcal mol-1",
                   "force correction units": "kcal mol-1", "pressure units":    "bar",
                   "volume units":           "A3",         "temperature units": "K"}
    return [(segment + "/log",  log[1:]),
            (segment + "/log",  log_attrs),
            (segment + "/",     seg_attrs)]
def _check_log(hdf5_file, segment, require_trajectory_files = False, force = False, **kwargs):
    if      (require_trajectory_files
    and not (segment.topology   and os.path.isfile(segment.topology)
    and      segment.trajectory and os.path.isfile(segment.trajectory))):
            return False
    if     (force
    or not (segment + "/log" in hdf5_file)): return [(log, segment, kwargs)]
    else:                                    return False
