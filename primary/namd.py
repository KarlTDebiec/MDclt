#!/usr/bin/python
#   MD_toolkit.primary.namd.py
#   Written by Karl Debiec on 14-06-06, last updated by Karl Debiec on 14-06-06
"""
Functions for primary analysis of NAMD trajectories
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
###################################################### FUNCTIONS #######################################################
def log(segment, **kwargs):
    """
    Parses log for <segment>

    .. todo:
        - Implement unidentified fields
    """
    fieldnames = None
    seg_attrs  = {}
    with open(segment[".log"]) as log:
        for line in log.readlines():
            if   line.startswith("Info: TIMESTEP"):
                timestep   = float(line.split()[2])
            elif line.startswith("TCL: Running for"):
                nsteps     = float(line.split()[3])
            elif line.startswith("WallClock:"):
                seconds    = int(float(line.split()[1]))
                seg_attrs["duration"] = "{0}:{1:02d}:{2:02d}".format(seconds//60//60, seconds//60%60, seconds%60)
            elif fieldnames is None and line.startswith("ETITLE"):
                fieldnames = line.split()[1:]
                data       = {field: [] for field in fieldnames}
            elif line.startswith("ENERGY"):
                for field, value in zip(fieldnames, line.split()[1:]):
                    data[field] += [value]
    for key, value in data.items():
        data[key]   = np.array(value, np.float32)
    data["TS"]     *= timestep / 1000000
    formatted_data  = [(data["TS"])]
    dtype           = [("time", "f4")]
    attrs           = {"time units": "ns"}
    field_info      = [("TOTAL",     "total energy",     "kcal mol-1"),
                       ("POTENTIAL", "potential energy", "kcal mol-1"),
                       ("KINETIC",   "kinetic energy",   "kcal mol-1"),
                       ("BOND",      "bond",             "kcal mol-1"),
                       ("ANGLE",     "angle",            "kcal mol-1"),
                       ("DIHED",     "dihedral",         "kcal mol-1"),
                       ("IMPRP",     "improper",         "kcal mol-1"),
                       ("CROSS",     "cmap",             "kcal mol-1"),
                       ("ELECT",     "electrostatic",    "kcal mol-1"),
                       ("VDW",       "van der Waals",    "kcal mol-1"),
                       ("TEMP",      "temperature",      "K"),
                       ("PRESSURE",  "pressure",         "bar"),
                       ("VOLUME",    "volume",           "A^3"),
                       ("DRUDEBOND", "drude bond",       "kcal mol-1"),
#                       ("BOUNDARY",  "?",                "?"),
#                       ("MISC",      "?",                "?"),
#                       ("TOTAL3",    "?",                "?"),
#                       ("TEMPAVG",   "?",                "?"),
#                       ("GPRESSURE", "?",                "?"),
#                       ("PRESSAVG",  "?",                "?"),
#                       ("GPRESSAVG", "?",                "?"),
#                       ("DRBONDAVG", "?",                "?")
                      ]
    for source, destination, units in field_info:
        if source in data:
            formatted_data += [data[source]]
            dtype          += [(destination, "f4")]
            attrs["{0} units".format(destination)] = units
    data = np.array([tuple(frame) for frame in np.column_stack(formatted_data)[1:]], dtype)
    return [(segment + "/log",  data),
            (segment + "/log",  attrs),
            (segment + "/",     seg_attrs)]
    return []
def _check_log(hdf5_file, segment, require_trajectory_files = False, force = False, **kwargs):
    """
    """
    if      (require_trajectory_files
    and not (segment.topology   and os.path.isfile(segment.topology)
    and      segment.trajectory and os.path.isfile(segment.trajectory))):
        return False
    if     (force
    or not (segment + "/log" in hdf5_file)): return [(log, segment, kwargs)]
    else:                                    return False


