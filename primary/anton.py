#!/usr/bin/python
desc = """anton.py
    Functions for primary analysis of Anton trajectories
    Written by Karl Debiec on 12-11-29
    Last updated 13-05-12"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import month
################################################## ANALYSIS FUNCTIONS ##################################################
def log(segment, **kwargs):
    """ Parses log for <segment>; may be either calculated on Anton or generated afterwards using vrun (PENDING)"""
    log     = segment.file_of_type(".ene")
    head    = commands.getoutput("head -n 1 {0}".format(log)).split()
    if "vrun" in head:
        seg_attrs           = {"type": "vrun"}
        return None
    else:
        seg_attrs           = {"type": "ene"}
        seg_attrs["date"]   = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
        seg_attrs["time"]   = head[7]
        dtype       = np.dtype([("time",     "f"), ("total",    "f"), ("potential",        "f"),
                                ("kinetic",  "f"), ("exchange", "f"), ("force correction", "f"),
                                ("pressure", "f"), ("volume",   "f"), ("temperature",      "f")])
        log         = np.loadtxt(log, dtype, usecols = (0,1,2,3,4,5,6,7,8))
        log["time"]/= 1000
        log_attrs   = {"time ":    "ns",         "total":    "kcal mol-1", "potential":        "kcal mol-1",
                       "kinetic":  "kcal mol-1", "exchange": "kcal mol-1", "force correction": "kcal mol-1",
                       "pressure": "bar",        "volume":   "A3",         "temperature":      "K"}
    return [(segment + "/log",    log),
            (segment + "/",       seg_attrs),
            (segment + "/log",    log_attrs)]
def _check_log(hdf5_file, segment, **kwargs):
    if not (segment + "/log" in hdf5_file): return [(log, segment, kwargs)]
    else:                                   return False
