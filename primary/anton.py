#!/usr/bin/python
desc = """anton.py
    Functions for primary analysis of Anton trajectories
    Written by Karl Debiec on 12-11-29
    Last updated 13-05-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(segment, **kwargs):
    """ Parses log for <segment>; may be either calculated on Anton or generated afterwards using vrun (PENDING)"""
    ene     = segment.file_of_type(".ene")
    data    = np.loadtxt(ene)
    head    = commands.getoutput("head -n 1 {0}".format(ene)).split()
    time    = data[1:,0] / 1000.
    if "vrun" in head:
        seg_attrs           = {"type": "vrun"}
    else:
        seg_attrs           = {"type": "ene"}
        seg_attrs["date"]   = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
        seg_attrs["time"]   = head[7]
        log         = [tuple(frame) for frame in np.column_stack((time, data[1:,1:9]))]
        dtype       = np.dtype([("time",     "f"), ("total",    "f"), ("potential",        "f"),
                                ("kinetic",  "f"), ("exchange", "f"), ("force correction", "f"),
                                ("pressure", "f"), ("volume",   "f"), ("temperature",      "f")])
        log         = np.array(log, dtype)
        log_attrs   = {"time ":    "ns",         "total":    "kcal mol-1", "potential":        "kcal mol-1",
                       "kinetic":  "kcal mol-1", "exchange": "kcal mol-1", "force correction": "kcal mol-1",
                       "pressure": "bar",        "volume":   "A3",         "temperature":      "K"}
    return [("/" + segment + "/time",   time),
            ("/" + segment + "/log",    log),
            ("/" + segment,             seg_attrs),
            ("/" + segment + "/time",   {'units': "ns"}),
            ("/" + segment + "/log",    log_attrs)]
def _check_energy(hdf5_file, segment, **kwargs):
    if not ([segment + "/time",
             segment + "/log"] in hdf5_file):
            return [(energy, segment, kwargs)]
    else:   return False
