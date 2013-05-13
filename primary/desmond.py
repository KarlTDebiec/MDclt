#!/usr/bin/python
desc = """desmond.py
    Functions for primary analysis of Desmond trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-05-12"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import month
################################################## ANALYSIS FUNCTIONS ##################################################
def log(segment, **kwargs):
    """ parses log for <segment> """
    log         = segment.file_of_type(".ene")
    head        = commands.getoutput("head -n 3 {0}".format(log)).split("\n")[2].split()
    date        = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
    start_time  = head[7]
    dtype       = np.dtype([("time",      "f"), ("total",       "f"),  ("potential",        "f"), ("kinetic",  "f"),
                            ("conserved", "f"), ("exchange",    "f"),  ("force correction", "f"), ("pressure", "f"),
                            ("volume",    "f"), ("temperature", "f")])
    log         = np.loadtxt(log, dtype, usecols = (0,1,2,3,4,5,6,7,8,9))
    log["time"]/= 1000
    log_attrs   = {"time ":            "ns",         "total":     "kcal mol-1", "potential": "kcal mol-1",
                   "kinetic":          "kcal mol-1", "conserved": "kcal mol-1", "exchange":  "kcal mol-1",
                   "force correction": "kcal mol-1", "pressure":  "bar",        "volume":    "A3",
                   "temperature":      "K"}
    return [(segment + "/log",    log[1:]),
            (segment + "/",       {"date":  date,  "time": start_time}),
            (segment + "/log",    log_attrs)]
def _check_log(hdf5_file, segment, **kwargs):
    if not (segment + "/log" in hdf5_file): return [(log, segment, kwargs)]
    else:                                   return False
