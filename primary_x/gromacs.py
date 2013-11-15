#!/usr/bin/python
desc = """MD_toolkit.primary_x.gromacs.py
    Functions for analysis of Gromacs Simulations
    Written by Marissa Pacey on 13-09-16
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys, re
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def config(infile, **kwargs):
    """ Parses configuration (mdp) <infile> """
    attr            = {"file" : infile}
    configuration   = []
    with open(infile, "r") as config_file:
        for line in config_file:
            line_tuple  = tuple([s.strip() for s in re.split(r"[=;]", line)])
            if not line.startswith(";"): configuration.append(line_tuple)
    configuration   = np.array(configuration, dtype=np.dtype([("field","S25"),("value","S25"),("comment","S100")]))
    yield ("config", configuration)
    yield ("config", attr)
def _check_config(hdf5_file, force = False, **kwargs):
    if (force
    or  not "config" in hdf5_file): return [(config, kwargs)]
    else:                           return False

