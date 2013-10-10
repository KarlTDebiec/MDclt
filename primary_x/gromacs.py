#!/usr/bin/python
desc = """gromacs.py
    Functions for analysis of Gromacs Simulations
    Written by Marissa Pacey on 13-09-16
    Last updated 13-09-16"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys, re
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def config(segments, **kwargs):
    """ Parses configuration <infile> """
    # One of 'kwargs' should be the input file name
    # Open the infile, loop over lines, for each field add to a list, convert list to array
    # It should make a numpy table whose data type is (field, value, comment)
    # It should return this table
    # It should also return the input filename as an attribute
    config = kwargs["infile"]
    attr = {"file" : config}
    configuration = []
    with open(config, "r") as config_file:
        for line in config_file:
           line_tuple = tuple([s.strip() for s in re.split(r"[=;]", line)])
           if not line.startswith(";"): configuration.append(line_tuple)
    dtype = [("field", "S25"), ("value", "S25"), ("comment", "S100")]
    configuration = np.array(configuration, dtype=dtype)
    yield ("config", configuration)
    yield ("config", attr)
def _check_config(hdf5_file, segments, force = False, **kwargs):
    if (force
    or  not "config" in hdf5_file): return [(config, kwargs)]
    else:                           return False

