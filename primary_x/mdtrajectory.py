#!/usr/bin/python
desc = """mdtrajectory.py
    Functions for analysis using MDTraj
    Written by Marissa Pacey on 13-09-16
    Last updated on 13-10-10"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, json
import numpy as np
import mdtraj as md
from standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
def topology(infile, **kwargs):
    """ Parses a pdb <infile> using MDTraj, and stores the resulting topology as a json string.
        Adapted from 'topology.py' in MDTraj """
    topology        = md.load(infile).topology
    topology        = topology_to_json(topology)

    yield ("topology", {"json": str(topology_str)})

def _check_topology(hdf5_file, force = False, **kwargs):
    if    (force
    or not "topology" in hdf5_file):    return [(topology, kwargs)]
    else:                               return False


