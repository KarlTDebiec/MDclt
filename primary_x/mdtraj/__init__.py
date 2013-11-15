#!/usr/bin/python
desc = """MD_toolkit.primary_x.__init__.py
    Functions for primary analysis using MDTraj
    Written by Marissa Pacey on 13-09-16
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, json
import numpy as np
import mdtraj
from   MD_toolkit.standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
def topology(destination, infile, **kwargs):
    """ Parses a pdb <infile> using MDTraj, and stores the resulting topology as a json string. 
        Adapted from 'topology.py' in MDTraj """
    topology = mdtraj.load(infile).topology
    topology = topology_to_json(topology)

    yield (destination, topology, {"data_kwargs": {"chunks": False}})

def _check_topology(hdf5_file, force = False, **kwargs):
    kwargs["destination"]   = destination   = kwargs.get("destination", "topology")
    kwargs["infile"]        = infile        = kwargs.get("infile", [s for s in kwargs.get("segments", [])
                                              if  s.topology   and os.path.isfile(s.topology)][0].topology)
    if    (force
    or not "topology" in hdf5_file):    return [(topology, kwargs)]
    else:                               return False


