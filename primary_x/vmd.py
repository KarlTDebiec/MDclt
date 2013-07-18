#!/usr/bin/python
desc = """vmd.py
    Functions for primary cross-segment analysis of MD trajectories using Visual Molecular Dynamics
    Written by Karl Debiec on 13-07-18
    Last updated 13-07-18"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, types
import numpy as np
from   standard_functions import shell_iterator
################################################## ANALYSIS FUNCTIONS ##################################################
def com(segments, vmd = "vmd", selection = ["protein and name CA"], destination = "", **kwargs):
    """ Calculates com of <selection> accounting for periodic boundary conditions using <vmd> """
    script          = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["tcl", "vmd_com_x.tcl"])
    if isinstance(selection, types.StringType):
        selection   = [selection]
    n_selections    = len(selection)
    if not destination.startswith("_"):
        destination = "_" + destination

    topology_string         = segments[0].topology
    trajectory_string       = " ".join(["\"{0}\"".format(segment.trajectory)    for segment in segments])
    selection_string        = " ".join(["\"{0}\"".format(sel.replace(" ", "_")) for sel     in selection])
    selection_string_attr   = " ".join(["\"{0}\"".format(sel)                   for sel     in selection])
    command                 = "{0} -dispdev text -e {1} -args -topology {2} -trajectory {3} -selection {4}".format(
                              vmd, script, topology_string, trajectory_string, selection_string)
    com                     = None
    attr                    = {"selection": selection_string_attr, "method": "vmd", "units": "A"}
    for line in shell_iterator(command):

        if line.startswith("SEGMENT"):
            if com is not None:
                yield (segments[int(segment_i)] + "/com" + destination, com)
                yield (segments[int(segment_i)] + "/com" + destination, attr)
            _, segment_i, _, n_frames   = line.split()
            com                         = np.zeros((int(n_frames), n_selections, 3), dtype = np.float32)
        elif line.startswith("FRAME"):
            _, i, _, j, _, x, _, y, _, z    = line.split()
            com[int(i), int(j)] = float(x), float(y), float(z)
    yield (segments[int(segment_i)] + "/com" + destination, com)
    yield (segments[int(segment_i)] + "/com" + destination, attr)
    
def _check_com(hdf5_file, segments, force = False, **kwargs):
    destination     = kwargs.get("destination", "")
    if not destination.startswith("_"):
        destination = "_" + destination
    expected        = [segment + "/com" + destination for segment in segments]
    if (force
    or  not expected in hdf5_file): return [(com, kwargs)]
    else:                           return False
