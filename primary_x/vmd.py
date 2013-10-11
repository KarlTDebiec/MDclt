#!/usr/bin/python
desc = """vmd.py
    Functions for primary cross-segment analysis of MD trajectories using Visual Molecular Dynamics
    Written by Karl Debiec on 13-07-18
    Last updated 13-07-22"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys, types
import numpy as np
from   standard_functions import shell_iterator
################################################## ANALYSIS FUNCTIONS ##################################################
def com(segments, vmd = "vmd", selection = ["protein and name CA"], destination = "", **kwargs):
    """ Calculates com of <selection>(s) accounting for periodic boundary conditions using <vmd> """
    if commands.getstatusoutput("hash vmd")[0] != 0:
        raise Exception("VMD command '{0}' is not available; check your $PATH and modules".format(vmd))
    script          = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["tcl", "vmd_com_x.tcl"])
    if isinstance(selection, types.StringType):
        selection   = [selection]
    n_selections    = len(selection)
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination

    topology_string         = segments[0].topology
    trajectory_string       = " ".join(["\"{0}\"".format(s.trajectory)        for s in segments if s.trajectory])
    selection_string        = " ".join(["\"{0}\"".format(s.replace(" ", "_")) for s in selection])
    selection_string_attr   = " ".join(["\"{0}\"".format(s)                   for s in selection])
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
    
def _check_com(hdf5_file, force = False, **kwargs):
    segments        = kwargs.get("segments",    [])
    destination     = kwargs.get("destination", "")
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination
    expected        = [s + "/com" + destination for s in segments if  s.topology   and os.path.isfile(s.topology)
                                                                  and s.trajectory and os.path.isfile(s.trajectory)]
    if (force
    or  not expected in hdf5_file): return [(com, kwargs)]
    else:                           return False


