#!/usr/bin/python
desc = """__init__.py
    Functions for primary analysis of MD trajectories
    Written by Karl Debiec on 13-10-30
    Last updated 13-10-30"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, types
import numpy as np
import mdtraj
import scipy.spatial.distance as ssd
################################################## ANALYSIS FUNCTIONS ##################################################
def com_resname(segment, destination, resname, **kwargs):
    trj         = mdtraj.load(segment.trajectory, top = segment.topology)
    resname_str = ""
    indexes     = []
    masses      = []
    total_mass  = []
    import time
    start   = time.time()
    for name in resname:
        for i, res in enumerate(trj.topology.residues, 1):
            if res.name == name:
                indexes        += [np.array([a.index         for a in res.atoms], np.int)]
                masses         += [np.array([a.element.mass  for a in res.atoms], np.float32)]
                total_mass     += [np.sum(masses[-1])]
                masses[-1]      = np.column_stack((masses[-1], masses[-1], masses[-1]))
                resname_str    += "{0} {1} ".format(res.name, i)
    total_mass  = np.array(total_mass)
    com         = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
    for i, frame in enumerate(trj.xyz):
        for j, index in enumerate(indexes):
            com[i][j]   = np.sum(trj.xyz[i][index] * masses[j], axis = 0) / total_mass[j]
    return  [(segment + "/" + destination, com * 10.0),
             (segment + "/" + destination, {"resname": resname_str[:-1], "method": "mdtraj", "units": "A"})]
def _check_com_resname(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    destination = kwargs.get("destination", "com")
    resname     = kwargs.get("resname",     "")
    if isinstance(resname, types.StringType):
        resname   = [resname]
    if    (force
    or not segment + "/" + destination in hdf5_file):
            return [(com_resname, segment, kwargs)]
    else:   return False
