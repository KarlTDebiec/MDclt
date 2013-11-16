#!/usr/bin/python
desc = """MD_toolkit.primary.mdtraj.__init__.py
    Functions for primary analysis of MD trajectories
    Written by Karl Debiec on 13-10-30
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, types, warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mdtraj
from MD_toolkit.standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
def coordinates(segment, **kwargs):
    """ Loads coordinates in format compatible with mdtraj """
    trajectory          = mdtraj.load(segment.trajectory, top = segment.topology)
    cell_lengths        = trajectory.unitcell_lengths
    cell_lengths_attr   = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "degrees"}
    cell_angles         = trajectory.unitcell_angles
    cell_angles_attr    = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "nanometers"}
    coordinates         = trajectory.xyz
    coordinates_attr    = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "nanometers"}
    time                = trajectory.time
    time_attr           = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "picoseconds"}
    topology_json       = str(topology_to_json(trajectory.topology))
    topology            = np.array(topology_json, dtype="S{}".format(len(topology_json)))
    topology_attr       = {"CLASS" : "ARRAY", "VERSION": 2.3, "TITLE": "", "FLAVOR": "python"}

    return [(segment + "/cell_lengths", cell_lengths),
            (segment + "/cell_lengths", cell_lengths_attr),
            (segment + "/cell_angles",  cell_angles),
            (segment + "/cell_angles",  cell_angles_attr),
            (segment + "/coordinates",  coordinates),
            (segment + "/coordinates",  coordinates_attr),
            (segment + "/time",         time),
            (segment + "/time",         time_attr),
            (segment + "/topology",     topology, {"data_kwargs": {"chunks": False}}),
            (segment + "/topology",     topology_attr)]

def _check_coordinates(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    if    (force
    or not segment + "/coordinates" in hdf5_file):
            return [(coordinates, segment, kwargs)]
    else:   return False


def com_resname(segment, destination, resname, **kwargs):
    """ Calculates center of mass of each instance of <resname> """
    trj         = mdtraj.load(segment.trajectory, top = segment.topology)
    resname_str = ""
    indexes     = []
    masses      = []
    total_mass  = []
    
    for name in resname:
        for i, res in enumerate(trj.topology.residues, 1):
            if res.name == name:
                indexes        += [np.array([a.index        for a in res.atoms], np.int)]
                masses         += [np.array([a.element.mass for a in res.atoms], np.float32)]
                total_mass     += [np.sum(masses[-1])]
                masses[-1]      = np.column_stack((masses[-1], masses[-1], masses[-1]))
                resname_str    += "{0} {1} ".format(res.name, i)
    total_mass  = np.array(total_mass)
    com         = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
    mean        = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
    std         = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
    for i, frame in enumerate(trj.xyz):
        for j, index in enumerate(indexes):
            com[i][j]   = np.sum(trj.xyz[i][index] * masses[j], axis = 0) / total_mass[j]
    return  [(segment + "/" + destination, com * 10.0),
             (segment + "/" + destination, {"resname": resname_str[:-1], "method": "mdtraj", "units": "A"})]
def _check_com_resname(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    kwargs["destination"] = destination = kwargs.get("destination", "com")
    kwargs["resname"]     = resname     = kwargs.get("resname",     ["HOH"])
    if isinstance(resname, types.StringType):
        kwargs["resname"] = [resname]
    if    (force
    or not segment + "/" + destination in hdf5_file):
            return [(com_resname, segment, kwargs)]
    else:   return False


