#!/usr/bin/python
desc = """mdtrajectory.py
    Functions for primary analysis of MD trajectories
    Written by Marissa Pacey on 13-09-23
    Last updated 13-10-10"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, json
import numpy as np
import mdtraj
from standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
def coordinates(segment, **kwargs):
    trajectory  = mdtraj.load(segment.trajectory, top = segment.topology)

    cell_lengths = trajectory.unitcell_lengths
    cell_lengths_attr = {"CLASS" : "EARRAY",
                         "VERSION": 1.0,
                         "TITLE": "",
                         "EXTDIM": 0,
                         "units" : "degrees",
                        }

    cell_angles = trajectory.unitcell_angles
    cell_angles_attr = {"CLASS" : "EARRAY",
                        "VERSION": 1.0,
                        "TITLE": "",
                        "EXTDIM": 0,
                        "units": "nanometers"}

    coordinates = trajectory.xyz
    coordinates_attr = {"CLASS" : "EARRAY",
                        "VERSION": 1.0,
                        "TITLE": "",
                        "EXTDIM": 0,
                        "units" : "nanometers"}

    time = trajectory.time
    time_attr = {"CLASS" : "EARRAY",
                 "VERSION": 1.0,
                 "TITLE": "",
                 "EXTDIM": 0,
                 "units": "picoseconds"}

    topology_json = str(topology_to_json(trajectory.topology))
    topology = np.array(topology_json, dtype="S{}".format(len(topology_json)))
    topology_attr = {"CLASS" : "ARRAY",
                     "VERSION": 2.3,
                     "TITLE": "",
                     "FLAVOR": "python"}

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
