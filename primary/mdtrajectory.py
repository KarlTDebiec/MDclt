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
    cell_lengths_attr = {"units" : "degrees"}

    cell_angles = trajectory.unitcell_angles
    cell_angles_attr = {"units": "nanometers"}

    coordinates = trajectory.xyz
    coordinates_attr = {"units" : "nanometers"}

    time = trajectory.time
    time_attr = {"units": "picoseconds"}

    topology = topology_to_json(trajectory.topology)
    topology_attr = {"json": topology}

    return [(segment + "/cell_lengths", cell_lengths),
            (segment + "/cell_lengths", cell_lengths_attr),
            (segment + "/cell_angles",  cell_angles),
            (segment + "/cell_angles",  cell_angles_attr),
            (segment + "/coordinates",  coordinates),
            (segment + "/coordinates",  coordinates_attr),
            (segment + "/time",         time),
            (segment + "/time",         time_attr),
            # (segment + "/topology",     topology),
            (segment + "/topology",     topology_attr)
            ]

def _check_coordinates(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    if    (force
    or not segment + "/coordinates" in hdf5_file):
            return [(coordinates, segment, kwargs)]
    else:   return False
