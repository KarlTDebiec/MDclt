#!/usr/bin/python
desc = """mdtraj.py
    Functions for primary analysis of MD trajectories
    Written by Karl Debiec on 13-09-23
    Last updated 13-09-23"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
import mdtraj.topology
import mdtraj.trajectory
################################################## ANALYSIS FUNCTIONS ##################################################
def topology(segment, **kwargs):
    # Accepts pdb structure input file
    # Loads using mdtraj.topology
    # Returns results to hdf5 file for storage
    # Look into mdtraj.hdf5.HDF5TrajectoryFile for how to store in HDF5
    pass
def _check_topology(hdf5_file, segment, force = False, **kwargs):
    # Checks if topology needs to be run for this segment
    pass
def coordinates(segment, **kwargs):
    # Accepts xtc trajectory input file
    # Loads using mdtraj.trajectory
    # Returns results to hdf5 file for storage
    # Look into mdtraj.hdf5.HDF5TrajectoryFile for how to store in HDF5
    pass
def _check_coordinates(hdf5_file, segment, force = False, **kwargs):
    # Checks if coordinates needs to be run for this segment
    pass
