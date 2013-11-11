#!/usr/bin/python
desc = """MDAT_Trajectory.py
    Subclass of MDTraj.trajectory.Trajectory with added ability to load from our HDF5 File format
    Written by Karl Debiec on 13-10-20
    Last updated 13-10-20"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from mdtraj.trajectory import load, Trajectory
from standard_functions import topology_from_json
######################################################## CLASS #########################################################
class MDAT_Trajectory(Trajectory):
    @staticmethod
    def load(filenames, mdat_format = False, source = None, **kwargs):
        if mdat_format:
            # Load trajectory from our HDF5 format
            # Filename is like ", source is like "0000"
            hdf5_file       = filenames # Only accept one filename for now, is not really a filename but and HDF5_File object
            coordinates     = hdf5_file.load(source + "/coordinates")
            topology_json  = hdf5_file.load(source + "/topology")
            topology        = topology_from_json(topology_json.tostring())
            time            = hdf5_file.load(source + "/time")
            cell_lengths    = hdf5_file.load(source + "/cell_lengths")
            cell_angles     = hdf5_file.load(source + "/cell_angles")
            trajectory = Trajectory(xyz=coordinates, topology=topology,
                                    time=time, unitcell_lengths=cell_lengths,
                                    unitcell_angles=cell_angles)
            return trajectory
        else:
            # Otherwise, just use standard loader
            load(filenames, **kwargs)
