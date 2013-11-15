#!/usr/bin/python
desc = """MD_toolkit.MDAT_Trajectory.py
    Subclass of MDTraj.trajectory.Trajectory with added ability to load from our HDF5 File format
    Written by Marissa pacey on 13-10-20
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   mdtraj import Topology
from   mdtraj.trajectory import load, Trajectory
from   MD_toolkit.HDF5_File import HDF5_File
from   MD_toolkit.standard_functions import topology_from_json
######################################################## CLASS #########################################################
class MDAT_Trajectory(Trajectory):
    def __init__(self, filenames = [], source = "", **kwargs):
       self.load(filenames, source, **kwargs)
    def load(self, filenames, source = "", **kwargs):
        if isinstance(filenames, HDF5_File):
            hdf5_file       = filenames
            coordinates     = hdf5_file.load(source + "/coordinates")
            topology_json   = hdf5_file.load(source + "/topology")
            topology        = topology_from_json(topology_json.tostring())
            time            = hdf5_file.load(source + "/time")
            cell_lengths    = hdf5_file.load(source + "/cell_lengths")
            cell_angles     = hdf5_file.load(source + "/cell_angles")
            Trajectory.__init__(self, xyz=coordinates, topology=topology,
                                         time=time, unitcell_lengths=cell_lengths,
                                         unitcell_angles=cell_angles)
        else:
            load(filenames, **kwargs)
