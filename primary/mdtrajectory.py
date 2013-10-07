#!/usr/bin/python
desc = """mdtrajectory.py
    Functions for primary analysis of MD trajectories
    Written by Karl Debiec on 13-09-23
    Last updated 13-09-23"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
# import mdtraj.topology
# import mdtraj.trajectory
import mdtraj
import json
################################################## ANALYSIS FUNCTIONS ##################################################
def topology(segment, **kwargs):
    # Accepts pdb structure input file
    # Loads using mdtraj.topology
    # Returns results to hdf5 file for storage
    # Look into mdtraj.hdf5.HDF5TrajectoryFile for how to store in HDF5
    pdb = kwargs["topology"]
    
    topology        = mdtraj.load(pdb).topology
    topology_dict   = {"chains":  [],
                       "bonds":   []} 

    chain_iter      = topology.chains
    if not hasattr(chain_iter, "__iter__"):
        chain_iter  = chain_iter()
    for chain in chain_iter:
        chain_dict  = {
          "residues":  [], 
          "index":     int(chain.index)}

        residue_iter        = chain.residues
        if not hasattr(residue_iter, "__iter__"):
            residue_iter    = residue_iter()
        for residue in residue_iter:
            residue_dict    = {
              "index": int(residue.index),
              "name":  str(residue.name),
              "atoms": []} 

            atom_iter       = residue.atoms
            if not hasattr(atom_iter, "__iter__"):
                atom_iter   = atom_iter()
            for atom in atom_iter:
                residue_dict["atoms"].append(
                  {"index":   int(atom.index),
                   "name":    str(atom.name),
                   "element": str(atom.element.symbol)})
            chain_dict['residues'].append(residue_dict)
        topology_dict['chains'].append(chain_dict)

    bond_iter       = topology.bonds
    if not hasattr(bond_iter, "__iter__"):
        bond_iter   = bond_iter()
    for atom1, atom2 in bond_iter:
        topology_dict['bonds'].append([int(atom1.index),
                                       int(atom2.index)])
    data = [str(json.dumps(topology_dict))]
    return [("topology", data)]

def _check_topology(hdf5_file, segment, force = False, **kwargs):
    # # Checks if topology needs to be run for this segment
    # if not (segment.topology   and os.path.isfile(segment.topology)):
    #         return False
    if    (force
    or not "/topology" in hdf5_file):
            return [(topology, segment, kwargs)]

def coordinates(segment, **kwargs):
    # Accepts xtc trajectory input file
    # Loads using mdtraj.trajectory
    # Returns results to hdf5 file for storage
    # Look into mdtraj.hdf5.HDF5TrajectoryFile for how to store in HDF5
    trajectory = mdtraj.load_xtc(segment.trajectory, top=segment.topology)
    # mdtraj writes trajectory to HDF5 like this:
    # f.write(coordinates=self.xyz, time=self.time,
                    # cell_angles=self.unitcell_angles,
                    # cell_lengths=self.unitcell_lengths)
    # interested in xyz-np.ndarray, shape=(n_frames, n_atoms, 3)
    xyz = trajectory.xyz
    return [(segment+ "/coordinates", xyz)]

def _check_coordinates(hdf5_file, segment, force = False, **kwargs):
    # Checks if coordinates needs to be run for this segment
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    if    (force
    or not segment + "/coordinates" in hdf5_file):
            return [(coordinates, segment, kwargs)]
    else:   return False