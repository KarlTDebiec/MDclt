########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
import mdtraj as md
import json
################################################## ANALYSIS FUNCTIONS ##################################################
def topology(segment, **kwargs):
    # Accepts pdb structure input file
    # Loads using mdtraj.topology
    # Returns results to hdf5 file for storage
    # Look into mdtraj.hdf5.HDF5TrajectoryFile for how to store in HDF5
    pdb             = kwargs["topology"]
    topology        = md.load(pdb).topology
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
    yield ("topology", data)

def _check_topology(hdf5_file, segment, force = False, **kwargs):
    if    (force
    or not "/topology" in hdf5_file):
            return [(topology, kwargs)]
    else:   return False
