#!/usr/bin/python
desc = """MD_toolkit.primary.mdanalysis.association.py
    Functions for primary analysis of molecular association using MDAnalysis
    Written by Karl Debiec on 12-11-30
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
import MDAnalysis as md
from   cython_functions import _cy_distance_pbc
################################################## ANALYSIS FUNCTIONS ##################################################
def comdist(segment, selection_1, selection_2, mode = "residue", **kwargs):
    """ Calculates center of mass distance between two selections or residue types; assumes cubic box and pbc """
    trj         = md.Universe(segment.topology, segment.trajectory)
    if   mode  == "residue":
        mol1s   = [r.atoms for r in trj.residues if r.name == selection_1]
        mol2s   = [r.atoms for r in trj.residues if r.name == selection_2]
    elif mode  == "selection":
        mol1s   = [trj.selectAtoms(selection_1)]
        mol2s   = [trj.selectAtoms(selection_2)]
    mol1s_com   = np.zeros((len(mol1s), 3), np.float32)
    mol2s_com   = np.zeros((len(mol2s), 3), np.float32)
    comdist     = np.zeros((len(trj.trajectory), len(mol1s), len(mol2s)), np.float32)
    for i, frame in enumerate(trj.trajectory):
        for j, mol1 in enumerate(mol1s):  mol1s_com[j] = mol1.centerOfMass()
        for j, mol2 in enumerate(mol2s):  mol2s_com[j] = mol2.centerOfMass()
        comdist[i]  = _cy_distance_pbc(mol1s_com, mol2s_com, float(frame.dimensions[0]))
    return  [(segment + "/association_comdist", comdist),
             (segment + "/association_comdist", {"units": "A"})]
def _check_comdist(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    if (force
    or  not (segment + "/association_comdist" in hdf5_file)):
            return [(comdist, segment, kwargs)]
    else:   return False

def mindist(segment, selection_1, selection_2, mode = "residue", destination = "association_mindist", **kwargs):
    """ Calculates minimum distance between two selections or two residue types; assumes cubic box and pbc """
    trj         = md.Universe(segment.topology, segment.trajectory)
    if   mode  == "residue":
        res1, sel1  = selection_1
        res2, sel2  = selection_2
        mol1s   = [trj.selectAtoms("({0}) and (resnum {1}) and (resname {2})".format(sel1, r.resnum, res1))
                     for r in trj.residues if r.name == res1]
        mol2s   = [trj.selectAtoms("({0}) and (resnum {1}) and (resname {2})".format(sel2, r.resnum, res2))
                     for r in trj.residues if r.name == res2]
    elif mode  == "selection":
        mol1s   = [trj.selectAtoms(selection_1)]
        mol2s   = [trj.selectAtoms(selection_2)]
    distance    = np.zeros((len(mol1s[0].atoms), len(mol2s[0].atoms)),    np.float32)
    mindist     = np.zeros((len(trj.trajectory), len(mol1s), len(mol2s)), np.float32)
    for i, frame in enumerate(trj.trajectory):
        for j, mol1 in enumerate(mol1s):
            for k, mol2 in enumerate(mol2s):
                distance         = _cy_distance_pbc(mol1.coordinates(), mol2.coordinates(), float(frame.dimensions[0]))
                mindist[i, j, k] = np.min(distance)
    return  [(segment + "/" + destination, mindist),
             (segment + "/" + destination, {"units": "A"})]
def _check_mindist(hdf5_file, segment, force = False, **kwargs):
    destination = kwargs.get("destination", "association_mindist")
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    if (force
    or  not (segment + "/" + destination in hdf5_file)):
            return [(mindist, segment, kwargs)]
    else:   return False

##################################################### DEPRECIATED ######################################################
#def salt_bridge(arguments):
#    """ salt bridge analysis, assumes cubic box and pbc """
#    segment, topology, trajectory = arguments
#    trj             = md.Universe(topology, trajectory)
#    O_negative      = trj.selectAtoms("(resname ACT and (type O))")
#    N_positive      = trj.selectAtoms("(resname GND and (type N))")
#    NO_dists        = np.zeros((len(trj.trajectory), len(N_positive), len(O_negative)))
#    for frame_i, frame in enumerate(trj.trajectory):
#        N_crd               = np.array(N_positive.coordinates(), dtype = np.float64)
#        O_crd               = np.array(O_negative.coordinates(), dtype = np.float64)
#        NO_dists[frame_i]   = cy_distance_pbc(N_crd, O_crd, float(frame.dimensions[0]))
#    all_mindist     = []
#    all_denticity   = []
#    for gnd_i in range(50):
#        sub_NO_dists    = NO_dists[:, 3*gnd_i:3*gnd_i+3, :]
#        mindist         = np.min(np.min(sub_NO_dists, axis = 1), axis = 1)
#        contacts        = sub_NO_dists < 3.2
#        unique_Ns       = np.sum(contacts.any(axis = 1), axis = 1)
#        unique_Os       = np.sum(contacts.any(axis = 2), axis = 1)
#        denticity       = np.zeros(sub_NO_dists.shape[0], dtype = np.int8)
#        denticity[unique_Ns >= 1]                      += 1
#        denticity[(unique_Ns >= 2) & (unique_Os >= 2)] += 1
#        all_mindist    += [mindist]
#        all_denticity  += [denticity]
#    all_mindist     = np.reshape(np.concatenate(all_mindist),   (50, frame_i + 1))
#    all_denticity   = np.reshape(np.concatenate(all_denticity), (50, frame_i + 1))
#    return [("/" + segment + "/salt_bridge/GND_ACT/min_dist",  all_mindist),
#            ("/" + segment + "/salt_bridge/GND_ACT/denticity", all_denticity),
#            ("/" + segment + "/salt_bridge/GND_ACT/min_dist",  {'units': "A"})]
#def check_salt_bridge(hierarchy, segment, arguments):
#    segment, path, topology, trajectory = segment
#    if not segment + "/salt_bridge" in hierarchy:   return [(salt_bridge, (segment, topology, trajectory))]
#    else:                                           return False
