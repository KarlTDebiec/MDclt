#!/usr/bin/python

desc = """analogue.py
    Functions for analysis of amino acid analogue trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-03-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
import MDAnalysis as md
from   cython_functions import _cy_distance_pbc
################################################## ANALYSIS FUNCTIONS ##################################################
def com(arguments):
    """ Calculates center of mass distance between two residue types; assumes cubic box and pbc """
    alg1        = kwargs.get("alg1")
    alg2        = kwargs.get("alg1") 
    trj         = md.Universe(segment.topology, segment.trajectory)
    alg1s       = [r.atoms for r in trj.residues if r.name == alg1]
    alg2s       = [r.atoms for r in trj.residues if r.name == alg2]
    alg1s_com   = np.zeros((len(alg1s), 3))
    alg2s_com   = np.zeros((len(alg2s), 3))
    distances   = np.zeros((len(trj.trajectory), len(alg1s), len(alg2s)))
    for i, frame in enumerate(trj.trajectory):
        for j, alg1 in enumerate(alg1s):  alg1s_com[j] = np.array(alg1.centerOfMass(), dtype = np.float64)
        for j, alg2 in enumerate(alg2s):  alg2s_com[j] = np.array(alg2.centerOfMass(), dtype = np.float64)
        distances[i]  = _cy_distance_pbc(alg1s_com, alg2s_com, float(frame.dimensions[0]))
    return  [("/" + segment + "/association_com",   distances),
             ("/" + segment + "/association_com",   {'units': "A"})]
def _check_com(hdf5_file, segment, **kwargs):
    if not (segment + "/association_com" in hdf5_file):
            return [(com, segment, kwargs)]
    else:   return False

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


