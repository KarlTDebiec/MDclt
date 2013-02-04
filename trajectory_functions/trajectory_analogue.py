#!/usr/bin/python

desc = """analyze_analogue.py
    Functions for analysis of amino acid analogue simulations
    Written by Karl Debiec on 12-11-30
    Last updated 13-01-25"""

########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
import MDAnalysis as md
sys.path += [os.getenv('HOME') + "/Dropbox/chong_group/12-12-09_analysis/cython"]
sys.path += ["scripts/cython"]
from analysis_cython import cy_distance_pbc
################################################## GENERAL FUNCTIONS ###################################################
def is_num(test):
    try:    float(test)
    except: return False
    return  True
################################################## ANALYSIS FUNCTIONS ##################################################
def com(arguments):
    """ center of mass distance, assumes cubic box and pbc """
    jobstep, topology, trajectory, alg1, alg2 = arguments
    trj         = md.Universe(topology, trajectory)
    alg1s       = [r.atoms for r in trj.residues if r.name == alg1]
    alg2s       = [r.atoms for r in trj.residues if r.name == alg2]
    alg1s_com   = np.zeros((len(alg1s), 3))
    alg2s_com   = np.zeros((len(alg2s), 3))
    distances   = np.zeros((len(trj.trajectory), len(alg1s), len(alg2s)))
    for frame_i, frame in enumerate(trj.trajectory):
        for alg1_i, alg1 in enumerate(alg1s):  alg1s_com[alg1_i] = np.array(alg1.centerOfMass(), dtype = np.float64)
        for alg2_i, alg2 in enumerate(alg2s):  alg2s_com[alg2_i] = np.array(alg2.centerOfMass(), dtype = np.float64)
        distances[frame_i]  = cy_distance_pbc(alg1s_com, alg2s_com, float(frame.dimensions[0]))
    return  [("/" + jobstep + "/association_com",   distances),
             ("/" + jobstep + "/association_com",   {'units': "A"})]
def check_com(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    alg1, alg2                          = arguments
    if not os.path.isfile(topology):    return False
    if not os.path.isfile(trajectory):  return False
    if not jobstep + "/association_com" in hierarchy:   return [(com, (jobstep, topology, trajectory, alg1, alg2))]
    else:                                               return False

#def salt_bridge(arguments):
#    """ salt bridge analysis, assumes cubic box and pbc """
#    jobstep, topology, trajectory = arguments
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
#    return [("/" + jobstep + "/salt_bridge/GND_ACT/min_dist",  all_mindist),
#            ("/" + jobstep + "/salt_bridge/GND_ACT/denticity", all_denticity),
#            ("/" + jobstep + "/salt_bridge/GND_ACT/min_dist",  {'units': "A"})]
#def check_salt_bridge(hierarchy, jobstep, arguments):
#    jobstep, path, topology, trajectory = jobstep
#    if not jobstep + "/salt_bridge" in hierarchy:   return [(salt_bridge, (jobstep, topology, trajectory))]
#    else:                                           return False


