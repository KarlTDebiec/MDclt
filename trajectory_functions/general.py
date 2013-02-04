#!/usr/bin/python

desc = """general.py
    Functions for general analysis of trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-02-03"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, time
import numpy as np
import MDAnalysis as md
import MDAnalysis.analysis.align as mdaa
import scipy.spatial.distance as ssd
from   trajectory_cython import cy_contact, cy_distance_pbc
################################################## GENERAL FUNCTIONS ###################################################
def is_num(test):
    try:    float(test)
    except: return False
    return  True
def month(string):
    month = {'jan':  1, 'feb':  2, 'mar':  3, 'apr':  4, 'may':  5, 'jun':  6,
             'jul':  7, 'aug':  8, 'sep':  9, 'oct': 10, 'nov': 11, 'dec': 12}
    try:    return month[string.lower()]
    except: return None
################################################## ANALYSIS FUNCTIONS ##################################################
def com(arguments):
    """ Calculates center of mass of selected group """
    jobstep, name, group, topology, trajectory = arguments
    trj         = md.Universe(topology, trajectory)
    com         = np.zeros((len(trj.trajectory), 3), dtype = np.float)
    trj_sel     = trj.selectAtoms(group)
    n_atoms     = trj_sel.numberOfAtoms()
    for frame_i, frame in enumerate(trj.trajectory):
        com[frame_i]    = trj_sel.centerOfMass()
    return  [("/" + jobstep + "/com_" + name, com),
             ("/" + jobstep + "/com_" + name, {'group': group, 'units': "A"})]
def check_com(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    task_list   = []
    for name, group in arguments:
        if not jobstep + "/com_" + name in hierarchy:
            task_list  += [(com, (jobstep, name, group, topology, trajectory))]
    if task_list != []: return task_list
    else:               return False

def rmsd(arguments):
    """ Calculates rmsd of selected group against provided reference """
    jobstep, name, fit, reference, topology, trajectory = arguments
    ref         = md.Universe(reference)
    trj         = md.Universe(topology, trajectory)
    rmsd        = np.zeros((len(trj.trajectory)),       dtype = np.float)
    rotmat      = np.zeros((len(trj.trajectory), 9),    dtype = np.float)
    ref_sel     = ref.selectAtoms(fit)
    trj_sel     = trj.selectAtoms(fit)
    n_atoms     = trj_sel.numberOfAtoms()
    ref_frame   = (ref_sel.coordinates() - ref_sel.centerOfMass()).T.astype('float64')
    for frame_i, frame in enumerate(trj.trajectory):
        trj_frame       = (trj_sel.coordinates() - trj_sel.centerOfMass()).T.astype('float64')
        rmsd[frame_i]   = mdaa.qcp.CalcRMSDRotationalMatrix(ref_frame, trj_frame, n_atoms, rotmat[frame_i], None)
    return  [("/" + jobstep + "/rmsd_"   + name,    rmsd),
             ("/" + jobstep + "/rotmat_" + name,    rotmat),
             ("/" + jobstep + "/rmsd_"   + name,    {'fit': fit,    'units': "A"}),
             ("/" + jobstep + "/rotmat_" + name,    {'fit': fit})]
def check_rmsd(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    task_list   = []
    for name, fit, reference in arguments:
        if not (set([jobstep + "/rmsd_" + name, jobstep + "/rotmat_" + name]).issubset(hierarchy)):
            task_list  += [(rmsd, (jobstep, name, fit, reference, topology, trajectory))]
    if task_list != []: return task_list
    else:               return False

def contact(arguments):
    """ Calculates inter-residue contacts, defined as heavy-atom minimum distance within 5.5 Angstrom """
    jobstep, topology, trajectory = arguments
    trj         = md.Universe(topology, trajectory)
    trj_sel     = trj.selectAtoms("(protein or resname ACE) and (name C* or name N* or name O* or name S*)")
    n_res       = len(trj_sel.residues)
    atomcounts  = np.array([len(R.selectAtoms("(name C* or name N* or name O* or name S*)")) for R in trj_sel.residues])
    atomcounts  = np.array([(np.sum(atomcounts[:i]), np.sum(atomcounts[:i]) + n) for i, n in enumerate(atomcounts)])
    contacts    = np.zeros((len(trj.trajectory), (n_res**2-n_res)/2), dtype = np.int8)
    indexes     = _contact_2D_to_1D_indexes(n_res)
    for frame_i, frame in enumerate(trj.trajectory):
        contacts[frame_i]   = cy_contact(trj_sel.coordinates().T.astype('float64'), atomcounts, indexes)
    return  [("/" + jobstep + "/contact",  contacts)]
def check_contact(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    if not jobstep + "/contact" in hierarchy:   return [(contact, (jobstep, topology, trajectory))]
    else:                                       return False
def _contact_1D_to_2D_map(contact_1D):
    n_res       = int(1 + np.sqrt(1 + 8 * contact_1D.size)) / 2
    indexes     = _contact_1D_to_2D_indexes(n_res)
    contact_2D  = np.zeros((n_res, n_res), dtype = np.int8)
    contact_2D[indexes[:,0], indexes[:,1]]  = contact_1D
    contact_2D[indexes[:,1], indexes[:,0]]  = contact_1D
    contact_2D[range(n_res), range(n_res)]  = 1
    return contact_2D
def _contact_2D_to_1D_indexes(n_res):
    indexes = np.zeros((n_res,n_res), dtype = np.int)
    i       = 0
    for j in range(n_res-1):
        for k in range(j+1,n_res):
            indexes[j, k]   = i
            i              += 1
    return indexes
def _contact_1D_to_2D_indexes(n_res):
    indexes = np.zeros(((n_res**2-n_res)/2,2), dtype = np.int)
    i       = 0
    for j in range(n_res-1):
        for k in range(j+1,n_res):
            indexes[i]  = [j, k]
            i          += 1
    return indexes
    pass

#def mindist(arguments):
#    jobstep, topology, trajectory, name, group_1, group_2 = arguments
#    trj         = md.Universe(topology, trajectory)
#    group1      = trj.selectAtoms(group1)
#    group2      = trj.selectAtoms(group2)
#    distances   = np.zeros((len(trj.trajectory), len(group1), len(group2)))
#    for frame_i, frame in enumerate(trj.trajectory):
#        group1_crd          = np.array(group1.coordinates(), dtype = np.float64)
#        group2_crd          = np.array(group2.coordinates(), dtype = np.float64)
#        distances[frame_i]  = distance_pbc(group1_crd, group2_crd, float(frame.dimensions[0]))
#    mindists = np.min(np.min(distances, axis = 1), axis = 1)
#    return  [("/" + jobstep + "/min_dist" + name,   mindists),
#             ("/" + jobstep + "/min_dist" + name,   {'units': "A", 'group 1': group_1, 'group 2': group_2})]



#def salt_bridge(arguments):
#    jobstep, topology, trajectory, forced = arguments
#    trj         = md.Universe(topology, trajectory)
#    protein     = trj.selectAtoms("protein")
#    N_positive  = trj.selectAtoms("(resname ARG or resname HIS or resname LYS) and (type N) and not (name N)")
#    O_negative  = trj.selectAtoms("(resname ASP or resname GLU)                and (type O) and not (name O)")
#    try:    N_positive += trj.residues[0].selectAtoms("name N")
#    except: pass
#    try:    O_negative += trj.residues[-1].selectAtoms("type O")
#    except: pass
#    NO_dists    = np.zeros((len(trj.trajectory), len(N_positive), len(O_negative)))
#    for frame_i, frame in enumerate(trj.trajectory):
#        N_crd               = N_positive.coordinates() - protein.centerOfMass()
#        O_crd               = O_negative.coordinates() - protein.centerOfMass()
#        NO_dists[frame_i]   = ssd.cdist(N_crd, O_crd)
#    salt_bridges    = {}
#    for N_i, N in enumerate(N_positive):
#        for O_i, O in enumerate(O_negative):
#            dist    = NO_dists[:, N_i, O_i]
#            name    = "{}{}_{}{}".format(md.core.util.convert_aa_code(N.resname), N.resnum,
#                                         md.core.util.convert_aa_code(O.resname), O.resnum)
#            if   (name in salt_bridges):
#                prev_dist, atoms    = salt_bridges[name]
#                salt_bridges[name]  = (np.row_stack((np.array(prev_dist), dist)), atoms + [[N_i, O_i]])
#            elif (name in forced or np.min(dist) < 3.2):
#                salt_bridges[name] = (dist, [[N_i, O_i]])
#    final_array = []
#    for name, value in salt_bridges.iteritems():
#        distance, atoms = value
#        atoms       = np.array(atoms)
#        contact     = distance < 3.2
#        denticity   = np.zeros((dist.shape[-1]), dtype = np.int8)
#        for frame_i, frame in enumerate(np.transpose(contact)):
#            contact_atoms   = np.array([i for i, b in enumerate(list(frame)) if b])
#            if contact_atoms.size == 0: continue
#            contact_atoms   = atoms.take(contact_atoms, axis = 0)
#            if   (len(set(contact_atoms[:,0])) >= 2 and len(set(contact_atoms[:,1])) >= 2): denticity[frame_i] = 2
#            else:                                                                           denticity[frame_i] = 1
#        if (len(distance.shape) != 1):  distance    = np.min(distance, axis = 0)
#        final_array += [("/" + jobstep + "/salt_bridge/" + name + "/min_dist",   distance),
#                        ("/" + jobstep + "/salt_bridge/" + name + "/denticity",  denticity),
#                        ("/" + jobstep + "/salt_bridge/" + name + "/min_dist",   {'units': "A"})]
#    return final_array

