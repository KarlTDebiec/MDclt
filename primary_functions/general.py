#!/usr/bin/python

desc = """general.py
    Functions for primary analysis of molecular dynamics trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-02-08"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
import MDAnalysis as md
import MDAnalysis.analysis.align as mdaa
import scipy.spatial.distance as ssd
import trajectory_cython
from   cython_functions  import _cy_contact
from   standard_functions import is_num, _contact_2D_to_1D_indexes
################################################## ANALYSIS FUNCTIONS ##################################################
def com(segment, **kwargs):
    """ Calculates center of mass of <name> with selection <group> from <topology> and <trajectory> for <segment> """
    domain      = kwargs.get("domain",      "")
    selection   = kwargs.get("selection",   "protein")
    trj         = md.Universe(segment.topology, segment.trajectory)
    com         = np.zeros((len(trj.trajectory), 3))
    trj_sel     = trj.selectAtoms(selection)
    n_atoms     = trj_sel.numberOfAtoms()
    for i, frame in enumerate(trj.trajectory):
        com[i]  = trj_sel.centerOfMass()
    return  [("/" + segment + "/com_" + domain, com),
             ("/" + segment + "/com_" + domain, {'selection': selection, 'units': "A"})]
def _check_com(hdf5_file, segment, **kwargs):
    domain  = kwargs.get("domain",    "")
    if not (segment + "/com_" + domain in hdf5_file):
            return [(com, segment, kwargs)]
    else:   return False


def rmsd(segment, **kwargs):
    """ Calculates rmsd of <domain> relative to <reference> with <selection> """
    domain      = kwargs.get("domain",    "")
    selection   = kwargs.get("selection", "protein and name CA")
    reference   = kwargs.get("reference")
    ref         = md.Universe(reference)
    trj         = md.Universe(segment.topology, segment.trajectory)
    rmsd        = np.zeros(len(trj.trajectory))
    rotmat      = np.zeros((len(trj.trajectory), 9))
    ref_sel     = ref.selectAtoms(selection)
    trj_sel     = trj.selectAtoms(selection)
    n_atoms     = trj_sel.numberOfAtoms()
    ref_frame   = (ref_sel.coordinates() - ref_sel.centerOfMass()).T.astype('float64')
    for i, frame in enumerate(trj.trajectory):
        trj_frame   = (trj_sel.coordinates() - trj_sel.centerOfMass()).T.astype('float64')
        rmsd[i]     = mdaa.qcp.CalcRMSDRotationalMatrix(ref_frame, trj_frame, n_atoms, rotmat[i], None)
    return  [("/" + segment + "/rmsd_"   + domain,  rmsd),
             ("/" + segment + "/rotmat_" + domain,  rotmat),
             ("/" + segment + "/rmsd_"   + domain,  {'selection': selection,    'units': "A"}),
             ("/" + segment + "/rotmat_" + domain,  {'selection': selection})]
def _check_rmsd(hdf5_file, segment, **kwargs):
    domain  = kwargs.get("domain",    "")
    if not ([segment + "/rmsd_"   + domain,
             segment + "/rotmat_" + domain] in hdf5_file):
            return [(rmsd, segment, kwargs)]
    else:   return False


def contact(segment, **kwargs):
    """ Calculates inter-residue contacts from <topology> and <trajectory> for <segment>
        Contact defined as heavy-atom minimum distance within 5.5 Angstrom """
    res_selection   = kwargs.get("res_selection",  "(protein or resname ACE)")
    atom_selection  = kwargs.get("atom_selection", "(name C* or name N* or name O* or name S*)")
    trj             = md.Universe(segment.topology, segment.trajectory)
    trj_sel         = trj.selectAtoms(res_selection + " and " + atom_selection)
    n_res           = len(trj_sel.residues)
    atomcounts      = np.array([len(R.selectAtoms(atom_selection)) for R in trj_sel.residues])
    atomcounts      = np.array([(np.sum(atomcounts[:i]), np.sum(atomcounts[:i]) + n) for i, n in enumerate(atomcounts)])
    contacts        = np.zeros((len(trj.trajectory), (n_res ** 2 - n_res) / 2), dtype = np.int8)
    indexes         = _contact_2D_to_1D_indexes(n_res)
    for frame_i, frame in enumerate(trj.trajectory):
        contacts[frame_i]   = _cy_contact(trj_sel.coordinates().T.astype('float64'), atomcounts, indexes)
    return  [("/" + segment + "/contact",  contacts)]
def _check_contact(hdf5_file, segment, **kwargs):
    if not (segment + "/contact" in hdf5_file):
            return [(contact, segment, kwargs)]
    else:   return False

#def mindist(arguments):
#    segment, topology, trajectory, name, group_1, group_2 = arguments
#    trj         = md.Universe(topology, trajectory)
#    group1      = trj.selectAtoms(group1)
#    group2      = trj.selectAtoms(group2)
#    distances   = np.zeros((len(trj.trajectory), len(group1), len(group2)))
#    for frame_i, frame in enumerate(trj.trajectory):
#        group1_crd          = np.array(group1.coordinates(), dtype = np.float64)
#        group2_crd          = np.array(group2.coordinates(), dtype = np.float64)
#        distances[frame_i]  = distance_pbc(group1_crd, group2_crd, float(frame.dimensions[0]))
#    mindists = np.min(np.min(distances, axis = 1), axis = 1)
#    return  [("/" + segment + "/min_dist" + name,   mindists),
#             ("/" + segment + "/min_dist" + name,   {'units': "A", 'group 1': group_1, 'group 2': group_2})]



#def salt_bridge(arguments):
#    segment, topology, trajectory, forced = arguments
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
#        final_array += [("/" + segment + "/salt_bridge/" + name + "/min_dist",   distance),
#                        ("/" + segment + "/salt_bridge/" + name + "/denticity",  denticity),
#                        ("/" + segment + "/salt_bridge/" + name + "/min_dist",   {'units': "A"})]
#    return final_array

