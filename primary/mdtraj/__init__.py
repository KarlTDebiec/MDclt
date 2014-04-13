#!/usr/bin/python
desc = """MD_toolkit.primary.mdtraj.__init__.py
    Functions for primary analysis of MD trajectories
    Written by Karl Debiec on 13-10-30
    Last updated by Karl Debiec on 14-03-17"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys, types, warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mdtraj
from MD_toolkit.standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
def dipole(segment, destination, solvent, verbose = True, **kwargs):
    """ Calculates dipole moment of solvent molecules in the system. Uses partial charges for <solvent> and calculates
        and includes pseudoatoms if applicable. """

    # Prepare charges for selected solute
    if   solvent.lower() in ["spce"]:
        O_chg, H_chg, M_chg, M_d, resnames = -0.83400, 0.41700, None,     None,   ["HOH", "SPC"]
    elif solvent.lower() in ["tip3p", "tips3p"]:
        O_chg, H_chg, M_chg, M_d, resnames = -0.84760, 0.42380, None,     None,   ["HOH", "T3P"]
    elif solvent.lower() in ["tip4p"]:
        O_chg, H_chg, M_chg, M_d, resnames =  0.0,     0.52000, -1.04000, 0.15,   ["HOH", "T4P"]
    elif solvent.lower() in ["tip4p2005"]:
        O_chg, H_chg, M_chg, M_d, resnames =  0.0,     0.55640, -1.11280, 0.125,  ["HOH", "T4P5"]
    elif solvent.lower() in ["tip4pew"]:
        O_chg, H_chg, M_chg, M_d, resnames =  0.0,     0.52422, -1.04844, 0.1546, ["HOH", "T4PE"]

    # Load trajectory and partial charges
    trj      = mdtraj.load(segment.trajectory, top = segment.topology)
    trj.xyz *= 10.0                                                             # nm -> A

    # Configure arrays for atom indexes and charges
    all_indexes = []
    chg_indexes = []
    charges     = []
    for residue in trj.topology.residues:
        if verbose and not residue.name in resnames:
            print "WARNING: RESIDUE NAME {0} DOES NOT MATCH EXPECTED ({1})".format(residue.name, resnames)
        else:
            O  = [atom for atom in residue.atoms if atom.name == "O"][0]        # Order of atoms is required for
            H1 = [atom for atom in residue.atoms if atom.name == "H1"][0]       # pseudoatom calculation
            H2 = [atom for atom in residue.atoms if atom.name == "H2"][0]
            all_indexes     += [[O.index, H1.index, H2.index]]
            if O_chg == 0.0:                                                    # If oxygen's charge is 0, we do not
                chg_indexes += [H1.index, H2.index]                             # need to consider it in the dipole
                charges     += [H_chg,    H_chg]                                # calculation
            else:
                chg_indexes += [O.index, H1.index, H2.index]
                charges     += [O_chg,   H_chg,    H_chg]
    all_indexes = np.array(all_indexes)
    chg_indexes = np.array(chg_indexes)
    charges     = np.array(charges)
    charges     = np.column_stack((charges, charges, charges))

    # Calculate dipole vector for each trajectory frame
    dipole             = np.zeros((trj.n_frames, 3))
    for i, frame in enumerate(trj.xyz):
        dipole[i]      = np.sum(frame[chg_indexes] * charges, axis = 0)
        if M_chg is not None:                                                   # Calculate pseudoatom contribution
            pseudo     = ((frame[all_indexes[:,1]] + frame[all_indexes[:,2]]) / 2.0) - frame[all_indexes[:,0]]
            mag        = M_d / np.sqrt(np.sum(pseudo*pseudo, axis = 1))
            pseudo    *= np.column_stack((mag, mag, mag))
            pseudo    += frame[all_indexes[:,0],:]
            dipole[i] += np.sum(pseudo * M_chg, axis = 0)
        if verbose and i % 100 == 0:
            print i, dipole[i], np.mean(np.sqrt(np.sum(dipole[:i] ** 2, axis = 1)))

    # Configure and return results
    attrs = {"method": "mdtraj", "units": "e A", "number of waters": all_indexes.shape[0]}
    return [(segment + "/" + destination, dipole),
            (segment + "/" + destination, attrs)]

def _check_dipole(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    kwargs["destination"] = destination = kwargs.get("destination", "dipole")
    kwargs["solvent"]                   = kwargs.get("solvent",     "TIP3P")
    if    (force
    or not segment + "/" + destination in hdf5_file):
            return [(dipole, segment, kwargs)]
    else:   return False


def coordinates(segment, **kwargs):
    """ Loads coordinates in format compatible with mdtraj """
    trajectory          = mdtraj.load(segment.trajectory, top = segment.topology)
    cell_lengths        = trajectory.unitcell_lengths
    cell_lengths_attr   = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "degrees"}
    cell_angles         = trajectory.unitcell_angles
    cell_angles_attr    = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "nanometers"}
    coordinates         = trajectory.xyz
    coordinates_attr    = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "nanometers"}
    time                = trajectory.time
    time_attr           = {"CLASS" : "EARRAY", "VERSION": 1.0, "TITLE": "", "EXTDIM": 0, "units": "picoseconds"}
    topology_json       = str(topology_to_json(trajectory.topology))
    topology            = np.array(topology_json, dtype="S{}".format(len(topology_json)))
    topology_attr       = {"CLASS" : "ARRAY", "VERSION": 2.3, "TITLE": "", "FLAVOR": "python"}

    return [(segment + "/cell_lengths", cell_lengths),
            (segment + "/cell_lengths", cell_lengths_attr),
            (segment + "/cell_angles",  cell_angles),
            (segment + "/cell_angles",  cell_angles_attr),
            (segment + "/coordinates",  coordinates),
            (segment + "/coordinates",  coordinates_attr),
            (segment + "/time",         time),
            (segment + "/time",         time_attr),
            (segment + "/topology",     topology, {"data_kwargs": {"chunks": False}}),
            (segment + "/topology",     topology_attr)]

def _check_coordinates(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    if    (force
    or not segment + "/coordinates" in hdf5_file):
            return [(coordinates, segment, kwargs)]
    else:   return False


def com_resname(segment, destination, resname, **kwargs):
    """ Calculates center of mass of each instance of <resname> """
    trj         = mdtraj.load(segment.trajectory, top = segment.topology)
    resname_str = ""
    indexes     = []
    masses      = []
    total_mass  = []
    
    for name in resname:
        for i, res in enumerate(trj.topology.residues, 1):
            if res.name == name:
                indexes     += [np.array([a.index        for a in res.atoms], np.int)]
                masses      += [np.array([a.element.mass for a in res.atoms], np.float32)]
                total_mass  += [np.sum(masses[-1])]
                masses[-1]   = np.column_stack((masses[-1], masses[-1], masses[-1]))
                resname_str += "{0} {1} ".format(res.name, i)
    total_mass = np.array(total_mass)
    com        = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
    for i, frame in enumerate(trj.xyz):
        for j, index in enumerate(indexes):
            com[i][j] = np.sum(trj.xyz[i][index] * masses[j], axis = 0) / total_mass[j]
    return  [(segment + "/" + destination, com * 10.0),
             (segment + "/" + destination, {"resname": resname_str[:-1], "method": "mdtraj", "units": "A"})]

def _check_com_resname(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    kwargs["destination"] = destination = kwargs.get("destination", "com")
    kwargs["resname"]     = resname     = kwargs.get("resname",     ["HOH"])
    if isinstance(resname, types.StringType):
        kwargs["resname"] = [resname]
    if    (force
    or not segment + "/" + destination in hdf5_file):
            return [(com_resname, segment, kwargs)]
    else:   return False


def com_selection(segment, destination, selection, name, **kwargs):
    """
    Calculates center of mass of selection(s) given by residues selected by <selection>
    """

    # Load trajectory and initialize variables
    trj         = mdtraj.load(segment.trajectory, top = segment.topology)       # Trajectory
    indexes     = []                                                            # Indexes of each atom in each selection
    masses      = []                                                            # Masses of each atom in each selection
    total_mass  = []                                                            # Total mass of each selection
    sel_str     = ""
    
    # Prepare atom index and mass arrays
    for sel in selection:
        indexes += [[]]
        masses  += [[]]
        for res in trj.topology.residues:
            if res.index + 1 in sel:
                indexes[-1] += [a.index        for a in res.atoms]
                masses[-1]  += [a.element.mass for a in res.atoms]
                sel_str     += "{0} {1} ".format(res.name, res.index + 1)
        sel_str = sel_str[:-1] + "\n"
        indexes[-1] = np.array(indexes[-1])
        masses[-1]  = np.array(masses[-1])
        masses[-1]  = np.column_stack((masses[-1], masses[-1], masses[-1]))
        total_mass += [np.sum(masses[-1], axis = 0)]
    sel_str     = sel_str[:-1]
    com         = np.zeros((trj.n_frames, len(selection), 3), np.float32)
    if kwargs.get("debug", False):
        for i, sel in enumerate(selection):
            print "Selection {0} includes '{1}' and contains {2} atoms".format(i, sel_str.split("\n")[i], len(indexes[i]))
    name_str    = "\n".join(name)

    # Loop over trajectory and calculate centers of mass
    for i, frame in enumerate(trj.xyz):
        for j, sel in enumerate(selection):
            com[i,j] = np.sum(trj.xyz[i][indexes[j]] * masses[j], axis = 0) / total_mass[j]
    com *= 10.0

    # Return results
    return  [(segment + "/" + destination, com),
             (segment + "/" + destination, {"selection": sel_str, "name": name_str, "method": "mdtraj", "units": "A"})]

def _check_com_selection(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    kwargs["destination"] = destination = kwargs.get("destination", "com")
    kwargs["selection"]   =               kwargs.get("selection",   [[1]])
    kwargs["name"]        =               kwargs.get("name",        "1")
    if    (force
    or not segment + "/" + destination in hdf5_file):
            return [(com_selection, segment, kwargs)]
    else:   return False


