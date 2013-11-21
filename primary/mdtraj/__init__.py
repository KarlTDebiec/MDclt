#!/usr/bin/python
desc = """MD_toolkit.primary.mdtraj.__init__.py
    Functions for primary analysis of MD trajectories
    Written by Karl Debiec on 13-10-30
    Last updated by Karl Debiec on 13-11-17"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys, types, warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mdtraj
from MD_toolkit.standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
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
    mean       = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
    std        = np.zeros((trj.n_frames, len(indexes), 3), np.float32)
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

def dipole(segment, destination, cms_file, solvent, **kwargs):
    """ Calculates dipole moment of system. Loads partial charges from <cms_file> of <cms_file_format>. Calculates
        and includes pseudoatoms if <solvent> is a four-point water model """

    # Load trajectory using mdtraj
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
    trj = mdtraj.load(segment.trajectory, top = segment.topology)

    # Load data from Desmond cms format
    awk_command ="""
        /sites/{
            in_sites=1; in_atoms=0; next }
        /\}/{
            in_sites=0; in_atoms=0; next }
        /:::/{
            if (in_sites == 1) {
                if (in_atoms == 0) {in_atoms = 1}
                else               {in_atoms = 0}}}
        in_atoms"""
    command   = "cat {0} | awk '{1}' | grep -v :::".format(cms_file, awk_command)
    raw_data  =  commands.getoutput(command).split("\n")
    dtype     = np.dtype([("atom number", "i4"), ("residue number", "i4"), ("residue name", "S10"),
                          ("charge",      "f4"), ("mass",           "f4"), ("vdw type",     "S10")])
    cms_sites = np.array([(a,f,g,c,d,e) for a,b,c,d,e,f,g in [d.split() for d in raw_data]], dtype)
    for s in cms_sites: print s

    # Separate solvent charge data from solute charge data
    solvent_charge = {}
    solvent_mass   = {}
    for atom in cms_sites[-3:]:
        if   (int(round(atom["mass"])) == 16):
            solvent_charge["O"] = atom["charge"]
            solvent_mass["O"]   = atom["mass"]
        elif (int(round(atom["mass"])) ==  1):
            solvent_charge["H"] = atom["charge"]
            solvent_mass["H"]   = atom["mass"]
    cms_sites       = cms_sites[np.where((cms_sites["residue name"] != "SPCE") &
                                         (cms_sites["residue name"] != "Na+")  &
                                         (cms_sites["residue name"] != "Cl-"))[0]]
    for s in cms_sites: print s

    # Copy charge information from cms file into mdtraj topology
    for i, atom in enumerate(trj.topology.atoms):
        if   i < cms_sites.size:
            if   (int(round(atom.element.mass)) != int(round(cms_sites[i]["mass"]))):
                raise Exception("MASSES OF ATOM {0} ({1}, {2}) DO NOT MATCH".format(
                        i, atom.element.mass, cms_sites[i]["mass"]))
            elif (atom.residue.name             != cms_sites[i]["residue name"]):
                if   cms_sites[i]["residue name"].lower().startswith(atom.residue.name.lower()):
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                elif cms_sites[i]["residue name"].lower().endswith(atom.residue.name.lower()):
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                elif (cms_sites[i]["residue name"].lower().startswith("hi") and
                      atom.residue.name.lower().startswith("hi")):
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                else:
                    raise Exception("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
            atom.charge = cms_sites[i]["charge"]
            atom.mass   = cms_sites[i]["mass"]
        elif (int(round(atom.element.mass)) == 35):
            atom.charge = -1.0
            atom.mass   = atom.element.mass
        elif (int(round(atom.element.mass)) == 23):
            atom.charge = 1.0
            atom.mass   = atom.element.mass
        elif (atom.residue.name in ["HOH", "WAT", "T3P", "SPC"]):
            atom.charge = solvent_charge[atom.name[0]]
            atom.mass   = solvent_mass[atom.name[0]]
        else:
            raise Exception("UNRECOGNIZED ATOM {0} in RESIDUE {1}".format(atom.name, atom.residue.name))
#        print atom.residue.name, atom.name, atom.charge

    # Calculate dipole moment
    dipole       = np.zeros((trj.n_frames, 3))
    net_dipole   = np.zeros(trj.n_frames)
    for i, frame in enumerate(trj.xyz):
        for residue in trj.topology.residues:
            for atom in residue.atoms:
                dipole[i] += atom.charge * frame[atom.index] * 10
        net_dipole[i]  = np.sqrt(np.sum(dipole[i] ** 2))
        print i, net_dipole[i]
    attrs = {"cms_file": cms_file, "solvent": solvent, "method": "mdtraj", "units": "e A"}
    return  [(segment + "/" + destination, net_dipole),
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


