#!/usr/bin/python
desc = """MD_toolkit.primary_x.mdtraj.__init__.py
    Functions for primary analysis using MDTraj
    Written by Marissa Pacey on 13-09-16
    Last updated by Karl Debiec on 13-11-21"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, json, os, sys, warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mdtraj
from   MD_toolkit.standard_functions import topology_to_json
################################################## ANALYSIS FUNCTIONS ##################################################
def dipole(segments, destination, cms_file, side_length, **kwargs):
    """ Calculates dipole moment of system. Loads partial charges from <cms_file> of <cms_file_format>. Calculates
        and includes pseudoatoms if <solvent> is a four-point water model """

    # Load trajectory using mdtraj
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
    trj = mdtraj.load(segments[0].trajectory, top = segments[0].topology)
    
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
    print
    
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
                                         (cms_sites["residue name"] != "TIP3")  &
                                         (cms_sites["residue name"] != "T3P")  &
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
                if   cms_sites[i]["residue name"].lower().startswith(atom.residue.name.lower()):    # Salt
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                elif cms_sites[i]["residue name"].lower().endswith(atom.residue.name.lower()):      # Termini
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                elif (cms_sites[i]["residue name"].lower().startswith("hi") and                     # Histidine
                      atom.residue.name.lower().startswith("hi")):
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                elif (cms_sites[i]["residue name"].lower() == "tip3" and                             # TIP3P
                      atom.residue.name.lower()            == "t3p"):
                    print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                        i, atom.residue.name, cms_sites[i]["residue name"]))
                else:
                    raise Exception("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH".format(
                        i, cms_sites[i]["residue name"], atom.residue.name))
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

    # Calculate dipole moment
    inner_cutoff = side_length / 4
    last_sign    = None
    offset       = np.zeros_like(trj.xyz[0])
    segment      = segments.pop(0)
    trj.xyz     *= 10.0
    while True:
        dipole       = np.zeros((trj.n_frames, 3))
        for i, frame in enumerate(trj.xyz):
            sign       = np.sign(frame)
            delta_sign = sign - last_sign if last_sign is not None else np.zeros_like(sign)
            last_sign  = sign
            offset[(delta_sign == -2) & (np.abs(frame) > inner_cutoff)] += side_length
            offset[(delta_sign ==  2) & (np.abs(frame) > inner_cutoff)] -= side_length
            frame     += offset
            for residue in trj.topology.residues:
                if residue.name != "T3P": continue
                for atom in residue.atoms:
                    dipole[i] += atom.charge * frame[atom.index]
            if i % 100 == 0:
                print i, dipole[i], np.mean(np.sqrt(np.sum(dipole[:i] ** 2, axis = 1)))
        attrs = {"cms_file": cms_file, "method": "mdtraj", "units": "e A"}
        yield (segment + "/" + destination, dipole)
        yield (segment + "/" + destination, attrs)

        trj.xyz /= 10.0
        trj.save("{0}_unwrapped.xtc".format(segment))

        if len(segments) != 0:
            segment  = segments.pop(0)
            trj      = mdtraj.load(segment.trajectory, top = trj.topology)
            trj.xyz *= 10.0
        else:                  break
def _check_dipole(hdf5_file, force = False, **kwargs):
    kwargs["segments"]    = segments    = [s for s in kwargs.get("segments", [])
                                             if  s.topology   and os.path.isfile(s.topology)
                                             and s.trajectory and os.path.isfile(s.trajectory)]
    kwargs["destination"] = destination = kwargs.get("destination", "dipole")
    kwargs["side_length"] = kwargs.get("side_length",np.mean(hdf5_file.load("*/log",type="table")["volume"])**(1.0/3.0))
    if    (force
    or not [s + destination for s in segments] in hdf5_file):
            return [(dipole, kwargs)]
    else:   return False


def topology(destination, infile, **kwargs):
    """ Parses a pdb <infile> using MDTraj, and stores the resulting topology as a json string. 
        Adapted from 'topology.py' in MDTraj """
    topology = mdtraj.load(infile).topology
    topology = topology_to_json(topology)

    yield (destination, topology, {"data_kwargs": {"chunks": False}})

def _check_topology(hdf5_file, force = False, **kwargs):
    kwargs["destination"]   = destination   = kwargs.get("destination", "topology")
    kwargs["infile"]        = infile        = kwargs.get("infile", [s for s in kwargs.get("segments", [])
                                              if  s.topology   and os.path.isfile(s.topology)][0].topology)
    if    (force
    or not "topology" in hdf5_file):    return [(topology, kwargs)]
    else:                               return False


