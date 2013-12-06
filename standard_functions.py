#!/usr/bin/python
desc = """MD_toolkit.standard_functions.py
    Standard functions
    Written by Karl Debiec on 13-02-03
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, subprocess, sys
from   importlib import import_module
import numpy as np
####################################################### CLASSES ########################################################
class Segment:
    def __init__(self, number, path, topology = None, trajectory = None, files = None):
        self.number     = number
        self.path       = path
        self.topology   = topology
        self.trajectory = trajectory
        self.files      = files
    def __float__(self):        return float(self.number)
    def __str__(self):          return self.number
    def __repr__(self):         return self.number
    def __add__(self, other):   return str(self.number) + other
    def __radd__(self, other):  return other + str(self.number)
    def __getitem__(self, extension):
        matches = [f for f in self.files if f.endswith(extension)]
        if   len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            raise Exception("No files with extension {0} present\n".format(extension) +
                            "Files present: {1}".format(self.files))
        else:
            raise Exception("Multiple files with extension {0} present\n".format(extension) +
                            "Files present: {1}".format(self.files))
class Function_to_Method_Wrapper:
    def __init__(self, host, function):
        self.host       = host
        self.function   = function
        setattr(host, function.__name__, self)
    def __call__(self, *args, **kwargs):
        args    = [self.host] + list(args)
        return self.function(*args, **kwargs)
################################################ SEGMENT LIST FUNCTIONS ################################################
def segments_standard(sim_root):
    """ Lists segment files, topologies, and trajectories at <sim_root>, assuming the format ####/####.* """
    segments = []
    for seg_dir in sorted([f for f in os.listdir(sim_root) if is_num(f)]):
        files      = ["{0}/{1}/{2}".format(sim_root, seg_dir, f)
                       for f in os.listdir("{0}/{1}/".format(sim_root, seg_dir))]
        topology   = "{0}/{1}/{1}_solute.pdb".format(sim_root, seg_dir)
        trajectory = "{0}/{1}/{1}_solute.xtc".format(sim_root, seg_dir)
        if not os.path.isfile(topology):   topology   = None
        if not os.path.isfile(trajectory): trajectory = None
        segments  += [Segment(number     = seg_dir,
                              path       = "{0}/{1}/".format(sim_root, seg_dir),
                              topology   = topology,
                              trajectory = trajectory,
                              files      = files)]
    return segments
################################################## GENERAL FUNCTIONS ###################################################
def ignore_index(time, ignore):
    if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
    elif ignore == 0:   return 0
    elif ignore >  0:   return np.where(time > ignore)[0][0]
def is_num(test):
    try:    float(test)
    except: return False
    return  True
def month(string):
    month = {"jan":  1, "feb":  2, "mar":  3, "apr":  4, "may":  5, "jun":  6,
             "jul":  7, "aug":  8, "sep":  9, "oct": 10, "nov": 11, "dec": 12}
    try:    return month[string.lower()]
    except: return None
def string_to_function(module_function):
    module, function = ".".join(module_function.split(".")[:-1]), module_function.split(".")[-1]
    return getattr(import_module(module), function)
def _shell_executor(command):
    pipe    = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    for line in iter(pipe.stdout.readline, ""):
        yield line.rstrip().replace("\n", " ")
    pipe.wait()
def shell_iterator(command, leader = "", verbose = False, **kwargs):
    if verbose:     print leader + command
    pipe    = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, **kwargs)
    for line in iter(pipe.stdout.readline, ""):
        if verbose: print leader + line.rstrip().replace("\n", " ")
        yield             leader + line.rstrip().replace("\n", " ")
    pipe.wait()
def block_average(data, func = np.mean, func_kwargs = {"axis": 1}, min_size = 1, **kwargs):
    full_size   = data.size
    sizes       = [s for s in list(set([full_size / s for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    sds         = np.zeros(sizes.size)
    n_blocks    = full_size // sizes
    for i, size in enumerate(sizes):
        resized = np.resize(data, (full_size // size, size))
        values  = func(resized, **func_kwargs)
        sds[i]  = np.std(values)
    ses                 = sds / np.sqrt(n_blocks - 1.0)
#    se_sds              = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
    se_sds              = (1.0 / np.sqrt(2.0 * (n_blocks - 1.0))) * ses
    if ses[-1] == 0.0 or se_sds[-1] == 0.0:                                     # This happens occasionally and
        sizes   = sizes[:-1]                                                    # disrupts curve_fit; it is not clear
        ses     = ses[:-1]                                                      # why
        se_sds  = se_sds[:-1]
    return sizes, ses, se_sds
def fit_curve(fit_func = "single_exponential", **kwargs):
    import warnings
    from   scipy.optimize import curve_fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        def single_exponential(x, y, **kwargs):
            def func(x, a, b, c):       return a + b * np.exp(c * x)
            a, b, c       = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, func(x, a, b, c)
        def double_exponential(x, y, **kwargs):
            def func(x, a, b, c, d, e): return a + b * np.exp(c * x) + d * np.exp(e * x)
            a, b, c, d, e = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, e, func(x, a, b, c, d, e)
        def sigmoid(x, y, **kwargs):
            def func(x, a, b, c, d):    return b + (a - b) / (1.0 + (x / c) ** d)
            a, b, c, d    = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, func(x, a, b, c, d)
        return locals()[fit_func](**kwargs)
################################################## ANALYSIS FUNCTIONS ##################################################
def _contact_1D_to_2D_map(contact_1D):
    """ Converts a 1D (sparse) contact map <contact_1D> to a 2D (complete) contact map """
    n_res       = int(1 + np.sqrt(1 + 8 * contact_1D.size)) / 2
    indexes     = contact_1D_to_2D_indexes(n_res)
    contact_2D  = np.zeros((n_res, n_res), np.int8)
    contact_2D[indexes[:,0], indexes[:,1]]  = contact_1D
    contact_2D[indexes[:,1], indexes[:,0]]  = contact_1D
    contact_2D[range(n_res), range(n_res)]  = 1
    return contact_2D
def _contact_2D_to_1D_indexes(n_res):
    """ Generates indexes for conversion of 2D (complete) contact map to a 1D (sparse) contact map of <n_res> """
    indexes = np.zeros((n_res,n_res), np.int)
    i       = 0
    for j in range(n_res-1):
        for k in range(j+1,n_res):
            indexes[j, k]   = i
            i              += 1
    return indexes
def _contact_1D_to_2D_indexes(n_res):
    """ Generates indexes for conversion of 1D (sparse) contact map to a 2D (complete) contact map of <n_res> """
    indexes = np.zeros(((n_res**2-n_res)/2,2), np.int)
    i       = 0
    for j in range(n_res-1):
        for k in range(j+1,n_res):
            indexes[i]  = [j, k]
            i          += 1
    return indexes

######################################################## MDTRAJ ########################################################
def topology_to_json(topology):
    """Generates a json string from a MDTraj topology
    Adapted from 'topology.py' in MDTraj"""
    import json
    topology_dict   = {"chains":  [],
                       "bonds":   []} 

    chain_iter      = topology.chains
    if not hasattr(chain_iter, "__iter__"):
        chain_iter  = chain_iter()
    for chain in chain_iter:
        chain_dict  = {"residues":  [], 
                       "index":     int(chain.index)}

        
        if not hasattr(residue_iter, "__iter__"):
            residue_iter    = residue_iter()
        for residue in residue_iter:
            residue_dict    = {"index": int(residue.index),
                               "name":  str(residue.name),
                               "atoms": []} 

            atom_iter       = residue.atoms
            if not hasattr(atom_iter, "__iter__"):
                atom_iter   = atom_iter()
            for atom in atom_iter:
                residue_dict["atoms"].append({"index":   int(atom.index),
                                              "name":    str(atom.name),
                                              "element": str(atom.element.symbol)})
            chain_dict["residues"].append(residue_dict)
        topology_dict["chains"].append(chain_dict)

    bond_iter       = topology.bonds
    if not hasattr(bond_iter, "__iter__"):
        bond_iter   = bond_iter()
    for atom1, atom2 in bond_iter:
        topology_dict["bonds"].append([int(atom1.index),
                                       int(atom2.index)])
    return json.dumps(topology_dict)

def topology_from_json(topology_json):
    import operator
    import json
    from mdtraj import Topology
    import mdtraj.pdb.element as elem

    topology_dict   = json.loads(topology_json)
    topology        = Topology()

    for chain_dict in sorted(topology_dict['chains'], key=operator.itemgetter('index')):
        chain = topology.add_chain()
        for residue_dict in sorted(chain_dict['residues'], key=operator.itemgetter('index')):
            residue = topology.add_residue(residue_dict['name'], chain)
            for atom_dict in sorted(residue_dict['atoms'], key=operator.itemgetter('index')):
                try:
                    element = elem.get_by_symbol(atom_dict['element'])
                except KeyError:
                    raise ValueError('The symbol %s isn\'t a valid element' % atom_dict['element'])
                topology.add_atom(atom_dict['name'], element, residue)

    atoms = list(topology.atoms)
    for index1, index2 in topology_dict['bonds']:
        topology.add_bond(atoms[index1], atoms[index2])

    return topology

def load_charges_from_cms(trj, cms, verbose = False, debug = False, **kwargs):
    """ Parses partial charge information from a Desmond/Anton <cms> topology/force field file and adds the results into
        the topology of mdtraj trajectory <trj>. This function is not yet robust and care should be taken to make sure
        it behaves as expected. """

    # Load raw data from cms file
    awk_command ="""
        /sites/{ in_sites=1; in_atoms=0; next }
        /\}/{    in_sites=0; in_atoms=0; next }
        /:::/{
            if (in_sites == 1) {
                if (in_atoms == 0) {in_atoms = 1}
                else               {in_atoms = 0}}}
        in_atoms"""
    command   = "cat {0} | awk '{1}' | grep -v :::".format(cms, awk_command)
    raw_data  =  commands.getoutput(command).split("\n")
    dtype     = np.dtype([("atom number", "i4"), ("residue number", "i4"), ("residue name", "S10"),
                          ("charge",      "f4"), ("mass",           "f4"), ("vdw type",     "S10")])
    if debug:
        for line in raw_data:
            print line
    all_sites = np.array([(a,f,g,c,d,e) for a,b,c,d,e,f,g in [d.split() for d in raw_data]], dtype)
    if verbose:
        print "ALL SITES"
        for s in all_sites: print s

    # Separate solvent charge data from solute charge data
    solute_sites  = all_sites[np.where((all_sites["residue name"] != "SPCE") &
                                       (all_sites["residue name"] != "T3P")  & (all_sites["residue name"] != "TIP3")   &
                                       (all_sites["residue name"] != "T4P")  & (all_sites["residue name"] != "TIP4")   &
                                       (all_sites["residue name"] != "T4P5") & (all_sites["residue name"] != "TIP405") &
                                       (all_sites["residue name"] != "Na+")  & (all_sites["residue name"] != "Cl-"))[0]]
    solvent_sites = all_sites[np.where((all_sites["residue name"] == "SPCE") |
                                       (all_sites["residue name"] == "T3P")  | (all_sites["residue name"] == "TIP3")   |
                                       (all_sites["residue name"] == "T4P")  | (all_sites["residue name"] != "TIP4")   |
                                       (all_sites["residue name"] == "T4P5") | (all_sites["residue name"] != "TIP405") |
                                       (all_sites["residue name"] == "Na+")  | (all_sites["residue name"] == "Cl-"))[0]]
    solvent_charge = {}
    solvent_mass   = {}
    for atom in solvent_sites:
        if   (int(round(atom["mass"])) == 16):
            solvent_charge["O"] = atom["charge"]
            solvent_mass["O"]   = atom["mass"]
        elif (int(round(atom["mass"])) ==  1):
            solvent_charge["H"] = atom["charge"]
            solvent_mass["H"]   = atom["mass"]
    if verbose:
        print "\nSOLUTE SITES"
        for s in solute_sites: print s
        print "\nSOLVENT SITES"
        for s in solvent_sites: print s

    # Copy charge information from cms file into mdtraj topology
    for i, atom in enumerate(trj.topology.atoms):
        if   i < solute_sites.size:
            if   (int(round(atom.element.mass)) != int(round(solute_sites[i]["mass"]))):
                raise Exception("MASSES OF ATOM {0} ({1}, {2}) DO NOT MATCH".format(
                        i, atom.element.mass, solute_sites[i]["mass"]))
            elif (atom.residue.name             != solute_sites[i]["residue name"]):
                if   solute_sites[i]["residue name"].lower().startswith(atom.residue.name.lower()):     # Salt
                    if verbose:
                        print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                            i, atom.residue.name, solute_sites[i]["residue name"]))
                elif solute_sites[i]["residue name"].lower().endswith(atom.residue.name.lower()):       # Termini
                    if verbose:
                        print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                            i, atom.residue.name, solute_sites[i]["residue name"]))
                elif (solute_sites[i]["residue name"].lower().startswith("hi") and                      # Histidine
                      atom.residue.name.lower().startswith("hi")):
                    if verbose:
                        print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                            i, atom.residue.name, solute_sites[i]["residue name"]))
                elif (solute_sites[i]["residue name"].lower() == "tip3" and                             # TIP3P
                      atom.residue.name.lower()            == "t3p"):
                    if verbose:
                        print("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH, CONTINUING".format(
                            i, atom.residue.name, solute_sites[i]["residue name"]))
                else:
                    raise Exception("RESIDUE NAMES of ATOM {0} ({1}, {2}) DO NOT MATCH".format(
                        i, solute_sites[i]["residue name"], atom.residue.name))
            atom.charge = solute_sites[i]["charge"]
            atom.mass   = solute_sites[i]["mass"]
        elif ((int(round(atom.element.mass)) == 35)                                                     # Chloride
        or    (atom.name.lower() == "cl" and atom.residue.name.lower() == "cl")):
            atom.charge = -1.0
            atom.mass   = 35.450000
        elif ((int(round(atom.element.mass)) == 23)
        or    (atom.name.lower() == "na" and atom.residue.name.lower() == "na")):                       # Sodium
            atom.charge = 1.0
            atom.mass   = 22.989770
        elif (atom.residue.name in ["HOH", "WAT", "T3P", "T4P", "T4P5", "SPC"]):                        # Water
            if atom.name == "pseu": continue
            atom.charge = solvent_charge[atom.name[0]]
            atom.mass   = solvent_mass[atom.name[0]]
        else:
            raise Exception("UNRECOGNIZED ATOM {0} in RESIDUE {1}".format(atom.name, atom.residue.name))


