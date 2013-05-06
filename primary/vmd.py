#!/usr/bin/python
desc = """vmd.py
    Functions for primary analysis of molecular dynamics trajectories using vmd
    Written by Karl Debiec on 13-03-06
    Last updated 13-05-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   standard_functions import _shell_executor
################################################## ANALYSIS FUNCTIONS ##################################################
def rmsd(segment, **kwargs):
    """ Calculates rmsd of <domain> relative to <reference> with <selection> using <vmd> """
    vmd         = kwargs.get("vmd",       "vmd")
    domain      = kwargs.get("domain",    "")
    selection   = kwargs.get("selection", "protein and name CA")
    reference   = kwargs.get("reference")
    script      = "/".join(os.path.abspath(__file__).split('/')[:-2] + ["tcl", "vmd_rmsd.tcl"])
    command     = "{0} -dispdev text -e {1} -args {2} {3} {4} \"{5}\"".format(vmd, script, segment.topology,
                                                                              segment.trajectory, reference, selection)
    for line in _shell_executor(command):
        if line.startswith("N_FRAMES"):
            n_frames    = float(line.split()[1])
            rmsd        = np.zeros(n_frames)
            rotmat      = np.zeros((n_frames, 9))
            i           = 0
        elif line.startswith("ROTMAT"):
            rotmat[i]   = line.split()[1:]
        elif line.startswith("RMSD"):
            rmsd[i]     = line.split()[1]
            i          += 1
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
