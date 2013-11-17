#!/usr/bin/python
desc = """MD_toolkit.primary.vmd.__init__.py
    Functions for primary analysis of MD trajectories using Visual Molecular Dynamics
    Written by Karl Debiec on 13-03-06
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   MD_toolkit.standard_functions import _shell_executor
################################################## ANALYSIS FUNCTIONS ##################################################
def rmsd(segment, reference, vmd = "vmd", domain = "", selection = "protein and name CA", **kwargs):
    """ Calculates rmsd of <domain> relative to <reference> with <selection> using <vmd> """
    script      = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["tcl", "vmd_rmsd.tcl"])
    command     = "{0} -dispdev text -e {1} -args {2} {3} {4} \"{5}\"".format(vmd, script, segment.topology,
                                                                              segment.trajectory, reference, selection)
    for line in _shell_executor(command):
        if line.startswith("N_FRAMES"):
            n_frames    = float(line.split()[1])
            rmsd        = np.zeros(n_frames,      dtype = np.float32)
            rotmat      = np.zeros((n_frames, 9), dtype = np.float32)
            i           = 0
        elif line.startswith("ROTMAT"):
            rotmat[i]   = line.split()[1:]
        elif line.startswith("RMSD"):
            rmsd[i]     = line.split()[1]
            i          += 1
    return  [(segment + "/rmsd_"   + domain,  rmsd),
             (segment + "/rotmat_" + domain,  rotmat),
             (segment + "/rmsd_"   + domain,  {"selection": selection, "method": "vmd", "units": "A"}),
             (segment + "/rotmat_" + domain,  {"selection": selection, "method": "vmd"})]
def _check_rmsd(hdf5_file, segment, force = False, **kwargs):
    if not (segment.topology   and os.path.isfile(segment.topology)
    and     segment.trajectory and os.path.isfile(segment.trajectory)):
            return False
    domain  = kwargs.get("domain", "")
    if (force
    or  not [segment + "/rmsd_"   + domain,
             segment + "/rotmat_" + domain] in hdf5_file):
            return [(rmsd, segment, kwargs)]
    else:   return False