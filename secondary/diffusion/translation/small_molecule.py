#!/usr/bin/python
desc = """small_molecule.py
    Functions for analysis of translational diffusion of small molecules
    Written by Karl Debiec on 13-02-08
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   collections import OrderedDict
################################################## ANALYSIS FUNCTIONS ##################################################
def small_molecule(hdf5_file,
        time,                                               # Simulation time
        com,                                                # Centers of mass of each molecule of each residue type
        delta_ts,                                           # Delta_t(s) (ns)
        selection,                                          # Residue types and numbers for each molecule
        destination,                                        # Destination of secondary data
        verbose = False, n_cores = 1, **kwargs):

    # Prepare residue types and indexes
    dt          = time[1] - time[0]
    dtype       = [("delta t", "f4")]
    res_indexes = OrderedDict()
    for i, resname in enumerate(selection.split()[::2]):
        res_indexes[resname]    = res_indexes.get(resname, []) + [i]
    for resname in res_indexes:
        res_indexes[resname]    = np.array(res_indexes[resname])
        dtype                  += [("{0} D ({1})".format(resname, res_indexes[resname].size), "f4")]
        dtype                  += [("{0} D se".format(resname), "f4")]

    # Calculate translational diffusion
    data    = []
    for delta_t in delta_ts:
        data           += [[delta_t]]
        delta_t_index   = int(np.round(delta_t / dt))
        delta_xyz_2     = np.sum((com[delta_t_index:] - com[:-delta_t_index]) ** 2, axis = 2)
        D               = np.mean(delta_xyz_2, axis = 0) / (6 * delta_t) / 1000.0
        for res, indexes in res_indexes.iteritems():
            data[-1]   += [np.mean(D[indexes]), np.std(D[indexes])]

    # Organize and return data
    data    = np.array([tuple(D) for D in data], np.dtype(dtype))
    attrs   = {"time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1])),
               "delta t units": "ns", "D units": "A2 ps-1"}
    if verbose:     _print_small_molecule(data, attrs)
    return  [(destination, data),
             (destination, attrs)]
def _check_small_molecule(hdf5_file, force = False, **kwargs):
    def _ignore_index(time, ignore):
        if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
        elif ignore == 0:   return 0
        elif ignore >  0:   return np.where(time > ignore)[0][0]

    # Parse kwargs and set defaults
    source                              = kwargs.get("source",      "*/com_unwrap")
    source                              = source if source.startswith("*/") else "*/" + source
    ignore                              = kwargs.pop("ignore",      0)
    kwargs["delta_ts"]    = delta_ts    = kwargs.get("delta_ts",    np.array([0.1]))
    kwargs["destination"] = destination = kwargs.get("destination", "/diffusion/translation")
    kwargs["selection"]                 = hdf5_file.attrs("0000/com")["resname"]                    # LAZY HACK; FIX

    # If analysis has not been run previously, run analysis
    if     (force
    or not (destination in hdf5_file)):
        log             = hdf5_file.load("*/log", type = "table")
        ignore_index    = _ignore_index(log["time"], ignore)
        kwargs["time"]  = log["time"][ignore_index:]
        kwargs["com"]   = hdf5_file.load(source)[ignore_index:]
        return [(small_molecule, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    data            = hdf5_file[destination]
    attrs           = hdf5_file.attrs(destination)
    log             = hdf5_file.load("*/log", type = "table")
    ignore_index    = _ignore_index(log["time"], ignore)
    kwargs["time"]  = log["time"][ignore_index:]
    if (np.any(data["delta t"] != np.array(delta_ts, np.float32))
    or (attrs["time"]          != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        kwargs["com"]   = hdf5_file.load(source)[ignore_index:]
        return [(small_molecule, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False):    _print_small_molecule(data, attrs)
    return False
def _print_small_molecule(data, attrs):
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    for name in data.dtype.names: print "{0:<12}".format(name),
    print
    for D in data:
        for a in D: print "{0:<12.8f}".format(float(a)),
        print 


