#!/usr/bin/python
desc = """translation.py
    Functions for analysis of translational diffusion
    Written by Karl Debiec on 13-02-08
    Last updated 13-10-31"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import time as time_module
import numpy as np
from   collections import OrderedDict
################################################## ANALYSIS FUNCTIONS ##################################################
def small_molecule(hdf5_file,
                   time,                                    # Time
                   com,                                     # Centers of mass of each molecule of each residue type
                   delta_ts,                                # Delta_t(s) (ns)
                   destination,                             # Destination of secondary data
                   selection,                               # Residue types and numbers for each molecule
                   verbose = False, n_cores = 1, **kwargs):
    size        = time.size
    dt          = time[1] - time[0]
    dtype       = [("delta t", "f4")]
    res_indexes = OrderedDict()
    for i, resname in enumerate(selection.split()[::2]):
        res_indexes[resname]    = res_indexes.get(resname, []) + [i]
    for resname in res_indexes:
        res_indexes[resname]    = np.array(res_indexes[resname])
        dtype                  += [("{0} D ({1})".format(resname, res_indexes[resname].size), "f4")]
        dtype                  += [("{0} D se".format(resname), "f4")]
    table   = []
    attrs   = {"time": "{0:.3f} {1:.3f}".format(time[0], time[-1]), "delta t units": "ns", "D units": "A2 ps-1"}
    for delta_t in delta_ts:
        table          += [[delta_t]]
        delta_t_index   = int(np.round(delta_t / dt))
        delta_xyz_2     = np.sum((com[delta_t_index:] - com[:-delta_t_index]) ** 2, axis = 2)
        D               = np.mean(delta_xyz_2, axis = 0) / (6 * delta_t) / 1000.0
        for res, indexes in res_indexes.iteritems():
            table[-1]  += [np.mean(D[indexes]), np.std(D[indexes])]
    table   = np.array([tuple(D) for D in table], np.dtype(dtype))
    if verbose:     _print_small_molecule(table, attrs)
    return  [(destination, table),
             (destination, attrs)]
def _check_small_molecule(hdf5_file, force = False, **kwargs):
    def _ignore_time(time, dataset, ignore):
        if   ignore <  0:     return dataset[time > time[-1] + ignore - (time[1] - time[0])]
        elif ignore == 0:     return dataset
        elif ignore >  0:     return dataset[time > ignore]
    source                  = kwargs.get("source", "*/com_unwrap")
    ignore                  = kwargs.get("ignore", 0)
    kwargs["delta_ts"]      = delta_ts    = kwargs.get("delta_ts",    np.array([0.1]))
    kwargs["destination"]   = destination = kwargs.get("destination", "/diffusion/translation")
    kwargs["selection"]     = hdf5_file.attrs("0000/com")["resname"]

    if (force
    or not(destination in hdf5_file)):
        time  = hdf5_file.load("*/log", type = "table")["time"]
        com   = hdf5_file.load(source if source.startswith("*/") else "*/" + source)
        kwargs["time"]  = _ignore_time(time, time, ignore)
        kwargs["com"]   = _ignore_time(time, com,  ignore)
        return [(small_molecule, kwargs)]

    data            = hdf5_file[destination]
    attrs           = hdf5_file.attrs(destination)
    time            = hdf5_file.load("*/log", type = "table")["time"]
    kwargs["time"]  = _ignore_time(time, time, ignore)
    if (np.any(data["delta t"] != np.array(delta_ts, np.float32))
    or (attrs["time"]          != "{0:.3f} {1:.3f}".format(kwargs["time"][0], kwargs["time"][-1]))):
        com             = hdf5_file.load(source if source.startswith("*/") else "*/" + source)
        kwargs["com"]   = _ignore_time(time, com, ignore)
        return [(small_molecule, kwargs)]
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


