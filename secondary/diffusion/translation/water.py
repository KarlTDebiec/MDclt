#!/usr/bin/python
desc = """MD_toolkit.secondary.diffusion.water.py
    Functions for analysis of translational diffusion of water
    Written by Karl Debiec on 13-11-17
    Last updated by Karl Debiec on 13-11-17"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   collections import OrderedDict
################################################## ANALYSIS FUNCTIONS ##################################################
def water(hdf5_file,
        source,                                             # Source of primary data
        ignore_index,                                       # Index from which to start analysis
        time,                                               # Simulation time
        delta_ts,                                           # Delta_t(s) (ns)
        destination,                                        # Destination of secondary data
        verbose = False, n_cores = 1, **kwargs):
    def _load_com_water(self, path, ignore_index = 0, start_index = 0, end_index = None, **loader_kwargs):
        segments = self._segments()
        shapes   = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in segments])
        n_frames = np.sum(shapes[:,0]) - ignore_index
        if   end_index is None:            end_index = shapes[0,1]
        elif end_index > shapes[0,1]:      end_index = shapes[0,1]
        if   end_index - start_index <= 0: return None
        data     = np.zeros((n_frames, end_index - start_index, 3), np.float32)
        i        = 0
        for segment in segments:
            new_data  = np.array(self.hierarchy[self._strip_path(segment + "/" + path[2:])][:,start_index:end_index,:])
            if  ignore_index  > new_data.shape[0]:
                ignore_index -= new_data.shape[0]
                continue
            elif ignore_index > 0:
                new_data      = new_data[ignore_index:]
                ignore_index  = 0
            data[i:i+new_data.shape[0]] = new_data
            i                          += new_data.shape[0]
        return data
    # Prepare dt
    dt          = time[1] - time[0]

    # Calculate translational diffusion
    data    = []
    for delta_t in delta_ts:
        data               += [[delta_t]]
        delta_t_index       = int(np.round(delta_t / dt))
        res_index           = 0
        res_index_interval  = 100
        D                   = []
        while True:
            com             = hdf5_file.load(source, loader = _load_com_water, ignore_index = ignore_index,
                              start_index = res_index, end_index = res_index + res_index_interval)
            if com is None:   break
            delta_xyz_sq    = np.sum((com[delta_t_index:] - com[:-delta_t_index]) ** 2, axis = 2)
            D              += list(np.mean(delta_xyz_sq, axis = 0) / (6 * delta_t) / 1000.0)
            mean            = np.mean(D)
            se              = np.std(D) / np.sqrt(float(len(D)))
            print len(D), mean, se
            res_index      += res_index_interval
        data[-1]           += [np.mean(D), np.std(D) / np.sqrt(float(len(D)))]

    # Organize and return data
    data    = np.array([tuple(d) for d in data],
              np.dtype([("delta t", "f4"), ("WAT D ({0})".format(len(D)), "f4"), ("WAT D se", "f4")]))
    attrs   = {"time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1])),
               "delta t units": "ns", "D units": "A2 ps-1"}
    if verbose: _print_water(data, attrs)
    return  [(destination, data),
             (destination, attrs)]
    return []
def _check_water(hdf5_file, force = False, **kwargs):
    def _ignore_index(time, ignore):
        if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
        elif ignore == 0:   return 0
        elif ignore >  0:   return np.where(time > ignore)[0][0]

    # Parse kwargs and set defaults
    source                              = kwargs.get("source",      "*/com_water_unwrap")
    kwargs["source"]      = source      = source if source.startswith("*/") else "*/" + source
    ignore                              = kwargs.pop("ignore",      0)
    kwargs["delta_ts"]    = delta_ts    = kwargs.get("delta_ts",    np.array([0.1]))
    kwargs["destination"] = destination = kwargs.get("destination", "/diffusion/translation_water")

    # If analysis has not been run previously, run analysis
    if     (force
    or not (destination in hdf5_file)):
        log                                   = hdf5_file.load("*/log", type = "table")
        kwargs["ignore_index"] = ignore_index = _ignore_index(log["time"], ignore)
        kwargs["time"]                        = log["time"][ignore_index:]
        return [(water, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    data            = hdf5_file[destination]
    attrs           = hdf5_file.attrs(destination)
    log             = hdf5_file.load("*/log", type = "table")
    ignore_index    = _ignore_index(log["time"], ignore)
    kwargs["time"]  = log["time"][ignore_index:]
    if (np.any(data["delta t"] != np.array(delta_ts, np.float32))
    or (attrs["time"]          != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        return [(water, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False):    _print_water(data, attrs)
    return False
def _print_water(data, attrs):
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    for name in data.dtype.names: print "{0:<12}".format(name),
    print
    for D in data:
        for a in D: print "{0:<12.8f}".format(float(a)),
        print 

