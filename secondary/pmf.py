#!/usr/bin/python
desc = """MD_toolkit.secondary.pmf.py
    Calculation of potential of mean force (PMF)
    Written by Karl Debiec on 12-08-15
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, types
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def pmf_1D(hdf5_file,
        time,                                               # Simulation time
        count,                                              # Counts in each bin
        bins,                                               # Bins
        zero_point,                                         # Point or range at which to zero energy
        temperature,                                        # System temperature
        destination,                                        # Analysis output destination within HDF5 file
        verbose = False, n_cores = 1, **kwargs):
    """ Calculates potential of mean force along progress coordinate split into <bins> at <temperature>. Sets energy to
        zero at <zero point>, or to average of zero over <zero point> range in form of 'start:end'. """

    # Calculate free energy and PMF
    centers                         = (bins[:-1] + bins[1:]) / 2.0
    probability                     = np.array(count, dtype = np.float32) / np.sum(count)
    probability[probability == 0.0] = np.nan
    free_energy                     = -1.0 * np.log(probability)
    pmf                             = probability / (centers ** 2.0)
    pmf                            /= np.nansum(pmf)
    pmf                             = -1.0 * np.log(pmf) * 0.0019872041 * temperature

    # Zero PMF at selected point or over selected range
    if zero_point:
        if isinstance(zero_point, types.StringTypes):
            if   ":" in zero_point:     zero_start, zero_end = zero_point.split(":")
            elif "-" in zero_point:     zero_start, zero_end = zero_point.split("-")
            else:                       zero_start, zero_end = zero_point.split()
            zero_start      = np.abs(bins - float(zero_start)).argmin()
            zero_end        = np.abs(bins - float(zero_end)).argmin()
            value_at_zero   = np.mean(np.ma.MaskedArray(pmf[zero_start:zero_end], np.isnan(pmf[zero_start:zero_end])))
        else:
            value_at_zero   = pmf[np.abs(centers - zero_point).argmin()]
        pmf                -= value_at_zero
    else:
        zero_point          = "None"

    # Organize and return data
    data    = np.array([tuple(frame) for frame in zip(bins[:-1], bins[1:], count, probability, free_energy, pmf)],
                       np.dtype([("lower bound", "f4"), ("upper bound", "f4"), ("count", "i4"),
                                 ("probability", "f4"), ("free energy", "f4"), ("pmf",   "f4")]))
    attrs   = {"lower bound units": "A", "upper bound units": "A", "free energy units": "kBT",
               "pmf units": "kcal mol-1", "temperature": temperature, "zero_point": zero_point,
               "time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1]))}
    if verbose: _print_pmf_1D(destination, data, attrs)
    return  [(destination, data),
             (destination, attrs)]

def _check_pmf_1D(hdf5_file, force = False, **kwargs):
    """ Determines whether or not to run 'association.pmf' based on settings and data present in hdf5 file, and loads
        necessary primary data. 'source' and destination' may be set manually, but may also be guessed from pcoord. """
    def _ignore_index(time, ignore):
        if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
        elif ignore == 0:   return 0
        elif ignore >  0:   return np.where(time > ignore)[0][0]
    def _load_association(self, path, bins, ignore_index, closest = False, **kwargs):
        data            = np.zeros(bins.size - 1, np.int32)
        for segment in self._segments():
            new_data    = self[segment + "/" + path[2:]]
            if  ignore_index  > new_data.shape[0]:
                ignore_index -= new_data.shape[0]
                continue
            elif ignore_index > 0:
                new_data        = new_data[ignore_index:]
                ignore_index    = 0
            if closest:
                new_data    = np.min(new_data, axis = 2)
            hist, _         = np.histogram(new_data, bins)
            data           += hist
        return data

    # Parse kwargs and set defaults
    source                              = kwargs.pop("source",      "*/association_mindist")
    source                              = source if source.startswith("*/") else "*/" + source
    ignore                              = kwargs.pop("ignore",      0)
    closest                             = kwargs.get("closest",     False)
    kwargs["bins"]        = bins        = kwargs.get("bins",        np.linspace(0, 100, 101))
    kwargs["zero_point"]  = zero_point  = kwargs.get("zero_point",  np.nan)
    kwargs["temperature"] = temperature = kwargs.get("temperature", 298.0)
    kwargs["destination"] = destination = kwargs.get("destination", "/association/mindist/pmf")

    # If analysis has not been run previously, run analysis
    if     (force
    or not (destination in hdf5_file)):
        log             = hdf5_file.load("*/log", type = "table")
        ignore_index    = _ignore_index(log["time"], ignore)
        kwargs["time"]  = log["time"][ignore_index:]
        kwargs["count"] = hdf5_file.load(source, loader = _load_association,
                                         bins = bins, ignore_index = ignore_index, closest = closest)
        return [(pmf_1D, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    data            = hdf5_file[destination]
    attrs           = hdf5_file.attrs(destination)
    log             = hdf5_file.load("*/log", type = "table")
    ignore_index    = _ignore_index(log["time"], ignore)
    kwargs["time"]  = log["time"][ignore_index:]
    if (attrs["temperature"]         != temperature
    or (attrs["zero_point"]          != zero_point)
    or (data["lower bound"].size + 1 != bins.size)
    or (np.any(data["lower bound"]   != np.array(bins[:-1], np.float32)))
    or (np.any(data["upper bound"]   != np.array(bins[1:],  np.float32)))
    or (attrs["time"]                != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        kwargs["count"] = hdf5_file.load(source, loader = _load_association,
                                         bins = bins, ignore_index = ignore_index, closest = closest)
        return [(pmf_1D, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False): _print_pmf_1D(destination, data, attrs)
    return False

def _print_pmf_1D(destination, data, attrs):
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    print "PROGRESS COORDINATE {0} ".format(destination.upper())
    print "TEMPERAUTRE {0:6.3f} K ZERO POINT {1}".format(float(attrs["temperature"]), attrs["zero_point"])
    print "  LOWER  UPPER  COUNT        P      FE     PMF"
    for line in data:
        print "{0:>7.3f}{1:>7.3f}{2:>7.0f}{3:>9.6f}{4:>8.4f}{5:>8.4f}".format(*map(float, line))


