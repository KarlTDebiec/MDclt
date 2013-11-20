#!/usr/bin/python
desc = """MD_toolkit.secondary.dielectric.py
    Functions for calculation of the static dielectric constant
    Written by Karl Debiec on 13-11-17
    Last updated by Karl Debiec on 13-11-17"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def dielectric(hdf5_file,
        time,                                               # Simulation time
        dipole,                                             # Overall dipole moment vector of system (e A)
        temperature,                                        # System temperature (K)
        volume,                                             # System volume (m^3)
        destination,                                        # Destination of secondary data
        verbose = False, n_cores = 1, **kwargs):

    # Calculate static dielectric constant
    dipole_variance     = np.var(dipole * 1.6021761206380539e-29)               #   C^2     m^2
    boltzmann           = 1.3806488e-23                                         #       J       K-1
    vacuum_permittivity = 8.854187817620e-12                                    #   C^2 J-1 m-1
    epsilon             = dipole_variance / (3.0 * vacuum_permittivity * volume * boltzmann * temperature);
    print dipole
    print np.mean(dipole * 1.6021761206380539e-29)
    print dipole_variance
    print vacuum_permittivity, volume, boltzmann, temperature
    print epsilon
    # Organize and return data
#    data    = np.array([tuple(D) for D in data], np.dtype(dtype))
#    attrs   = {"time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1])),
#               "delta t units": "ns", "D units": "A2 ps-1"}
#    if verbose:     _print_dielectric(data, attrs)
#    return  [(destination, data),
#             (destination, attrs)]
    return []
def _check_dielectric(hdf5_file, force = False, **kwargs):
    def _ignore_index(time, ignore):
        if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
        elif ignore == 0:   return 0
        elif ignore >  0:   return np.where(time > ignore)[0][0]

    # Parse kwargs and set defaults
    source                              = kwargs.get("source",      "*/dipole")
    source                              = source if source.startswith("*/") else "*/" + source
    ignore                              = kwargs.pop("ignore",      0)
    kwargs["destination"] = destination = kwargs.get("destination", "/dielectric")
    kwargs["temperature"] = temperature = kwargs.get("temperature", 298.0)

    # If analysis has not been run previously, run analysis
    if     (force
    or not (destination in hdf5_file)):
        log              = hdf5_file.load("*/log", type = "table")
        ignore_index     = _ignore_index(log["time"], ignore)
        kwargs["time"]   = log["time"][ignore_index:]
        kwargs["volume"] = np.mean(log["volume"]) * 1e-30
        kwargs["dipole"] = np.array(hdf5_file.load(source)[ignore_index:], np.float64)
        return [(dielectric, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
#    data           = hdf5_file[destination]
#    attrs          = hdf5_file.attrs(destination)
#    log            = hdf5_file.load("*/log", type = "table")
#    ignore_index   = _ignore_index(log["time"], ignore)
#    kwargs["time"] = log["time"][ignore_index:]
#    if ((attrs["temperature"] != temperature)
#    or  (attrs["time"]        != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
#        kwargs["com"] = hdf5_file.load(source)[ignore_index:]
#        return [(dielectric, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
#    if kwargs.get("verbose", False):    _print_dielectric(data, attrs)
    return False
def _print_dielectric(data, attrs):
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
#    for name in data.dtype.names: print "{0:<12}".format(name),
#    print
#    for D in data:
#        for a in D: print "{0:<12.8f}".format(float(a)),
#        print 


