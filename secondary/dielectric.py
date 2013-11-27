#!/usr/bin/python
desc = """MD_toolkit.secondary.dielectric.py
    Functions for calculation of the static dielectric constant
    Written by Karl Debiec on 13-11-20
    Last updated by Karl Debiec on 13-11-25"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   MD_toolkit.standard_functions import ignore_index, fit_curve
###################################################### FUNCTIONS #######################################################
def block(data, func, min_size = 3):
    full_size   = data.shape[0]
    sizes       = [s for s in list(set([full_size / s for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    sds         = np.zeros(sizes.size)
    n_blocks    = full_size // sizes
    for i, size in enumerate(sizes):
        resized = np.resize(data, (full_size // size, size, 3))
        map(func, resized)
        values  = map(func, resized)
        sds[i]  = np.std(values)
    ses                 = sds / np.sqrt(n_blocks - 1.0)
    se_sds              = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
    se_sds[se_sds == 0] = se_sds[np.where(se_sds == 0)[0] + 1]
    return sizes, ses, se_sds
################################################## ANALYSIS FUNCTIONS ##################################################
def dielectric(hdf5_file,
        time,                                               # Simulation time
        dipole,                                             # Overall dipole moment vector of system or selection(e A)
        temperature,                                        # System temperature (K)
        volume,                                             # System volume (A^3)
        destination,                                        # Destination of secondary data
        verbose = False, n_cores = 1, **kwargs):
    def calc_epsilon(dipole):
        boltzmann           = 1.3806488e-23
        vacuum_permittivity = 8.854187817620e-12
        dipole_variance = np.mean(np.sum(dipole * dipole, axis = 1)) - np.dot(np.mean(dipole, axis = 0),
                                                                              np.mean(dipole, axis = 0))
        return            1 + dipole_variance / (3.0 * vacuum_permittivity * volume * boltzmann * temperature)

    # Calculate static dielectric constant
    dipole                  *= 1.6021761206380539e-29                           # e A -> C m
    volume                  *= 1e-30                                            # A^3 -> m^3
    dielectric               = calc_epsilon(dipole)
    sizes, ses, se_sds       = block(data = dipole, func = calc_epsilon)
    dielectric_se, b, c, fit = fit_curve(x = np.array(sizes, np.float), y = ses, sigma = se_sds,
                                         fit_func = "single_exponential", p0 =(1.0, -1.0, -1.0))
    # Organize, print, and return data
    data    = np.array([(dielectric, dielectric_se)], [("dielectric", "f4"), ("dielectric se", "f4")])
    attrs   = {"time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1])),
               "volume": volume / 1e-30, "temperature": temperature}
    if verbose: _print_dielectric(data, attrs)
    return  [(destination, data),
             (destination, attrs)]

def _check_dielectric(hdf5_file, force = False, **kwargs):
    # Water molecule volumes for various water models; should probably be recalculated for each simulation protocol
    water_V = {"SPCE":      (56.72962952 ** 3.0) / 6093, "TIP3P":     (56.73929214 ** 3.0) / 6017,
               "TIPS3P":    (56.76272202 ** 3.0) / 6209, "TIP4P":     (56.67108536 ** 3.0) / 6042,
               "TIP4P2005": (56.69507980 ** 3.0) / 6071, "TIP4PEW":   (56.70848083 ** 3.0) / 6063}
    
    # Parse kwargs and set defaults
    source                              = kwargs.get("source",      "*/dipole")
    source                              = source if source.startswith("*/") else "*/" + source
    ignore                              = kwargs.pop("ignore",      0)
    kwargs["destination"] = destination = kwargs.get("destination", "dielectric")
    kwargs["temperature"] = temperature = kwargs.get("temperature", 298.0)
    if "volume" in kwargs:
        kwargs["volume"]  = volume      = kwargs["volume"]
    elif "n_waters" in kwargs and "solvent" in kwargs:
        kwargs["volume"]  = volume      = kwargs["n_waters"] * water_V[kwargs["solvent"]]
    else:
        kwargs["volume"]  = volume      = None

    # If analysis has not been run previously, run analysis
    if     (force
    or not (destination in hdf5_file)):
        log              = hdf5_file.load("*/log", type = "table")
        index            = ignore_index(log["time"], ignore)
        kwargs["time"]   = log["time"][index:]
        kwargs["dipole"] = np.array(hdf5_file.load(source)[index:], np.float64)
        if kwargs["volume"] == None: kwargs["volume"] = volume = np.mean(log["volume"][index:])
        return [(dielectric, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    data           = hdf5_file[destination]
    attrs          = hdf5_file.attrs(destination)
    log            = hdf5_file.load("*/log", type = "table")
    index          = ignore_index(log["time"], ignore)
    kwargs["time"] = log["time"][index:]
    if kwargs["volume"] == None: kwargs["volume"] = volume = np.mean(log["volume"][index:])
    if ((attrs["temperature"] != temperature)
    or  (attrs["volume"]      != volume)
    or  (attrs["time"]        != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        kwargs["dipole"] = np.array(hdf5_file.load(source)[index:], np.float64)
        return [(dielectric, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False): _print_dielectric(data, attrs)
    return False
def _print_dielectric(data, attrs):
    print "TIME            {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION        {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    print "DIELECTRIC (SE) {0:6.2f} ({1:4.2f})".format(float(data["dielectric"]), float(data["dielectric se"]))

