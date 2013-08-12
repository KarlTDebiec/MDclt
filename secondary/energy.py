#!/usr/bin/python
desc = """energy.py
    Functions for secondary analysis of energy
    Written by Karl Debiec on 13-05-06
    Last updated 13-08-10"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import numpy as np
from   multiprocessing import Pool
from   scipy.stats     import linregress
from   scipy.optimize  import curve_fit
################################################## INTERNAL FUNCTIONS ##################################################
def _subblock(arguments):
    i, size, n_blocks, x, y  = arguments
    x_i         = np.resize(x, (n_blocks, size))
    y_i         = np.resize(y, (n_blocks, size))
    m_values    = np.zeros(n_blocks)
    b_values    = np.zeros(n_blocks)
    for j, x_j, y_j in zip(xrange(n_blocks), x_i, y_i):
        m_values[j], b_values[j], _, _, _   = linregress(x_j, y_j)
    return i, np.std(m_values), np.std(b_values)
def _block_linregress(x, y, min_size = 10, n_cores = 1, **kwargs):
    full_size   = x.size
    sizes       = [s for s in list(set([full_size / s for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    n_blocks    = full_size // sizes
    m_sds       = np.zeros(sizes.size)
    b_sds       = np.zeros(sizes.size)

    arguments   = [(i, sizes[i], n_blocks[i], x, y) for i in xrange(sizes.size)]
    pool        = Pool(n_cores)
    for result in pool.imap_unordered(_subblock, arguments):
        i, m_sd_i, b_sd_i   = result
        m_sds[i]            = m_sd_i
        b_sds[i]            = b_sd_i
    pool.close()
    pool.join()

    m_ses           = m_sds / np.sqrt(n_blocks)
    b_ses           = b_sds / np.sqrt(n_blocks)
    return sizes, m_ses, b_ses
def _block(data, function, **kwargs):
    full_size       = data.size
    sizes           = np.array(sorted(list(set([full_size / x for x in range(1, full_size)]))), np.int)[:-1]
    sds             = np.zeros(sizes.size)
    n_blocks        = full_size // sizes
    for i, size in enumerate(sizes):
        resized     = np.resize(data, (full_size // size, size))
        values      = function(resized, **kwargs)
        sds[i]      = np.std(values)
    ses             = sds / np.sqrt(n_blocks)
    return sizes, ses
def _fit_sigmoid(x, y):
    def func(x, min_asym, max_asym, poi, k): return max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_asym, max_asym, poi, k  = curve_fit(func, x, y, maxfev = 100000)[0]
    y_fit   = func(x, min_asym, max_asym, poi, k)
    return min_asym, max_asym, poi, k, y_fit
def _check_positive(x):
    if float(x) < 0:    raise Exception("x is not a positive number")
    else:               return x
def _load_duration(self, path, **kwargs):
    segments    = self._segments()
    durations   = np.zeros(len(segments))
    for i, segment in enumerate(segments):
        duration        = self.attrs(segment)["duration"].split(":")
        durations[i]    = (int(duration[0]) * 60 * 60) + (int(duration[1]) * 60) + int(duration[2])
    return durations
################################################## ANALYSIS FUNCTIONS ##################################################
def conservation(hdf5_file, verbose = False, n_cores = 1, **kwargs):
    """ Calculates energy drift using linear regression. Error is estimated using the blocking method of Flyvbjerg, H.,
        and Petersen, H. G. Error Estimates on Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466. """
    time        = hdf5_file.data["*/log"]["time"]
    energy      = hdf5_file.data["*/log"]["total"]
    temperature = hdf5_file.data["*/log"]["temperature"]
    attrs       = {"energy slope units":      "kcal mol-1 ns-1", "energy units":       "kcal mol-1",
                   "temperature slope units": "K ns -1",         "temperature units":  "K",
                   "time":                    time[-1]}

    E_mean                                  = np.mean(energy)
    try:    E_mean_se                       = _check_positive(_fit_sigmoid(*_block(energy, np.mean, axis = 1))[1])
    except: E_mean_se,     attrs["note"]    = np.nan, "Standard error calculation failed for one or more parameters"
    E_variance                              = np.var(energy)
    try:    E_variance_se                   = _check_positive(_fit_sigmoid(*_block(energy, np.var,  axis = 1))[1])
    except: E_variance_se, attrs["note"]    = np.nan, "Standard error calculation failed for one or more parameters"
    E_slope, E_intercept, E_R, _, _         = linregress(time, energy)
    E_times, E_slope_ses, E_intercept_ses   = _block_linregress(time, energy, n_cores = n_cores)
    try:    E_slope_se                      = _check_positive(_fit_sigmoid(E_times, E_slope_ses)[1])
    except: E_slope_se,     attrs["note"]   = np.nan, "Standard error calculation failed for one or more parameters"
    try:    E_intercept_se                  = _check_positive(_fit_sigmoid(E_times, E_intercept_ses)[1])
    except: E_intercept_se, attrs["note"]   = np.nan, "Standard error calculation failed for one or more parameters"

    T_mean                                  = np.mean(temperature)
    try:    T_mean_se                       = _check_positive(_fit_sigmoid(*_block(temperature, np.mean, axis = 1))[1])
    except: T_mean_se,      attrs["note"]   = np.nan, "Standard error calculation failed for one or more parameters"
    T_variance                              = np.var(temperature)
    try:    T_variance_se                   = _check_positive(_fit_sigmoid(*_block(temperature, np.var,  axis = 1))[1])
    except: T_variance_se,  attrs["note"]   = np.nan, "Standard error calculation failed for one or more parameters"
    T_slope, T_intercept, T_R, _, _         = linregress(time, temperature)
    T_times, T_slope_ses, T_intercept_ses   = _block_linregress(time, temperature, n_cores = n_cores)
    try:    T_slope_se                      = _check_positive(_fit_sigmoid(T_times, T_slope_ses)[1])
    except: T_slope_se,     attrs["note"]   = np.nan, "Standard error calculation failed for one or more parameters"
    try:    T_intercept_se                  = _check_positive(_fit_sigmoid(T_times, T_intercept_ses)[1])
    except: T_intercept_se, attrs["note"]   = np.nan, "Standard error calculation failed for one or more parameters"

    data    = [E_mean, E_mean_se, E_variance, E_variance_se, E_slope, E_slope_se, E_intercept, E_intercept_se, E_R ** 2, 
               T_mean, T_mean_se, T_variance, T_variance_se, T_slope, T_slope_se, T_intercept, T_intercept_se, T_R ** 2]
    dtype   = [("energy mean",           "f4"), ("energy mean se",           "f4"),
               ("energy variance",       "f4"), ("energy variance se",       "f4"),
               ("energy slope",          "f4"), ("energy slope se",          "f4"),
               ("energy intercept",      "f4"), ("energy intercept se",      "f4"),
               ("energy R2",             "f4"),
               ("temperature mean",      "f4"), ("temperature mean se",      "f4"),
               ("temperature variance",  "f4"), ("temperature variance se",  "f4"),
               ("temperature slope",     "f4"), ("temperature slope se",     "f4"),
               ("temperature intercept", "f4"), ("temperature intercept se", "f4"),
               ("temperature R2",        "f4")]

    if "*/duration" in hdf5_file.data:
        data   += [time[-1] / np.sum(hdf5_file.data["*/duration"]) * 86400.0]
        dtype  += [("efficiency", "f4")]
        attrs["efficiency units"] = "ns day-1"

    data                    = np.array(tuple(data), np.dtype(dtype))
    kwargs["data_kwargs"]   = {"chunks": False}             # h5py cannot chunk record array of length 1
    if verbose:               _print_conservation(data, attrs)
    return  [("energy/conservation", data, kwargs),
             ("energy/conservation", attrs)]
def _check_conservation(hdf5_file, force = False, **kwargs):
    verbose = kwargs.get("verbose",     False)
    hdf5_file.load("*/log", type = "table")
    try:    hdf5_file.load("*/duration", loader = _load_duration)
    except: pass

    if    (force
    or not "/energy/conservation" in hdf5_file):
        return [(conservation, kwargs)]

    attrs   = hdf5_file.attrs("energy/conservation")
    if hdf5_file.data["*/log"]["time"][-1] != attrs["time"]:
        return [(conservation, kwargs)]
    elif verbose:
        data    = hdf5_file["energy/conservation"]
        _print_conservation(data, attrs)
    return False
def _print_conservation(data, attrs):
    if ("('efficiency', '<f4')" in str(data.dtype)):
            print "Duration {0:d} ns at {1:1.1f} ns/day".format(int(attrs["time"]), float(data["efficiency"]))
    else:   print "Duration {0:d} ns".format(int(attrs["time"]))
    print "               Energy                                Temperature"
    print "Mean (se)      {0:12.4f} ({1:9.4f}) kcal mol-1      {2:8.4f} ({3:8.4f}) K".format(
           float(data["energy mean"]),              float(data["energy mean se"]),
           float(data["temperature mean"]),         float(data["temperature mean se"]))
    print "Variance (se)  {0:12.4f} ({1:9.4f}) kcal mol-1      {2:8.4f} ({3:8.4f}) K".format(
           float(data["energy variance"]),          float(data["energy variance se"]),
           float(data["temperature variance"]),     float(data["temperature variance se"]))
    print "Slope (se)     {0:12.4f} ({1:9.4f}) kcal mol-1 ns-1 {2:8.4f} ({3:8.4f}) K ns-1".format(
           float(data["energy slope"]),             float(data["energy slope se"]),
           float(data["temperature slope"]),        float(data["temperature slope se"]))
    print "Intercept (se) {0:12.4f} ({1:9.4f}) kcal mol-1      {2:8.4f} ({3:8.4f}) K".format(
           float(data["energy intercept"]),         float(data["energy intercept se"]),
           float(data["temperature intercept"]),    float(data["temperature intercept se"]))
    print "R^2            {0:12.4f}                             {1:8.4f}".format(
           float(data["energy R2"]), float(data["temperature R2"]))

