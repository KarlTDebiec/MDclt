#!/usr/bin/python
desc = """energy.py
    Functions for secondary analysis of energy
    Written by Karl Debiec on 13-05-06
    Last updated 13-08-02"""
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
def _block(x, y, min_size = 10, n_cores = 1, **kwargs):
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
def _fit_sigmoid(x, y):
    def func(x, min_asym, max_asym, poi, k): return max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_asym, max_asym, poi, k  = curve_fit(func, x, y, maxfev = 100000)[0]
    y_fit   = func(x, min_asym, max_asym, poi, k)
    return min_asym, max_asym, poi, k, y_fit
################################################## ANALYSIS FUNCTIONS ##################################################
def conservation(hdf5_file, verbose = False, n_cores = 1, **kwargs):
    """ Calculates energy drift using linear regression. Error is estimated using the blocking method of Flyvbjerg, H.,
        and Petersen, H. G. Error Estimates on Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466. """
    time        = hdf5_file.data["*/log"]["time"]
    energy      = hdf5_file.data["*/log"]["total"]
    temperature = hdf5_file.data["*/log"]["temperature"]
    attrs       = {"energy slope units":      "kcal mol-1 ns-1", "energy intercept units":       "kcal mol-1",
                   "temperature slope units": "K ns -1",         "temperature intercept units":  "K",
                   "time":                    time[-1]}

    E_m, E_b, E_R, _, _     = linregress(time, energy)
    E_ts, E_m_ses, E_b_ses  = _block(time, energy, n_cores = n_cores)
    try:
        E_m_se              = _fit_sigmoid(E_ts, E_m_ses)[1]
    except:
        E_m_se              = np.nan
        attrs["note"]       = "Standard error calculation failed for one or more parameters"
    try:
        E_b_se              = _fit_sigmoid(E_ts, E_b_ses)[1]
    except:
        E_m_se              = np.nan
        attrs["note"]       = "Standard error calculation failed for one or more parameters"
    T_m, T_b, T_R, _, _     = linregress(time, temperature)
    T_ts, T_m_ses, T_b_ses  = _block(time, temperature, n_cores = n_cores)
    try:
        T_m_se              = _fit_sigmoid(T_ts, T_m_ses)[1]
    except:
        T_m_se              = np.nan
        attrs["note"]       = "Standard error calculation failed for one or more parameters"
    try:
        T_b_se              = _fit_sigmoid(T_ts, T_b_ses)[1]
    except:
        T_b_se              = np.nan
        attrs["note"]       = "Standard error calculation failed for one or more parameters"

    dtype       = np.dtype([("energy slope",          "f4"), ("energy slope se",          "f4"),
                            ("energy intercept",      "f4"), ("energy intercept se",      "f4"),
                            ("energy R2",             "f4"), 
                            ("temperature slope",     "f4"), ("temperature slope se",     "f4"),
                            ("temperature intercept", "f4"), ("temperature intercept se", "f4"),
                            ("temperature R2",        "f4")])
    data        = np.array((E_m, E_m_se, E_b, E_b_se, E_R ** 2, T_m, T_m_se, T_b, T_b_se, T_R ** 2), dtype)
    kwargs["data_kwargs"]   = {"chunks": False}
    if verbose:     _print_conservation(data, attrs)
    return  [("energy/conservation", data, kwargs),
             ("energy/conservation", attrs)]
def _check_conservation(hdf5_file, force = False, **kwargs):
    verbose = kwargs.get("verbose",     False)
    hdf5_file.load("*/log", type = "table")

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
    print "Duration {0:6d} ns".format(int(attrs["time"]))
    print "              Energy                      Temperature"
    print "Slope        {0:12.4f} kcal mol-1 ns-1 {1:8.4f} K ns-1".format(float(data["energy slope"]),
                                                                       float(data["temperature slope"]))
    print "Slope se     {0:12.4f} kcal mol-1 ns-1 {1:8.4f} K ns-1".format(float(data["energy slope se"]),
                                                                       float(data["temperature slope se"]))
    print "Intercept    {0:12.4f} kcal mol-1      {1:8.4f} K".format(float(data["energy intercept"]),
                                                                  float(data["temperature intercept"]))
    print "Intercept se {0:12.4f} kcal mol-1      {1:8.4f} K".format(float(data["energy intercept se"]),
                                                                  float(data["temperature intercept se"]))
    print "R^2          {0:12.4f}                 {1:8.4f}".format(float(data["energy R2"]),
                                                                float(data["temperature R2"]))

