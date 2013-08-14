#!/usr/bin/python
desc = """energy.py
    Functions for secondary analysis of energy
    Written by Karl Debiec on 13-05-06
    Last updated 13-08-14"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import numpy as np
from   multiprocessing import Pool
from   scipy.stats     import linregress
from   scipy.optimize  import fmin, curve_fit
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

    m_ses       = m_sds / np.sqrt(n_blocks - 1.0)
    b_ses       = b_sds / np.sqrt(n_blocks - 1.0)
    m_se_sds    = np.sqrt((2.0) / (n_blocks - 1.0)) * m_ses
    b_se_sds    = np.sqrt((2.0) / (n_blocks - 1.0)) * b_ses
    m_fit       = _fit_curve(x = sizes, y = m_ses, sigma = m_se_sds, **kwargs["slope"])
    b_fit       = _fit_curve(x = sizes, y = b_ses, sigma = m_se_sds, **kwargs["int"])
    return m_fit[0], b_fit[0], sizes, m_ses, m_se_sds, m_fit[-1], b_ses, b_se_sds, b_fit[-1]
def _block(data, func, func_kwargs = {}, min_size = 1, **kwargs):
    full_size   = data.size
    sizes       = [s for s in list(set([full_size / s for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    sds         = np.zeros(sizes.size)
    n_blocks    = full_size // sizes
    for i, size in enumerate(sizes):
        resized = np.resize(data, (full_size // size, size))
        values  = func(resized, **func_kwargs)
        sds[i]  = np.std(values)
    ses                 = sds / np.sqrt(n_blocks - 1.0)
    se_sds              = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
    se_sds[se_sds == 0] = se_sds[np.where(se_sds == 0)[0] + 1]
    fit                 = _fit_curve(x = sizes, y = ses, sigma = se_sds, **kwargs)
    return fit[0], sizes, ses, se_sds, fit[-1]
def _fit_curve(fit_func, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        def single_exponential(x, y, **kwargs):
            def func(x, a, b, c):       return a + b * np.exp(c * x)
            a, b, c         = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, func(x, a, b, c)
        def double_exponential(x, y, **kwargs):
            def func(x, a, b, c, d, e): return a + b * np.exp(c * x) + d * np.exp(e * x)
            a, b, c, d, e   = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, e, func(x, a, b, c, d, e)
        return locals()[fit_func](**kwargs)
def _load_duration(self, path, **kwargs):
    segments    = self._segments()
    durations   = np.zeros(len(segments))
    for i, segment in enumerate(segments):
        duration        = self.attrs(segment)["duration"].split(":")
        durations[i]    = (int(duration[0]) * 60 * 60) + (int(duration[1]) * 60) + int(duration[2])
    return durations
################################################## ANALYSIS FUNCTIONS ##################################################
def conservation(hdf5_file,
                 block_settings = {},                       # Kwargs to be used for blocking; override defaults
                 check_plot     = True,                     # Show plot of blocking results before storing
                 verbose        = False, n_cores = 1, **kwargs):
    """ Calculates energy drift using linear regression. Error is estimated using the blocking method of Flyvbjerg, H.,
        and Petersen, H. G. Error Estimates on Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466. """
    time            = hdf5_file.data["*/log"]["time"]
    energy          = hdf5_file.data["*/log"]["total"]
    temperature     = hdf5_file.data["*/log"]["temperature"]
    block_kwargs    = {"E_mean": {"min_size": 1,
                                  "fit_func": "single_exponential", "p0": (  1.00,   -1.00, -0.10), "maxfev":10000},
                       "E_var":  {"min_size": 1,
                                  "fit_func": "single_exponential", "p0": (100.00, -100.00, -0.10), "maxfev":10000},
                       "E_line": {"min_size": 100,
                        "slope": {"fit_func": "single_exponential", "p0": (  1.00,  100.00, -0.01), "maxfev": 10000},
                        "int":   {"fit_func": "single_exponential", "p0": (100.00, 1000.00, -0.01), "maxfev": 10000}},
                       "T_mean": {"min_size": 1,
                                  "fit_func": "single_exponential", "p0": (  1.00,   -1.00, -1.00), "maxfev": 10000},
                       "T_var":  {"min_size": 10,
                                  "fit_func": "single_exponential", "p0": (  0.01,   -0.10, -0.10), "maxfev": 10000},
                       "T_line": {"min_size": 100,
                        "slope": {"fit_func": "single_exponential", "p0": (  0.01,    1.00, -0.01), "maxfev": 10000},
                        "int":   {"fit_func": "single_exponential", "p0": (  1.00,   10.00, -0.01), "maxfev": 10000}}}
    block_kwargs.update(block_settings)
    attrs   = {"energy slope units":      "kcal mol-1 ns-1", "energy units":       "kcal mol-1",
               "temperature slope units": "K ns -1",         "temperature units":  "K",
               "time":                    time[-1]}

    E_mean, E_var               = np.mean(energy),      np.var(energy)
    T_mean, T_var               = np.mean(temperature), np.var(temperature)
    E_slope, E_int, E_R, _, _   = linregress(time, energy)
    T_slope, T_int, T_R, _, _   = linregress(time, temperature)

    E_mean_block    = _block(data = energy,      func = np.mean, func_kwargs = {"axis": 1}, **block_kwargs["E_mean"])
    E_var_block     = _block(data = energy,      func = np.var,  func_kwargs = {"axis": 1}, **block_kwargs["E_var"])
    T_mean_block    = _block(data = temperature, func = np.mean, func_kwargs = {"axis": 1}, **block_kwargs["T_mean"])
    T_var_block     = _block(data = temperature, func = np.var,  func_kwargs = {"axis": 1}, **block_kwargs["T_var"])
    E_line_block    = _block_linregress(x = time, y = energy,      n_cores = n_cores,       **block_kwargs["E_line"])
    T_line_block    = _block_linregress(x = time, y = temperature, n_cores = n_cores,       **block_kwargs["E_line"])

    data    = [E_mean, E_mean_block[0], E_var, E_var_block[0], E_slope, E_line_block[0], E_int, E_line_block[1], E_R**2,
               T_mean, T_mean_block[0], T_var, T_var_block[0], T_slope, T_line_block[0], T_int, T_line_block[1], T_R**2]
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
    if check_plot:
        if not _plot_conservation(E_mean_block ,E_var_block, E_line_block, T_mean_block, T_var_block, T_line_block):
            return None
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
def _plot_conservation(E_mean_block, E_var_block, E_line_block, T_mean_block, T_var_block, T_line_block):
    import matplotlib.pyplot as plt
    def plot(axes, title, sizes, ses, se_sds, ses_fit):
        axes.set_title(title)
        axes.fill_between(np.log10(sizes), ses-1.96*se_sds, ses+1.96*se_sds, color = "blue", alpha = 0.5, lw = 0)
        axes.plot(        np.log10(sizes), ses,                              color = "blue")
        axes.plot(        np.log10(sizes), ses_fit,                          color = "black")
    figure  = plt.figure(figsize = (11, 8.5))
    figure.subplots_adjust(left = 0.05, right = 0.975, bottom = 0.05, top = 0.95, hspace = 0.3, wspace = 0.15)
    axes    = dict((i, figure.add_subplot(4, 2, i)) for i in xrange(1, 9))
    plot(axes[1], "Energy Mean",           *E_mean_block[1:])
    plot(axes[2], "Energy Variance",       *E_var_block[1:])
    plot(axes[3], "Energy Slope",          *E_line_block[2:6])
    plot(axes[4], "Energy Intercept",       E_line_block[2], *E_line_block[6:])
    plot(axes[5], "Temperature Mean",      *T_mean_block[1:])
    plot(axes[6], "Temperature Variance",  *T_var_block[1:])
    plot(axes[7], "Temperature Slope",     *T_line_block[2:6])
    plot(axes[8], "Temperature Intercept",  T_line_block[2], *T_line_block[6:])
    plt.show()
    if raw_input("store data in hdf5_file? (Y/n):").lower() in ["n", "no"]: return False
    else:                                                                   return True


