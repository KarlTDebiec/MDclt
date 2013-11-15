#!/usr/bin/python
desc = """MD_toolkit.secondary.diffusion.translation.__init__.py
    Functions for secondary analysis of diffusion
    Written by Karl Debiec on 13-02-08
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import numpy as np
from   scipy.optimize    import curve_fit
################################################## ANALYSIS FUNCTIONS ##################################################
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

def translation(hdf5_file,
                destination     = "",                       # Origin of primary data and destination of secondary data
                delta_t         = np.array([0.1]),          # Delta_t values to test (ns)
                selection       = [],                       # Selection names 
                explicit_names  = False,                    # Explicitly name selections in output table
                control         = np.nan,                   # Control value for comparison (if verbose)
                verbose         = False, n_cores = 1, **kwargs):
    """ Calculates the translational diffusion coefficient of <selection>(s) over intervals of <delta_t>. Stores in a
        table at <destination> numbered either as 0, 1, 2, ... or with explicit selection strings if <explicit_names>.
        Follows the protocol of McGuffee, S. R., and Elcock, A. H. PLoS Comp Bio. 2010. e1000694. Error is estimated
        using the blocking method of Flyvbjerg, H., and Petersen, H. G. Error Estimates on Averages of Correlated Data.
        J Phys Chem. 1989. 91. 461-466."""
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination
    time            = hdf5_file.data["*/log"]["time"]
    com             = hdf5_file.data["*/com" + destination]
    size            = time.size
    dt              = time[1] - time[0]
    n_selections    = com.shape[1]
    selection_attr  = " ".join(["\"{0}\"".format(sel) for sel in selection])
    D_table         = []
    dtype           = [("delta t", "f4")]
    for i, selection in enumerate(selection):
        if explicit_names:  dtype  += [("{0} D".format(selection), "f4"), ("{0} D se".format(selection), "f4")]
        else:               dtype  += [("{0} D".format(selection), "f4"), ("{0} D se".format(selection), "f4")]
    attrs   = {"time": time[-1], "delta t units": "ns", "D units": "A2 ps-1", "selection": selection_attr}

    def calc_D_block(dr_2, delta_t): return  np.mean(dr_2, axis = 1) / (6 * delta_t)

    for delta_t in delta_t:
        D_table        += [[delta_t]]
        delta_t_index   = int(np.round(delta_t / dt))
        dr_2            = np.sum((com[delta_t_index:] - com[:-delta_t_index]) ** 2, axis = 2)
        for j in xrange(dr_2.shape[1]):
            x, y        = _block(dr_2[:,j], calc_D_block, delta_t = delta_t)
            try:
                _, max_asym, _, _, _    = _fit_sigmoid(x, y)
            except:
                max_asym                = np.nan
                attrs["note"]           = "Standard error calculation failed for one or more delta t"
            D_table[-1]    += [np.mean(dr_2[:,j], axis = 0) / (6 * delta_t) / 1000.0, max_asym / 1000.0]
    for i in xrange(len(D_table)):
        D_table[i]          = tuple(D_table[i])
    D_table                 = np.array(D_table, np.dtype(dtype))

    if verbose:     _print_translation(D_table, attrs)
    return  [("diffusion/translation" + destination, D_table),
             ("diffusion/translation" + destination, attrs)]
def _check_translation(hdf5_file, force = False, **kwargs):
    destination = kwargs.get("destination", "")
    delta_ts    = kwargs.get("delta_t",     np.array([0.1]))
    verbose     = kwargs.get("verbose",     False)
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination
    selection           = hdf5_file.attrs("0000/com" + destination)["selection"].split("\" \"")
    selection           = [sel.strip("\"") for sel in selection]
    kwargs["selection"] = selection

    hdf5_file.load("*/log", type = "table")
    hdf5_file.load("*/com" + destination)

    if (force
    or not("diffusion/translation" + destination in hdf5_file)):
        return [(translation, kwargs)]

    data    = hdf5_file["diffusion/translation" + destination]
    attrs   = hdf5_file.attrs("diffusion/translation" + destination)

    if (np.any(np.array(delta_ts, np.float32) != data["delta t"])
    or (hdf5_file.data["*/log"]["time"][-1]   != attrs["time"])):
        return [(translation, kwargs)]
    elif verbose:   _print_translation(data, attrs)
    return False
def _print_translation(data, attrs):
    selection   = [sel.strip("\"") for sel in attrs["selection"].split("\" \"")]
    print "DURATION {0:5d} ns".format(int(attrs["time"]))
    print "DELTA_T  ",
    for sel in selection:
        print "{0:<9}{1:<9}".format(sel, "(se)"),
    print
    for line in data:
        print "{0:<9.3f}".format(float(line[0])),
        for i, sel in enumerate(selection):
            print "{0:<9.5f}({1:7.5f})".format(float(line[i*2+1]), float(line[i*2+2])),
        print

