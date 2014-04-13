#!/usr/bin/python
#   MD_toolkit.secondary.diffusion.translation.__init__.py
#   Written by Karl Debiec on 13-02-08
#   Last updated by Karl Debiec on 14-04-04
################################################# MODULES AND SETTINGS #################################################
import os, sys
import numpy as np
from   MD_toolkit.standard_functions import block, ignore_index, fit_curve
################################################## ANALYSIS FUNCTIONS ##################################################
def translation(hdf5_file, time, com, delta_ts, name, selection, destination,
                verbose = False, debug = False, n_cores = 1, **kwargs):
    """
    Calculates the translational diffusion coefficient of one or more selections of atoms.

    **Arguments:**
        :*hdf5_file*:   HDF5_File object
        :*time*:        simulation time
        :*com*:         centers of mass of selections; dimensions = (time, selection, xyz)
        :*delta_ts*:    delta t(s) over which to calculate diffusion
        :*name*:        names of selections
        :*selection*:   resides ordescription of selections(not used for calculation, just copied for reference)
        :*destination*: path in h5 file at which to store output

    **Returns:**
        PENDING

    Follows the protocol of McGuffee, S. R., and Elcock, A. H. PLoS Comp Bio. 2010. e1000694.
    Standard error is estimated using the block averaging method of Flyvbjerg, H., and Petersen, H. G. Error Estimates
    on Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466.
    """

    # Initialize variables
    dt          = np.mean(time[1:] - time[:-1])
    dtype       = [("delta t", "f4")]
    for n in name.split("\n"):
        dtype  += [("{0} D".format(n), "f4"), ("{0} D se".format(n), "f4")]

    # Calculate translational diffusion
    data    = []
    for delta_t in delta_ts:
        def calc_D_block(delta_xyz_2):                                          # Re-define within loop because map()
            return  np.mean(delta_xyz_2, axis = 1) / (6 * delta_t) / 1000.0     # cannot pass delta_t variable
        data           += [[delta_t]]
        delta_t_index   = int(np.round(delta_t / dt))
        delta_xyz_2     = np.sum((com[delta_t_index:] - com[:-delta_t_index]) ** 2, axis = 2)
        D               = np.mean(delta_xyz_2, axis = 0) / (6 * delta_t) / 1000.0
        for i, n in enumerate(name.split("\n")):
            sizes, ses, se_sds    = block(delta_xyz_2[:,i], calc_D_block, min_size = 10)
            D_se, b, c, fit       = fit_curve(x = sizes, y = ses, sigma = se_sds,
                                              fit_func = "single_exponential", p0 =(1.0, -1.0, -1.0))
            data[-1]             += [D[i], D_se]

    # Organize and return data
    data    = np.array([tuple(D) for D in data], dtype)
    attrs   = {"time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1])),
               "delta t units": "ns",
               "D units":       "A2 ps-1",
               "selection":     selection}        
    if verbose:     _print_translation(data, attrs)
    return  [(destination, data),
             (destination, attrs)]

def _check_translation(hdf5_file, force = False, **kwargs):

    # Parse kwargs and set defaults
    source                              = kwargs.get("source",      "*/com_unwrap")
    source                              = source if source.startswith("*/") else "*/" + source
    ignore                              = kwargs.pop("ignore",      0)
    kwargs["delta_ts"]    = delta_ts    = kwargs.get("delta_ts",    np.array([0.1]))
    kwargs["destination"] = destination = kwargs.get("destination", "/diffusion/translation")
    kwargs["selection"]                 = hdf5_file.attrs("0000/" + source[2:])["selection"]        # Still not ideal,
    kwargs["name"]                      = hdf5_file.attrs("0000/" + source[2:])["name"]             # fix in future

    # If analysis has not been run previously, run analysis
    if     (force
    or not (destination in hdf5_file)):
        log             = hdf5_file.load("*/log", type = "table")
        index           = ignore_index(log["time"], ignore)
        kwargs["time"]  = log["time"][index:]
        kwargs["com"]   = hdf5_file.load(source)[index:]
        return [(translation, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    data            = hdf5_file[destination]
    attrs           = hdf5_file.attrs(destination)
    log             = hdf5_file.load("*/log", type = "table")
    index           = ignore_index(log["time"], ignore)
    kwargs["time"]  = log["time"][index:]
    if (np.any(data["delta t"] != np.array(delta_ts, np.float32))
    or (attrs["time"]          != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        kwargs["com"]   = hdf5_file.load(source)[index:]
        return [(translation, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False):    _print_translation(data, attrs)
    return False

def _print_translation(data, attrs):
    names = [n for n in data.dtype.names[1::2]]
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    print "DELTA T", 
    for name in names: print "{0:>16} (se)".format(name).upper(),
    print
    for D in data:
        print "{0:>7.3f}".format(float(D[0])),
        for name in names:
            print "{0:>21}".format("{0:>7.5f} ({1:>7.5f})".format(float(D[name]), float(D[name + " se"]))),
        print


