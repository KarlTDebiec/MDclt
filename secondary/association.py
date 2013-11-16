#!/usr/bin/python
desc = """MD_toolkit.secondary.association.py
    Functions for secondary analysis of molecular association
    Written by Karl Debiec on 12-08-15
    Last updated by Karl Debiec on 13-11-15"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, time, types, warnings
import numpy as np
from   scipy.optimize import fmin
from   MD_toolkit.standard_functions import block_average, fit_curve
################################################## INTERNAL FUNCTIONS ##################################################
def _concentration(n, volume): return float(n) / 6.0221415e23 / volume
def _P_bound_to_Ka(P_bound, C_mol1_total, C_mol2_total):
    C_complex       = P_bound * np.min(C_mol1_total, C_mol2_total)
    C_mol1_unbound  = C_mol1_total - C_complex
    C_mol2_unbound  = C_mol2_total - C_complex
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return            C_complex / (C_mol1_unbound * C_mol2_unbound)
def _P_bound_se_to_Ka_se(P_bound, C_mol1_total, C_mol2_total, P_bound_se):
    return np.sqrt((((C_mol2_total - C_mol1_total * P_bound ** 2) * P_bound_se) /
                    ((P_bound - 1) ** 2 * (C_mol1_total * P_bound - C_mol2_total) ** 2)) ** 2)
def _Ka_to_P_bound(Ka, C_mol1_total, C_mol2_total):
    def model_function(P_bound, Ka, C_mol1_total, C_mol2_total):
        C_min   = np.min(C_mol1_total, C_mol2_total)
        return (((P_bound * C_min)/((C_mol1_total - P_bound * C_min) * (C_mol2_total - P_bound * C_min))) - Ka) ** 2
    P_bound     = fmin(func = model_function, x0 = 0.1, args = (Ka, C_mol1_total, C_mol2_total), disp = False)
    return P_bound
################################################ PRIMARY DATA FUNCTIONS ################################################
def _load_association_distance_histogram(self, path, bins, **kwargs):
    segments            = self._segments()
    data                = None
    for segment in segments:
        seg_data        = self[segment + "/" + path[2:]]
        sorted_seg_data = np.zeros(seg_data.shape, np.float32)
        if data is None:
            data        = np.zeros((seg_data.shape[2], bins.size - 1), np.int32)
        for t in xrange(seg_data.shape[0]):
            for i in xrange(seg_data.shape[1]):
                sorted_seg_data[t,i] = seg_data[t, i, seg_data[t,i].argsort()]
        for j in xrange(sorted_seg_data.shape[2]):
            hist, _     = np.histogram(sorted_seg_data[:,:,j], bins)
            data[j]    += hist
    return data
################################################## ANALYSIS FUNCTIONS ##################################################
def rate(hdf5_file,
        time,                                               # Simulation time
        distance,                                           # Distance metric
        volume,                                             # System volume (L)
        bound_cutoff,                                       # Bound state cutoff (A)
        unbound_cutoff,                                     # Unbound state cutoff (A)
        n_molecule_1,                                       # Number of molecules of type 1
        n_molecule_2,                                       # Number of molecules of type 2
        destination,                                        # Analysis output destination within hdf5 file
        verbose = False, n_cores = 1, **kwargs):
    """ Calculates Ka, kon, and koff of two sets of molecules along <pcoord> in a box of <volume> with the bound state
        defined as <pcoord> below <bound_cutoff> and the unbound state defined as <pcoord> above <unbound_cutoff>.
        Standard error is estimated using the blocking method of Flyvbjerg, H., and Petersen, H. G. Error Estimates on
        Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466. """

    # Prepare timestep and concentrations
    dt              = time[1] - time[0]
    C_mol1_total    = _concentration(n_molecule_1, volume)
    C_mol2_total    = _concentration(n_molecule_2, volume)

    # Assign to bound (1) and unbound (0) states
    bound           = np.zeros(distance.shape, np.int8)
    unbound         = np.zeros(distance.shape, np.int8)
    bound[distance < bound_cutoff]      = 1
    unbound[distance > unbound_cutoff]  = 1
    for i in xrange(n_molecule_1):
        trans_bound     =   bound[1:,i] -   bound[:-1,i]
        trans_unbound   = unbound[1:,i] - unbound[:-1,i]
        enter_bound     = np.where(trans_bound   == 1)[0] + 1
        enter_unbound   = np.where(trans_unbound == 1)[0] + 1
        try:    enter   = enter_bound[0]                                # Start at first entrance of bound state
        except: continue                                                # Molecule never enters bound state
        while True:
            try:                                                        # Look for next entrance of unbound state
                exit    = enter_unbound[enter_unbound > enter][0]       #   Next entrance found
                bound[enter:exit,i]   = 1                               #   Set state to bound between entrances
            except:                                                     # Trajectory ends with pair in bound state
                bound[enter::,i]      = 1                               #   Set state to bound until end of trajectory
                break                                                   #   Exit
            try:                                                        # Look for next entrance of bound state
                enter   = enter_bound[enter_bound > exit][0]            #   Next entrance found
            except:                                                     # Trajectory ends with pair in unbound state
                break                                                   #   Exit

    # Calculate Pbound, Ka, and their evolution
    n_bound  = np.array(np.sum(bound, axis = 1), np.float)
    P_bound  = n_bound.copy()
    for i in xrange(1, time.size):
        P_bound[i] += P_bound[i-1]
    P_bound /= np.arange(1, time.size + 1, 1) * min(n_molecule_1, n_molecule_2)
    Ka       = _P_bound_to_Ka(P_bound, C_mol1_total, C_mol2_total)

    # Calculate standard error of final Ka using block averaging
    block_size, Pbound_se, Pbound_se_sd         = block_average(n_bound / min(n_molecule_1, n_molecule_2))
    min_asym, max_asym, poi, k, Pbound_se_fit   = fit_curve(x = block_size, y = Pbound_se, sigma = Pbound_se_sd,
                                                            fit_func = "sigmoid", **kwargs)
    block_duration  = np.array(block_size, np.float32) * dt
    poi            *= dt
    Ka_se           = _P_bound_se_to_Ka_se(P_bound[-1], C_mol1_total, C_mol2_total, max_asym)

    # Calculate mean first passage time
    fpt_on      = []
    fpt_off     = []
    for i in range(n_molecule_1):
        trans_bound     = bound[:-1,i] - bound[1:,i]
        enter_bound     = np.where(trans_bound  == -1)[0] + 1
        enter_unbound   = np.where(trans_bound  ==  1)[0] + 1
        if   enter_bound[0] < enter_unbound[0] and enter_bound[-1] < enter_unbound[-1]:
            fpt_on     += [enter_bound[1:]   - enter_unbound[:-1]]
            fpt_off    += [enter_unbound     - enter_bound]
        elif enter_bound[0] < enter_unbound[0] and enter_bound[-1] > enter_unbound[-1]:
            fpt_on     += [enter_bound[1:]   - enter_unbound]
            fpt_off    += [enter_unbound     - enter_bound[:-1]]
        elif enter_bound[0] > enter_unbound[0] and enter_bound[-1] < enter_unbound[-1]:
            fpt_on     += [enter_bound       - enter_unbound[:-1]]
            fpt_off    += [enter_unbound[1:] - enter_bound]
        elif enter_bound[0] > enter_unbound[0] and enter_bound[-1] > enter_unbound[-1]:
            fpt_on     += [enter_bound       - enter_unbound]
            fpt_off    += [enter_unbound[1:] - enter_bound[:-1]]
    fpt_on      = np.array(np.concatenate(fpt_on),  np.float32) * dt
    fpt_off     = np.array(np.concatenate(fpt_off), np.float32) * dt

    # Convert mean first passage time into rates
    kon_sim     = 1 / np.mean(fpt_on)
    koff_sim    = 1 / np.mean(fpt_off)
    kon_sim_se  = (kon_sim  ** 2 * np.std(fpt_on))  / np.sqrt(fpt_on.size)
    koff_sim_se = (koff_sim ** 2 * np.std(fpt_off)) / np.sqrt(fpt_off.size)
    kon         = kon_sim  / (C_mol1_total * C_mol2_total)
    koff        = koff_sim /  C_mol1_total
    kon_se      = kon  * (kon_sim_se  / kon_sim)
    koff_se     = koff * (koff_sim_se / koff_sim)

    # Organize and return data
    attrs           = {"bound_cutoff": float(bound_cutoff), "unbound_cutoff": float(unbound_cutoff),
                       "time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1]))}
    Pbound_attrs    = {"Ka": float(Ka[-1]), "Ka se": Ka_se, "Ka units": "M-1"}
    Ka              = np.array([tuple(frame) for frame in zip(time, Ka)],
                        np.dtype([("time", "f4"), ("Ka", "f4")]))
    block           = np.array([tuple(frame) for frame in zip(block_duration, Pbound_se, Pbound_se_fit)],
                        np.dtype([("block duration", "f4"), ("Pbound se", "f4"), ("Pbound se fit", "f4")]))
    block_attrs     = {"block duration units": "ns", "minimum asymptote": min_asym, "maximum asymptote": max_asym,
                       "point of inflection":  poi,  "k":                 k}
    fpt_attrs       = {"Ka":   kon / koff, "Ka se":   kon / koff * np.sqrt((kon_se / kon) ** 2 + (koff_se / koff) ** 2),
                       "kon":  kon,        "kon se":  kon_se,
                       "koff": koff,       "koff se": koff_se,
                       "Ka units": "M-1",  "kon units": "M-2 ns-1", "koff units": "M-1 ns-1"}

    if verbose:        _print_rate(destination, attrs, Pbound_attrs, fpt_attrs)
    return  [(destination + "/Pbound/Ka",       Ka),
             (destination + "/Pbound/Ka",       {"time units": "ns", "Ka units": "M-1"}),
             (destination + "/Pbound/block",    block),
             (destination + "/Pbound/block",    block_attrs),
             (destination + "/Pbound",          Pbound_attrs),
             (destination + "/fpt/on",          fpt_on),
             (destination + "/fpt/on",          {"units": "ns"}),
             (destination + "/fpt/off",         fpt_off),
             (destination + "/fpt/off",         {"units": "ns"}),
             (destination + "/fpt",             fpt_attrs),
             (destination,                      attrs)]

def _check_rate(hdf5_file, force = False, **kwargs):
    """ Determines whether or not to run 'association.rate' based on settings and data present in hdf5 file, and loads
        necessary primary data. """
    def _ignore_index(time, ignore):
        if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
        elif ignore == 0:   return 0
        elif ignore >  0:   return np.where(time > ignore)[0][0]
    def _load_distance(self, path, ignore_index = 0, **loader_kwargs):
        segments                = self._segments()
        shapes                  = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in segments])
        kwargs["n_molecule_1"]  = shapes[0, 1]
        kwargs["n_molecule_2"]  = shapes[0, 2]
        n_frames                = np.sum(shapes[:,0]) - ignore_index
        data                    = np.zeros((n_frames, kwargs["n_molecule_1"]), np.float32)
        i                       = 0
        for segment in segments:
            new_data    = self[segment + "/" + path[2:]]
            if  ignore_index  > new_data.shape[0]:
                ignore_index -= new_data.shape[0]
                continue
            elif ignore_index > 0:
                new_data        = new_data[ignore_index:]
                ignore_index    = 0
            data[i:i+new_data.shape[0]] = np.min(new_data, axis = 2)
            i                          += new_data.shape[0]
        return data

    # Parse kwargs and set defaults
    source                                    = kwargs.pop("source",        "*/association_mindist")
    source                                    = source if source.startswith("*/") else "*/" + source
    ignore                                    = kwargs.pop("ignore",         0)
    kwargs["bound_cutoff"]   = bound_cutoff   = kwargs.get("bound_cutoff",   3.5)
    kwargs["unbound_cutoff"] = unbound_cutoff = kwargs.get("unbound_cutoff", 3.5)
    kwargs["destination"]    = destination    = kwargs.get("destination",    "/association/mindist")
    kwargs["volume"]         = volume         = kwargs.get("volume",         None)

    # If analysis has not been run previously, run analysis
    expected    = [destination + "/Pbound", destination + "/Pbound/Ka", destination + "/Pbound/block",
                   destination + "/fpt",    destination + "/fpt/on",    destination + "/fpt/off"]
    if     (force
    or not (expected in hdf5_file)):
        log                 = hdf5_file.load("*/log", type = "table")
        ignore_index        = _ignore_index(log["time"], ignore)
        kwargs["time"]      = log["time"][ignore_index:]
        if volume is None:    kwargs["volume"] = np.mean(log["volume"]) * 1e-27
        kwargs["distance"]  = hdf5_file.load(source, loader = _load_distance, ignore_index = ignore_index)
        return [(rate, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    attrs           = hdf5_file.attrs(destination)
    log             = hdf5_file.load("*/log", type = "table")
    ignore_index    = _ignore_index(log["time"], ignore)
    kwargs["time"]  = log["time"][ignore_index:]
    if (attrs["bound_cutoff"]   != bound_cutoff
    or (attrs["unbound_cutoff"] != unbound_cutoff)
    or (attrs["time"]           != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        if volume is None:    kwargs["volume"] = np.mean(log["volume"]) * 1e-27
        kwargs["distance"]  = hdf5_file.load(source, loader = _load_distance, ignore_index = ignore_index)
        return [(rate, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False):
        Pbound_attrs    = hdf5_file.attrs(destination + "/Pbound")
        fpt_attrs       = hdf5_file.attrs(destination + "/fpt")
        _print_rate(destination, attrs, Pbound_attrs, fpt_attrs)
    return False

def _print_rate(address, attrs, Pbound_attrs, fpt_attrs):
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    print "DATASET  {0}".format(address)
    print "BOUND < {0:3.1f} A UNBOUND > {1:3.1f} A".format(attrs["bound_cutoff"],     attrs["unbound_cutoff"])
    print "Ka (se)         {0:>6.3f} ({1:>6.3f}) M-1     ".format(Pbound_attrs["Ka"], Pbound_attrs["Ka se"])
    print "kon (se)        {0:>6.0f} ({1:>6.0f}) M-2 ns-1".format(fpt_attrs["kon"],   fpt_attrs["kon se"])
    print "koff (se)       {0:>6.0f} ({1:>6.0f}) M-1 ns-1".format(fpt_attrs["koff"],  fpt_attrs["koff se"])
    print "kon/koff (se)   {0:>6.3f} ({1:>6.3f}) M-1     ".format(fpt_attrs["Ka"],    fpt_attrs["Ka se"])


def distance_histogram(hdf5_file,
        pcoord,                                             # Progress coordinate name
        bins,                                               # Bins
        source,                                             # Progress coordinate key within hdf5_file.data
        destination,                                        # Analysis output destination within HDF5 file
        verbose     = False, n_cores = 1, **kwargs):
    """ Calculates distance histogram along <pcoord> split into <bins>. Calculates histogram for closest pair,
        second-closest, ..., second-furthest, furthest"""
    duration    = hdf5_file.data["*/log"]["time"][-1]
    count       = hdf5_file.data[source]

    dtype       = eval("np.dtype([('lower bound', 'f4'), ('upper bound', 'f4')" +
                       "".join([", ('{0}', 'i4')".format(i) for i in xrange(count.shape[0])]) + "])")
    data        = np.array([tuple(frame) for frame in zip(bins[:-1], bins[1:], *count)], dtype)
    attrs       = {"lower bound units": "A", "upper bound units": "A", "time": duration}
    if verbose: _print_distance_histogram(pcoord, data, attrs)
    return  [(destination, data),
             (destination, attrs)]
def _check_distance_histogram(hdf5_file, force = False, **kwargs):
    """ Determines whether or not to run 'association.distance_histogram' based on settings and data present in hdf5
        file, and loads necessary primary data. 'source' and destination' may be set manually, but may also be guessed
        from pcoord. """
    pcoord      = kwargs.get("pcoord")
    bins        = kwargs.get("bins")
    source      = kwargs.pop("source",      "*/association_" + pcoord)
    destination = kwargs.get("destination",   "association/" + pcoord + "/distance_histogram")
    verbose     = kwargs.get("verbose",     False)

    duration    = hdf5_file.load("*/log", type = "table")["time"][-1]
    kwargs["destination"]   = destination

    if     (force                                                               # Run analysis if forced or analysis has
    or not (destination in hdf5_file)):                                         #   not previously been run
        kwargs["source"]    = "association.distance_histogram:{0:f}".format(time.time())
        hdf5_file.load(source, destination = kwargs["source"], loader = _load_association_distance_histogram,
          bins = bins)
        return [(distance_histogram, kwargs)]

    data    = hdf5_file[destination]
    attrs   = hdf5_file.attrs(destination)
    if ((data["lower bound"].size + 1 != bins.size)                             # Run analysis if it has been run
    or  (np.any(data["lower bound"]   != np.array(bins[:-1], np.float32)))      #   previously, but with different
    or  (np.any(data["upper bound"]   != np.array(bins[1:],  np.float32)))      #   settings or trajectory duration
    or  (attrs["time"]                != duration)):
        kwargs["source"]    = "association.distance_histogram:{0:f}".format(time.time())
        hdf5_file.load(source, destination = kwargs["source"], loader = _load_association_distance_histogram,
          bins = bins)
        return [(distance_histogram, kwargs)]
    if verbose: _print_distance_histogram(pcoord, data, attrs)                  # Do not run analysis
    return False
def _print_distance_histogram(pcoord, data, attrs):
    print "DURATION {0:5d} ns PROGRESS COORDINATE {1} ".format(int(attrs["time"]),   pcoord.upper())
    print "  LOWER  UPPER{0:>7d}{1:>7d}{2:>7d}{3:>7d}{4:>7d}{5:>7d}{6:>7d}{7:>7d}".format(*range(9)),
    print "{0:>7d}{1:>7d}{2:>7d}{3:>7d}{4:>7d}{5:>7d}{6:>7d}".format(*range(8, 15, 1))
    for line in data:
        print "{0:>7.3f}{1:>7.3f}{2:>7d}{3:>7d}".format(float(line[0]), float(line[1]), int(line[2]), int(line[3])),
        print "{0:>7d}{1:>7d}{2:>7d}{3:>7d}".format(int(line[4]),  int(line[5]),  int(line[6]),  int(line[7])),
        print "{0:>7d}{1:>7d}{2:>7d}{3:>7d}".format(int(line[8]),  int(line[9]),  int(line[10]), int(line[11])),
        print "{0:>7d}{1:>7d}{2:>7d}{3:>7d}".format(int(line[12]), int(line[13]), int(line[14]), int(line[15]))



