#!/usr/bin/python
# MD_toolkit.secondary.association.py
#    Written by Karl Debiec on 12-08-15
#    Last updated by Karl Debiec on 14-03-28
"""
Functions for secondary analysis of molecular association
"""
####################################################### MODULES ########################################################
import os, sys, time, types, warnings
import numpy as np
from   scipy.optimize import fmin
from   MD_toolkit.standard_functions import block_average, fit_curve, ignore_index
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
    """
    Calculates Ka, kon, and koff for association of <n_molecule_1> molecules of type 1 and <n_molecule_2> molecules
    of type 2 in a box of <volume>. The bound state is defined as <distance> below <bound_cutoff> and the unbound
    state is defined as <distance> above <unbound_cutoff>. In between the cutoffs, the molecule is assigned to
    whichever state it most recently occupied.
    Standard error is estimated using the blocking method of Flyvbjerg, H., and Petersen, H. G. Error Estimates on
    Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466.
    """

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
    attrs        = {"concentration_1": C_mol1_total,        "concentration_2": C_mol2_total,
                    "bound_cutoff":    float(bound_cutoff), "unbound_cutoff":  float(unbound_cutoff),
                    "volume":          volume * 1e27,
                    "time": "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1]))}
    Pbound_attrs = {"Ka": float(Ka[-1]),          "Ka_se":     Ka_se,   "Ka_units": "M-1",
                    "Pbound": float(P_bound[-1]), "Pbound_se": max_asym}
    Ka           = np.array([tuple(frame) for frame in zip(time, P_bound, Ka)],
                     [("time", "f4"), ("Pbound", "f4"), ("Ka", "f4")])
    block        = np.array([tuple(frame) for frame in zip(block_duration, Pbound_se, Pbound_se_fit)],
                     [("block duration", "f4"), ("Pbound se", "f4"), ("Pbound se fit", "f4")])
    block_attrs  = {"block_duration_units": "ns", "minimum_asymptote": min_asym, "maximum_asymptote": max_asym,
                    "point_of_inflection":  poi,  "k":                 k}
    fpt_attrs    = {"Ka":   kon / koff, "Ka_se":   kon / koff * np.sqrt((kon_se / kon) ** 2 + (koff_se / koff) ** 2),
                    "kon":  kon,        "kon_se":  kon_se,
                    "koff": koff,       "koff_se": koff_se,
                    "Ka_units": "M-1",  "kon_units": "M-2 ns-1", "koff_units": "M-1 ns-1"}

    if verbose:        _print_rate(destination, attrs, Pbound_attrs, fpt_attrs)
    return  [(destination + "/Pbound/Ka",    Ka),
             (destination + "/Pbound/Ka",    {"time_units": "ns", "Ka_units": "M-1"}),
             (destination + "/Pbound/block", block),
             (destination + "/Pbound/block", block_attrs),
             (destination + "/Pbound",       Pbound_attrs),
             (destination + "/fpt/on",       fpt_on),
             (destination + "/fpt/on",       {"units": "ns"}),
             (destination + "/fpt/off",      fpt_off),
             (destination + "/fpt/off",      {"units": "ns"}),
             (destination + "/fpt",          fpt_attrs),
             (destination,                   attrs)]

def _check_rate(hdf5_file, force = False, **kwargs):
    """ Determines whether or not to run 'association.rate' based on settings and data present in hdf5 file, and loads
        necessary primary data. """
    def load_distance(self, path, index = 0, **loader_kwargs):
        segments               = self._segments()
        shapes                 = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in segments])
        kwargs["n_molecule_1"] = shapes[0, 1]
        kwargs["n_molecule_2"] = shapes[0, 2]
        n_frames               = np.sum(shapes[:,0]) - index
        data                   = np.zeros((n_frames, kwargs["n_molecule_1"]), np.float32)
        i                      = 0
        for segment in segments:
            new_data     = self[segment + "/" + path[2:]]
            if  index    > new_data.shape[0]:
                index   -= new_data.shape[0]
                continue
            elif index > 0:
                new_data = new_data[index:]
                index    = 0
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
    expected = [destination + "/Pbound", destination + "/Pbound/Ka", destination + "/Pbound/block",
                destination + "/fpt",    destination + "/fpt/on",    destination + "/fpt/off"]
    if     (force
    or not (expected in hdf5_file)):
        log                  = hdf5_file.load("*/log", type = "table")
        index                = ignore_index(log["time"], ignore)
        kwargs["time"]       = log["time"][index:]
        if volume is None:
            kwargs["volume"] = np.mean(log["volume"]) * 1e-27
        kwargs["distance"]   = hdf5_file.load(source, loader = load_distance, index = index)
        return [(rate, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    attrs          = hdf5_file.attrs(destination)
    log            = hdf5_file.load("*/log", type = "table")
    index          = ignore_index(log["time"], ignore)
    kwargs["time"] = log["time"][index:]
    if (not all(key in attrs for key in ["bound_cutoff", "unbound_cutoff", "time"])
    or (attrs["bound_cutoff"]   != bound_cutoff)
    or (attrs["unbound_cutoff"] != unbound_cutoff)
    or (attrs["time"]           != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        if volume is None:
            kwargs["volume"]     = np.mean(log["volume"]) * 1e-27
        kwargs["distance"]       = hdf5_file.load(source, loader = load_distance, index = index)
        return [(rate, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False):
        Pbound_attrs = hdf5_file.attrs(destination + "/Pbound")
        fpt_attrs    = hdf5_file.attrs(destination + "/fpt")
        _print_rate(destination, attrs, Pbound_attrs, fpt_attrs)
    return False

def _print_rate(address, attrs, Pbound_attrs, fpt_attrs):
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    print "DATASET  {0}".format(address)
    print "BOUND < {0:4.2f} A UNBOUND > {1:4.2f} A".format(attrs["bound_cutoff"],         attrs["unbound_cutoff"])
    print "VOLUME       {0:d} A^3".format(int(attrs["volume"]))
    print "[MOLECULE 1] {0:4.3f} M".format(float(attrs["concentration_1"]))
    print "[MOLECULE 2] {0:4.3f} M".format(float(attrs["concentration_2"]))
    print "Pbound   (se)   {0:>6.3f} ({1:>6.3f}) M-1     ".format(Pbound_attrs["Pbound"], Pbound_attrs["Pbound_se"])
    print "Ka       (se)   {0:>6.3f} ({1:>6.3f}) M-1     ".format(Pbound_attrs["Ka"],     Pbound_attrs["Ka_se"])
    print "kon      (se)   {0:>6.0f} ({1:>6.0f}) M-2 ns-1".format(fpt_attrs["kon"],       fpt_attrs["kon_se"])
    print "koff     (se)   {0:>6.0f} ({1:>6.0f}) M-1 ns-1".format(fpt_attrs["koff"],      fpt_attrs["koff_se"])
    print "kon/koff (se)   {0:>6.3f} ({1:>6.3f}) M-1     ".format(fpt_attrs["Ka"],        fpt_attrs["Ka_se"])

def exchange(hdf5_file,
        source,                                             # Primary data source
        index,                                              # Index of first data point to include
        time,                                               # Simulation time
        volume,                                             # System volume (L)
        bound_cutoff,                                       # Bound state cutoff (A)
        unbound_cutoff,                                     # Unbound state cutoff (A)
        n_molecule_1,                                       # Number of molecules of type 1
        n_molecule_2,                                       # Number of molecules of type 2
        destination,                                        # Analysis output destination within hdf5 file
        verbose = False, debug = False, n_cores = 1, **kwargs):
    """
    Analyzes pairwise association between <n_molecule_1> molecules of type 1 and <n_molecule_2> molecules of type 2 in a
    box of <volume>. The bound state is defined as <source> distance metric below <bound_cutoff> and the unbound state
    is defined as <distance> above <unbound_cutoff>. In between the cutoffs, the molecule is assigned to whichever state
    it most recently occupied. Calculates the kon and koff of each individual pairs, as well as averages over all pairs.
    Additionally calculates the KA values for the association of molecule_1 with molecule_2, as well as those for
    association of molecule_1 with multiple molecule_2. Does not current account for association of multiple molecule_1
    with a single molecule 2. 
    Standard errors of the probability of the unbound and each bound states are estimated using the blocking method of
    Flyvbjerg, H., and Petersen, H. G. Error Estimates on Averages of Correlated Data. J Phys Chem. 1989. 91. 461-466.
    These errors are propogated into the concentrations of each state, and into the calculated association constants.
    """

    def load_distance(self, path, j, k, n_frames, index = 0):
        """
        HDF5 loader function used to load distance metric for atoms <j> and <k> from <path>
        Requires total <n_frames> 
        """
        data                   = np.zeros(n_frames, np.float32)
        i                      = 0
        for segment in self._segments():
            new_data     = np.array(self.hierarchy[self._strip_path(segment + "/" + path[2:])][:, j, k])
            if  index    > new_data.size:
                index   -= new_data.size
                continue
            elif index > 0:
                new_data = new_data[index:]
                index    = 0
            data[i:i+new_data.size] = new_data
            i                      += new_data.size
        return data

    # Prepare timestep and concentrations
    dt           = np.mean(time[1:] - time[:-1], dtype = np.float64)            # Allows small number of dropped frames
    n_frames     = time.size
    C_1          = _concentration(1,            volume)
    C_mol1_total = _concentration(n_molecule_1, volume)
    C_mol2_total = _concentration(n_molecule_2, volume)
    count        = np.zeros((n_frames, n_molecule_1), np.int)                   # Number of mol_2 bound to each mol_1
    events       = []                                                           # List of binding events
    stat_dtype   = [("Pbound",       "f4"), ("Pbound se",       "f4"), ("mean fpt on", "f4"), ("mean fpt on se", "f4"),
                    ("mean fpt off", "f4"), ("mean fpt off se", "f4"), ("kon",         "f4"), ("kon se",         "f4"),
                    ("koff",         "f4"), ("koff se",         "f4")]
    pair_stats   = np.zeros((n_molecule_1, n_molecule_2), stat_dtype)           # Binding statistics for each pair
    total_stats  = np.zeros(1, stat_dtype)                                      # Averages over all pairs

    for j in range(n_molecule_1):
        for k in range(n_molecule_2):
            # Load distance data for j, k pair only
            distance    = hdf5_file.load(source, loader = load_distance, j = j, k = k, n_frames = n_frames, index = index)

            # Assign to bound (1) and unbound (0) states
            bound           = np.zeros(n_frames, np.int8)
            unbound         = np.zeros(n_frames, np.int8)
            bound[distance < bound_cutoff]      = 1
            unbound[distance > unbound_cutoff]  = 1
            trans_bound     =   bound[1:] -   bound[:-1]
            trans_unbound   = unbound[1:] - unbound[:-1]
            enter_bound     = np.where(trans_bound   == 1)[0] + 1
            enter_unbound   = np.where(trans_unbound == 1)[0] + 1
            try:    enter   = enter_bound[0]                                # Start at first entrance of bound state
            except: continue                                                # Pair never enters bound state
            while True:
                try:                                                        # Look for next entrance of unbound state
                    exit    = enter_unbound[enter_unbound > enter][0]       #   Next entrance found
                    bound[enter:exit] = 1                                   #   Set state to bound between entrances
                except:                                                     # Trajectory ends with pair in bound state
                    bound[enter::]    = 1                                   #   Set state to bound until end of trajectory
                    break                                                   #   Exit
                try:                                                        # Look for next entrance of bound state
                    enter   = enter_bound[enter_bound > exit][0]            #   Next entrance found
                except:                                                     # Trajectory ends with pair in unbound state
                    break                                                   #   Exit

            # Calculate Pbound
            Pbound  = np.sum(bound, dtype = np.float64) / float(bound.size)

            # Calculate standard error of Pbound using block averaging
            sizes, ses, se_sds      = block_average(bound)
            try:    a, Pbound_se, c, d, fit = fit_curve(x = sizes, y = ses, sigma = se_sds, fit_func = "sigmoid")
            except:    Pbound_se            = Pbound

            # Calculate mean first passage time and tabulate binding events
            trans_bound     = bound[:-1] - bound[1:]
            enter_bound     = np.where(trans_bound  == -1)[0] + 1
            enter_unbound   = np.where(trans_bound  ==  1)[0] + 1
            if enter_bound.size >= 1 and enter_unbound.size >= 1:
                if   enter_bound[0] < enter_unbound[0] and enter_bound[-1] < enter_unbound[-1]:     # Started unbound,
                    fpt_on   = enter_bound[1:]   - enter_unbound[:-1]                               #   ended unbound
                    fpt_off  = enter_unbound     - enter_bound
                    events  += [(j, k, bind, unbind, unbind - bind)
                                   for bind, unbind in np.column_stack((enter_bound, enter_unbound))] 
                elif enter_bound[0] < enter_unbound[0] and enter_bound[-1] > enter_unbound[-1]:     # Started unbound,
                    fpt_on   = enter_bound[1:]   - enter_unbound                                    #   ended bound
                    fpt_off  = enter_unbound     - enter_bound[:-1]
                    events  += [(j, k, bind, unbind, unbind - bind)
                                   for bind, unbind in np.column_stack((enter_bound[:-1], enter_unbound))] 
                elif enter_bound[0] > enter_unbound[0] and enter_bound[-1] < enter_unbound[-1]:     # Started bound,
                    fpt_on   = enter_bound       - enter_unbound[:-1]                               #   ended unbound
                    fpt_off  = enter_unbound[1:] - enter_bound
                    events  += [(j, k, bind, unbind, unbind - bind)
                                   for bind, unbind in np.column_stack((enter_bound, enter_unbound[1:]))] 
                elif enter_bound[0] > enter_unbound[0] and enter_bound[-1] > enter_unbound[-1]:     # Started bound,
                    fpt_on   = enter_bound       - enter_unbound                                    #   ended bound
                    fpt_off  = enter_unbound[1:] - enter_bound[:-1]
                    events  += [(j, k, bind, unbind, unbind - bind)
                                   for bind, unbind in np.column_stack((enter_bound[:-1] * dt, enter_unbound[1:] * dt))]

                # Convert mean first passage time into rates
                mean_fpt_on     = np.mean(fpt_on)
                mean_fpt_off    = np.mean(fpt_off)
                mean_fpt_on_se  = np.std(fpt_on)  / np.sqrt(fpt_on.size)
                mean_fpt_off_se = np.std(fpt_off) / np.sqrt(fpt_off.size)
                kon_sim         = 1 / mean_fpt_on
                koff_sim        = 1 / mean_fpt_off
                kon_sim_se      = kon_sim  * (mean_fpt_on_se  / mean_fpt_on)
                koff_sim_se     = koff_sim * (mean_fpt_off_se / mean_fpt_off)
                kon             = kon_sim  / (C_1 * C_1)
                koff            = koff_sim /  C_1
                kon_se          = kon      * (kon_sim_se      / kon_sim)
                koff_se         = koff     * (koff_sim_se     / koff_sim) 

            # Pair never switches between bound and unbound states
            else:
                mean_fpt_on,  mean_fpt_on_se    = np.nan, np.nan
                mean_fpt_off, mean_fpt_off_se   = np.nan, np.nan
                kon,          kon_se            = np.nan, np.nan
                koff,         koff_se           = np.nan, np.nan

            # Cleanup large unneeded numpy arrays (may or may not make a difference)
            del(distance, unbound, sizes, trans_bound, trans_unbound, enter_bound, enter_unbound)

            # Organize data
            count[:, j]                                                        += bound
            pair_stats[j,k]["Pbound"],       pair_stats[j,k]["Pbound se"]       = Pbound,       Pbound_se
            pair_stats[j,k]["mean fpt on"],  pair_stats[j,k]["mean fpt on se"]  = mean_fpt_on,  mean_fpt_on_se
            pair_stats[j,k]["mean fpt off"], pair_stats[j,k]["mean fpt off se"] = mean_fpt_off, mean_fpt_off_se
            pair_stats[j,k]["kon"],          pair_stats[j,k]["kon se"]          = kon,          kon_se
            pair_stats[j,k]["koff"],         pair_stats[j,k]["koff se"]         = koff,         koff_se
            if debug:
                print "Molecule 1: {0:03d}    Molecule 2: {1:03d}    Pbound (se): {2:7.5f} ({3:7.5f})".format(j, k,
                      float(pair_stats[j, k]["Pbound"]), float(pair_stats[j, k]["Pbound se"]))

    # Calculate overall averages and organize data
    for key in ["Pbound", "mean fpt on", "mean fpt off", "kon", "koff"]:
        pair_stat_masked         = np.ma.MaskedArray(pair_stats[key],         np.isnan(pair_stats[key]))
        pair_stat_masked_se      = np.ma.MaskedArray(pair_stats[key + " se"], np.isnan(pair_stats[key + " se"]))
        total_stats[key]         = np.mean(pair_stat_masked)
        total_stats[key + " se"] = np.sqrt(np.sum(pair_stat_masked_se ** 2.0)) / pair_stat_masked.size
    events                       = np.array(events, [("index 1", "i4"), ("index 2",  "i4"), ("start", "f4"),
                                                     ("end",     "f4"), ("duration", "f4")])

    # Calculate Pstate of different bound states
    n_states    = np.max(count) + 1
    Pstate      = np.zeros(n_states)
    Pstate_se   = np.zeros(n_states)
    for i in range(0, n_states):                                                # i = number of mol 2 bound to mol 1
        Pstate[i]       = float(count[count == i].size) / float(count.size)     # Accounts for presence of mult. mol 1
        mol1_in_state_i = np.zeros(count.shape, np.int8)                        # Binary over trajectory
        mol1_in_state_i[count == i] = 1
        total_mol1_in_state_i       = np.sum(mol1_in_state_i, axis = 1, dtype = np.float64)

        # Calculate standard error of Pstate using block averaging
        sizes, ses, se_sds                  = block_average(total_mol1_in_state_i / n_molecule_1)
        try:    a, Pstate_se[i], c, d, fit  = fit_curve(x = sizes, y = ses, sigma = se_sds, fit_func = "sigmoid")
        except:    Pstate_se[i]             = Pstate[i]
    if debug:
        for i in range(Pstate.size): print "P state {0:02d}: {1:7.5f} ({2:7.5f})".format(i, Pstate[i], Pstate_se[i])
        print "Sum       : {0:7.5f}".format(np.sum(Pstate))

    # Calculate concentrations
    C_mol1_free     = C_mol1_total * Pstate[0]
    C_mol1_free_se  = C_mol1_free  * (Pstate_se[0] / Pstate[0])
    C_mol2_free     = C_mol2_total - C_mol1_total * np.sum([i * P for i, P in enumerate(Pstate[1:], 1)])
    C_mol2_free_se  = C_mol1_total * np.sqrt(np.sum([(i * P_se) ** 2 for i, P_se in enumerate(Pstate_se[1:], 1)]))
    C_bound         = C_mol1_total * Pstate[1:]
    C_bound_se      = C_bound      * (Pstate_se[1:] / Pstate[1:])
    np.set_printoptions(precision = 5, suppress = True, linewidth = 120)
    if debug:
        print "[mol1 total]:      {0:7.5f}".format(C_mol1_total)
        print "[mol2 total]:      {0:7.5f}".format(C_mol2_total)
        print "[mol1 free] (se):  {0:7.5f} ({1:7.5f})".format(C_mol1_free, C_mol1_free_se)
        print "[mol2 free] (se):  {0:7.5f} ({1:7.5f})".format(C_mol2_free, C_mol2_free_se)
        for i in range(C_bound.size):
            print "[complex {0:02d}] (se): {1:7.5f} ({2:7.5f})".format(i + 1, C_bound[i],     C_bound_se[i])
        print "Sum [mol1]:        {0:7.5f}".format(C_mol1_free + np.sum(C_bound))
        print "Sum [mol2]:        {0:7.5f}".format(C_mol2_free + np.sum([i * C for i, C in enumerate(C_bound, 1)]))

    # Calculate KAs
    KA              = np.zeros(n_states - 1)
    KA_se           = np.zeros(n_states - 1)
    KA[0]           = C_bound[0] / (C_mol1_free * C_mol2_free)
    KA_se[0]        = KA[0] * np.sqrt((C_bound_se[0] / C_bound[0]) ** 2 + (C_mol1_free_se  / C_mol1_free)  ** 2
                                      + (C_mol2_free_se / C_mol2_free) ** 2)
    for i in range(1, n_states - 1):
        KA[i]       = C_bound[i] / (C_bound[i-1] * C_mol2_free)
        KA_se[i]    = KA[i] * np.sqrt((C_bound_se[i] / C_bound[i]) ** 2 + (C_bound_se[i-1] / C_bound[i-1]) ** 2
                                      + (C_mol2_free_se / C_mol2_free) ** 2)
    KA_list       = [Pstate[0],                             Pstate_se[0]]
    KA_dtype      = [("P unbound", "f4"),                   ("P unbound se", "f4")]
    for i in range(1, Pstate.size):
        KA_list  += [Pstate[i],                             Pstate_se[i]]
        KA_dtype += [("P {0} bound".format(i), "f4"),       ("P {0} bound se".format(i), "f4")]
    KA_list      += [C_mol1_free,                           C_mol1_free_se]
    KA_dtype     += [("[mol1 free]", "f4"),                 ("[mol1 free] se", "f4")]
    KA_list      += [C_mol2_free,                           C_mol2_free_se]
    KA_dtype     += [("[mol2 free]", "f4"),                 ("[mol2 free] se", "f4")]
    for i in range(C_bound.size):
        KA_list  += [C_bound[i],                            C_bound_se[i]]
        KA_dtype += [("[complex {0}]".format(i + 1), "f4"), ("[complex {0}] se".format(i + 1), "f4")]
    KA_list      += [KA[0],                                 KA_se[0]]
    KA_dtype     += [("KA 1", "f4"),                        ("KA 1 se", "f4")]
    for i in range(1, KA.size):
        KA_list  += [KA[i],                                 KA_se[i]]
        KA_dtype += [("KA {0}".format(i + 1), "f4"),        ("KA {0} se".format(i + 1), "f4")]
    KA_array      = np.array([tuple(KA_list)], KA_dtype)
    if debug:
        for i in range(KA.size): print "KA {0:02d}: {1:7.5f} ({2:7.5f})".format(i + 1, KA[i], KA_se[i])

    # Organize and return data
    pair_stat_attrs     = {"fpt units":      "ps", "kon units": "M-2 ps-1", "koff units": "M-1 ps-1"}
    total_stat_attrs    = {"fpt units":      "ps", "kon units": "M-2 ps-1", "koff units": "M-1 ps-1"}
    events_attrs        = {"duration units": "ps"}
    KA_attrs            = {"KA units":       "M-1"}
    attrs               = {"N molecule 1":    n_molecule_1,        "N molecule 2":    n_molecule_2,
                           "[molecule 1]":    C_mol1_total,        "[molecule 2]":    C_mol2_total,
                           "bound cutoff":    float(bound_cutoff), "unbound cutoff":  float(unbound_cutoff),
                           "volume":          volume * 1e27,
                           "time":            "{0:.3f} {1:.3f}".format(float(time[0]), float(time[-1]))}
    if verbose:    _print_exchange(destination, total_stats, events, KA_array, attrs)
    return  [(destination + "/pairs/pair_stats",   pair_stats),
             (destination + "/pairs/pair_stats",   pair_stat_attrs),
             (destination + "/pairs/total_stats",  total_stats),
             (destination + "/pairs/total_stats",  total_stat_attrs),
             (destination + "/pairs/events",       events),
             (destination + "/pairs/KA",           KA_array),
             (destination + "/pairs/KA",           KA_attrs),
             (destination + "/pairs",              attrs)]

def _check_exchange(hdf5_file, force = False, **kwargs):
    """
    Determines whether or not to run 'association.exchange' based on settings and data present in hdf5 file, and loads
    necessary primary data.
    """

    # Parse kwargs and set defaults
    source                                    = kwargs.pop("source",        "*/association_mindist")
    kwargs["source"]         = source         = source if source.startswith("*/") else "*/" + source
    ignore                                    = kwargs.pop("ignore",         0)
    kwargs["bound_cutoff"]   = bound_cutoff   = kwargs.get("bound_cutoff",   3.5)
    kwargs["unbound_cutoff"] = unbound_cutoff = kwargs.get("unbound_cutoff", 3.5)
    kwargs["destination"]    = destination    = kwargs.get("destination",    "/association/mindist")
    kwargs["volume"]         = volume         = kwargs.get("volume",         None)

    # If analysis has not been run previously, run analysis
    expected = [destination + "/pairs",             destination + "/pairs/pair_stats",
                destination + "/pairs/total_stats", destination + "/pairs/events",
                destination + "/pairs/KA"]
    if     (force
    or not (expected in hdf5_file)):
        log                     = hdf5_file.load("*/log", type = "table")
        kwargs["index"] = index = ignore_index(log["time"], ignore)
        kwargs["time"]          = log["time"][index:]
        if volume is None:
            kwargs["volume"]    = np.mean(log["volume"]) * 1e-27                                     # A^3 -> L (dm^3)
        shape                   = hdf5_file.hierarchy[hdf5_file._segments()[0] + "/" + source[2:]].shape
        kwargs["n_molecule_1"]  = shape[1]
        kwargs["n_molecule_2"]  = shape[2]
        return [(exchange, kwargs)]

    # If analysis has been run previously but with different settings, run analysis
    attrs                   = hdf5_file.attrs(destination + "/pairs")
    log                     = hdf5_file.load("*/log", type = "table")
    kwargs["index"] = index = ignore_index(log["time"], ignore)
    kwargs["time"]          = log["time"][index:]
    if (attrs["bound cutoff"]   != bound_cutoff
    or (attrs["unbound cutoff"] != unbound_cutoff)
    or (attrs["time"]           != "{0:.3f} {1:.3f}".format(float(kwargs["time"][0]), float(kwargs["time"][-1])))):
        if volume is None:
            kwargs["volume"]     = np.mean(log["volume"]) * 1e-27                                   # A^3 -> L (dm^3)
        shape                    = hdf5_file.hierarchy[hdf5_file._segments()[0] + "/" + source[2:]].shape
        kwargs["n_molecule_1"]   = shape[1]
        kwargs["n_molecule_2"]   = shape[2]
        return [(exchange, kwargs)]

    # If analysis has been run previously with the same settings, output data and return
    if kwargs.get("verbose", False):
        stats        = hdf5_file.load(destination + "/pairs/total_stats")
        events       = hdf5_file.load(destination + "/pairs/events")
        KA           = hdf5_file.load(destination + "/pairs/KA")
        _print_exchange(destination, stats, events, KA, attrs)
    return False

def _print_exchange(destination, stats, events, KA, attrs):
    stats  = {key: float(stats[0][key]) for key in stats.dtype.names}
    KA     = {key: float(KA[0][key])    for key in KA.dtype.names}
    print "TIME     {0:.3f} ns - {1:.3f} ns".format(*map(float, attrs["time"].split()))
    print "DURATION {0:.3f} ns".format(float(attrs["time"].split()[1]) - float(attrs["time"].split()[0]))
    print "DATASET  {0}".format(destination)
    print "BOUND < {0:4.2f} A UNBOUND > {1:4.2f} A".format(attrs["bound cutoff"], attrs["unbound cutoff"])
    print "VOLUME       {0:d} A^3".format(int(attrs["volume"]))
    print "[MOLECULE 1] {0:4.3f} M".format(float(attrs["[molecule 1]"]))
    print "[MOLECULE 2] {0:4.3f} M".format(float(attrs["[molecule 2]"]))
    print "PAIRS"
    print "    mean P bound (se)  {0:>7.3f} ({1:>7.3f})".format(         stats["Pbound"],       stats["Pbound se"])
    print "    mean fpt on  (se)  {0:>7.3f} ({1:>7.3f}) ps".format(      stats["mean fpt on"],  stats["mean fpt on se"])
    print "    mean fpt off (se)  {0:>7.3f} ({1:>7.3f}) ps".format(      stats["mean fpt off"], stats["mean fpt off se"])
    print "    mean kon     (se)  {0:>7.3f} ({1:>7.3f}) M-2 ps-1".format(stats["kon"],          stats["kon se"])
    print "    mean koff    (se)  {0:>7.3f} ({1:>7.3f}) M-2 ps-1".format( stats["koff"],         stats["koff se"])
    print "    N events           {0:>7d}".format(events.size)
    print "OVERALL"
    print "    P unbound    (se)  {0:>7.5f} ({1:>7.5f})".format(         KA["P unbound"],       KA["P unbound se"])
    for i, key in enumerate(sorted([k for k in KA if k.endswith(" bound")]), 1):
        print "    P {0:02d} bound   (se)  {1:>7.5f} ({2:>7.5f})".format(i,KA[key],               KA[key + " se"])
    print "    [mol1 free]  (se)  {0:>7.5f} ({1:>7.5f})".format(         KA["[mol1 free]"],     KA["[mol1 free] se"])
    print "    [mol2 free]  (se)  {0:>7.5f} ({1:>7.5f})".format(         KA["[mol2 free]"],     KA["[mol2 free] se"])
    for i, key in enumerate(sorted([k for k in KA if k.startswith("[complex ") and not k.endswith("se")]), 1):
        print "    {0}  (se)  {1:>7.5f} ({2:>7.5f})".format(key,         KA[key],     KA[key + " se"])
    for i, key in enumerate(sorted([k for k in KA if k.startswith("KA ") and not k.endswith("se")]), 1):
        print "    {0}         (se)  {1:>7.5f} ({2:>7.5f})".format(key,         KA[key],     KA[key + " se"])


