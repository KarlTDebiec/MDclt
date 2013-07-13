#!/usr/bin/python
desc = """association.py
    Functions for secondary analysis of molecular association
    Written by Karl Debiec on 12-08-15
    Last updated 13-07-12"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import numpy as np
from   scipy.optimize import curve_fit
################################################## INTERNAL FUNCTIONS ##################################################
def _concentration(n, volume): return n / 6.0221415e23 / volume
def _P_bound_to_Ka(P_bound, C_mol1_total, C_mol2_total):
    C_complex       = P_bound * np.min(C_mol1_total, C_mol2_total)
    C_mol1_unbound  = C_mol1_total - C_complex
    C_mol2_unbound  = C_mol2_total - C_complex
    return            C_complex / (C_mol1_unbound * C_mol2_unbound)
def _P_bound_se_to_Ka_se(P_bound, C_mol1_total, C_mol2_total, P_bound_se):
    return np.sqrt((((C_mol2_total - C_mol1_total * P_bound ** 2) * P_bound_se) /
                    ((P_bound - 1) ** 2 * (C_mol1_total * P_bound - C_mol2_total) ** 2)) ** 2)
def _block(data):
    full_size       = data.size
    sizes           = np.array(sorted(list(set([full_size / x for x in range(1, full_size)]))), np.int)[:-1]
    sds             = np.zeros(sizes.size)
    n_blocks        = full_size // sizes
    for i, size in enumerate(sizes):
        resized     = np.resize(data, (full_size // size, size))
        means       = np.mean(resized, axis = 1)
        sds[i]      = np.std(means)
    ses             = sds / np.sqrt(n_blocks)
    return sizes, ses
def _fit_sigmoid(x, y):
    def _model_function(x, min_asym, max_asym, poi, k): return max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_asym, max_asym, poi, k  = curve_fit(_model_function, x, y)[0]
    y_fit   = max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    return min_asym, max_asym, poi, k, y_fit
################################################ PRIMARY DATA FUNCTIONS ################################################
def _load_association(self, path, **kwargs):
    segments                    = self._segments()
    shapes                      = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in segments])
    self.data["n_molecule_1"]   = shapes[0,1]
    self.data["n_molecule_2"]   = shapes[0,2]
    data                        = np.zeros((np.sum(shapes[:,0]), self.data["n_molecule_1"]), np.float32)
    i                           = 0
    for j, segment in enumerate(segments):
        data[i:i+shapes[j,0]]   = np.min(self[segment + "/" + path[2:]], axis = 2)
        i                      += shapes[j,0]
    return data
def _load_association_pmf(self, path, bins, **kwargs):
    segments        = self._segments()
    data            = np.zeros(bins.size - 1, np.int32)
    for segment in segments:
        hist, _     = np.histogram(self[segment + "/" + path[2:]], bins)
        data       += hist
    return data
################################################## ANALYSIS FUNCTIONS ##################################################
def two_state(hdf5_file,
              volume,                                       # System volume
              primary       = "comdist",                    # Primary dataset; path is '*/association_<primary>'
              cutoff        = 4.5,                          # Bound/unbound cutoff (A)
              n_molecule_1  = 1,                            # Number of molecules of type 1
              n_molecule_2  = 1,                            # Number of molecules of type 2
              verbose       = False, n_cores = 1, **kwargs):
    """ Calculates Ka, kon, and koff of <n_molecule_1> molecules of type 1 and <n_molecule_2> molecules of type 2 in a
        cubic box of <side length> with the bound state defined as distance measurment <primary> below <cutoff> 
        Angstrom. Follows the protocol of Piana, S., Lindorff-Larsen, K., Shaw, D.E. How Robust Are Protein Folding
        Simulations with Respect to Force Field Parameterization? Biophys J. 2011. 100. L47-L49. Error is estimated
        using the blocking method of Flyvbjerg, H., and Petersen, H. G. Error Estimates on Averages of Correlated Data.
        J Phys Chem. 1989. 91. 461-466. """
    volume         *= 1e-27                                 # (A^3 > L)
    time            = hdf5_file.data["*/log"]["time"]
    distance        = hdf5_file.data["*/association_" + primary]
    dt              = time[1] - time[0]
    C_mol1_total    = _concentration(n_molecule_1, volume)
    C_mol2_total    = _concentration(n_molecule_2, volume)

    bound                       = np.zeros(distance.shape, np.float32)
    bound[distance < cutoff]    = 1
    P_bound                     = np.zeros(time.size, np.float32)
    total_bound                 = np.sum(bound, axis = 1)
    P_bound[0]                  = total_bound[0]
    for i in range(1, time.size):
        P_bound[i]              = (P_bound[i-1] + total_bound[i])
    P_bound                    /= np.arange(n_molecule_1, bound.size + 1, n_molecule_1)
    Ka                          = _P_bound_to_Ka(P_bound, C_mol1_total, C_mol2_total)

    block_size, Pbound_se       = _block(total_bound / n_molecule_1)
    min_asym, max_asym, poi, k, Pbound_se_fit   = _fit_sigmoid(block_size, Pbound_se)
    block_duration              = np.array(block_size, np.float32) * dt
    poi                        *= dt
    Ka_se                       = _P_bound_se_to_Ka_se(P_bound[-1], C_mol1_total, C_mol2_total, max_asym)

    fpt_on      = []
    fpt_off     = []
    for i in range(n_molecule_1):
        transitions =  bound[:-1,i] - bound[1:,i]
        ons         =  time[np.where(transitions == -1)[0] + 1]
        offs        =  time[np.where(transitions ==  1)[0] + 1]
        if   ons[0] < offs[0] and ons[-1] < offs[-1]:
            fpt_on     += [ons[1:]  - offs[:-1]]
            fpt_off    += [offs     - ons]
        elif ons[0] < offs[0] and ons[-1] > offs[-1]:
            fpt_on     += [ons[1:]  - offs]
            fpt_off    += [offs     - ons[:-1]]
        elif ons[0] > offs[0] and ons[-1] < offs[-1]:
            fpt_on     += [ons      - offs[:-1]]
            fpt_off    += [offs[1:] - ons]
        elif ons[0] > offs[0] and ons[-1] > offs[-1]:
            fpt_on     += [ons      - offs]
            fpt_off    += [offs[1:] - ons[:-1]]
    fpt_on      = np.concatenate(fpt_on)
    fpt_off     = np.concatenate(fpt_off)

    kon_sim     = 1 / np.mean(fpt_on)
    koff_sim    = 1 / np.mean(fpt_off)
    kon_sim_se  = (kon_sim  ** 2 * np.std(fpt_on))  / np.sqrt(fpt_on.size)
    koff_sim_se = (koff_sim ** 2 * np.std(fpt_off)) / np.sqrt(fpt_off.size)
    kon         = kon_sim  / (C_mol1_total * C_mol2_total)
    koff        = koff_sim /  C_mol1_total
    kon_se      = kon  * (kon_sim_se  / kon_sim)
    koff_se     = koff * (koff_sim_se / koff_sim)

    attrs           = {"cutoff": float(cutoff), "time": time[-1]}
    Pbound_attrs    = {"Ka": float(Ka[-1]), "Ka se": Ka_se, "Ka units": "M-1"}
    Ka              = np.array([tuple(frame) for frame in zip(time, Ka)],
                        np.dtype([("time", "f4"), ("Ka", "f4")]))
    block           = np.array([tuple(frame) for frame in zip(block_duration, Pbound_se, Pbound_se_fit)],
                        np.dtype([("block duration", "f4"), ("Pbound se", "f4"), ("Pbound se fit", "f4")]))
    block_attrs     = {"block duration units": "ns", "minimum asymptote": min_asym, "maximum asymptote": max_asym,
                       "point of inflection": poi, "k": k}
    fpt_attrs       = {"Ka":   kon / koff, "Ka se":   kon / koff * np.sqrt((kon_se / kon) ** 2 + (koff_se / koff) ** 2),
                       "kon":  kon,        "kon se":  kon_se,
                       "koff": koff,       "koff se": koff_se,
                       "Ka units": "M-1",  "kon units": "M-2 ns-1", "koff units": "M-1 ns-1"}

    if verbose:
        _print_two_state(primary.upper(), attrs, Pbound_attrs, fpt_attrs)
    return  [("association/" + primary + "/Pbound/Ka",      Ka),
             ("association/" + primary + "/Pbound/Ka",      {"time units": "ns", "Ka units": "M-1"}),
             ("association/" + primary + "/Pbound/block",   block),
             ("association/" + primary + "/Pbound/block",   block_attrs),
             ("association/" + primary + "/Pbound",         Pbound_attrs),
             ("association/" + primary + "/fpt/on",         fpt_on),
             ("association/" + primary + "/fpt/on",         {"units": "ns"}),
             ("association/" + primary + "/fpt/off",        fpt_off),
             ("association/" + primary + "/fpt/off",        {"units": "ns"}),
             ("association/" + primary + "/fpt",            fpt_attrs),
             ("association/" + primary,                     attrs)]
def _check_two_state(hdf5_file, force = False, **kwargs):
    def _load_association_check():
        hdf5_file.load("*/association_" + primary, loader = _load_association)
        kwargs["volume"]        = np.mean(hdf5_file.data["*/log"]["volume"])
        kwargs["n_molecule_1"]  = hdf5_file.data.pop("n_molecule_1")
        kwargs["n_molecule_2"]  = hdf5_file.data.pop("n_molecule_2")
    primary     = kwargs.get("primary", "comdist")
    cutoff      = kwargs.get("cutoff",  4.5)
    verbose     = kwargs.get("verbose", False)
    expected    = ["association/" + primary + "/Pbound/Ka", "association/" + primary + "/Pbound/block",
                   "association/" + primary + "/fpt/on",    "association/" + primary + "/fpt/off"]

    hdf5_file.load("*/log", type = "table")
    if     (force
    or not (expected in hdf5_file)):
        _load_association_check()
        return [(two_state, kwargs)]

    attrs   = hdf5_file.attrs("association/" + primary)
    if (cutoff                              != attrs["cutoff"]
    or (hdf5_file.data["*/log"]["time"][-1] != attrs["time"])):
        _load_association_check()
        return [(two_state, kwargs)]
    elif verbose:
        Pbound_attrs    = hdf5_file.attrs("association/" + primary + "/Pbound")
        fpt_attrs       = hdf5_file.attrs("association/" + primary + "/fpt")
        _print_two_state(primary.upper(), attrs, Pbound_attrs, fpt_attrs)
    return False
def _print_two_state(pcoord, attrs, Pbound_attrs, fpt_attrs):
    print "DURATION {0:5d} ns CUTOFF {1} < {2:3.1f} A".format(int(attrs["time"]), pcoord, attrs["cutoff"])
    print "Ka          {0:>6.3f} M-1     ".format(Pbound_attrs["Ka"])
    print "Ka       se {0:>6.3f} M-1     ".format(Pbound_attrs["Ka se"])
    print "kon         {0:>6.0f} M-2 ns-1".format(fpt_attrs["kon"])
    print "kon      se {0:>6.0f} M-2 ns-1".format(fpt_attrs["kon se"])
    print "koff        {0:>6.0f} M-1 ns-1".format(fpt_attrs["koff"])
    print "koff     se {0:>6.0f} M-1 ns-1".format(fpt_attrs["koff se"])
    print "kon/koff    {0:>6.3f} M-1     ".format(fpt_attrs["Ka"])
    print "kon/koff se {0:>6.3f} M-1     ".format(fpt_attrs["Ka se"])


def pmf(hdf5_file, 
        pcoord      = "comdist",                            # Progress coordinate
        bins        = np.linspace(0.0, 10.0, 11),           # Bins
        boltzmann   = 0.0019872041,                         # Boltzmann's constat
        temperature = 298.0,                                # System temperature
        zero_point  = None,                                 # Point at which to zero energy
        verbose     = False, n_cores = 1, **kwargs):
    """ Calculates potential of mean force along <pcoord> by splitting into <bins> and using <boltzmann> and
        <temperature>. Sets energy to zero at <zero point> """
    time            = hdf5_file.data["*/log"]["time"]
    count           = hdf5_file.data["*/association_" + pcoord + "_pmf"]

    centers         = (bins[:-1] + bins[1:]) / 2.0
    P               = np.array(count, dtype = np.float32) / np.sum(count)
    P[P == 0.0]     = np.nan
    P_adjusted      = P / (centers ** 2.0)                  # Adjust by r^2
    P_adjusted     /= np.nansum(P_adjusted)                 # Normalize
    pmf_final       = np.log(P_adjusted) * -1 * boltzmann * temperature
    if zero_point:
        pmf_final  -= pmf_final[np.abs(centers - zero_point).argmin()]

    pmf_data        = np.array([tuple(frame) for frame in zip(bins[:-1], bins[1:], count, P, P_adjusted, pmf_final)],
                        np.dtype([("lower bound", "f4"), ("upper bound",          "f4"), ("count", "f4"),
                                  ("probability", "f4"), ("adjusted probability", "f4"), ("pmf",   "f4")]))
    pmf_attrs       = {"lower bound units": "A", "upper bound units": "A", "pmf units": "kcal mol-1",
                       "boltzmann": boltzmann, "temperature": temperature, "zero point": zero_point, "time": time[-1]}
    if verbose:
        print "DURATION {0:5d} ns".format(int(pmf_attrs["time"]))
    return  [("association/" + pcoord + "/pmf", pmf_data),
             ("association/" + pcoord + "/pmf", pmf_attrs)]
def _check_pmf(hdf5_file, force = False, **kwargs):
    pcoord      = kwargs.get("pcoord",      "comdist")
    bins        = kwargs.get("bins",        np.linspace(0.0, 10.0, 11))
    boltzmann   = kwargs.get("boltzmann",   0.0019872041)
    temperature = kwargs.get("temperature", 298.0)
    zero_point  = kwargs.get("zero_point",  None)
    verbose     = kwargs.get("verbose",     False)
    expected    = "association/" + pcoord + "/pmf"

    hdf5_file.load("*/log", type = "table")
    if     (force
    or not (expected in hdf5_file)):
        hdf5_file.load("*/association_" + pcoord, destination = "*/association_" + pcoord + "_pmf",
          loader = _load_association_pmf, bins = bins)
        return [(pmf, kwargs)]

    pmf_data    = hdf5_file[expected]
    pmf_attrs   = hdf5_file.attrs(expected)
    if (boltzmann                               != pmf_attrs["boltzmann"]
    or (temperature                             != pmf_attrs["temperature"])
    or (zero_point                              != pmf_attrs["zero point"])
    or (bins.size                               != pmf_data["lower bound"].size + 1)
    or (np.any(np.array(bins[:-1], np.float32)  != pmf_data["lower bound"]))
    or (np.any(np.array(bins[1:],  np.float32)  != pmf_data["upper bound"]))
    or (hdf5_file.data["*/log"]["time"][-1]     != pmf_attrs["time"])):
        hdf5_file.load("*/association_" + pcoord, destination = "*/association_" + pcoord + "_pmf",
          loader = _load_association_pmf, bins = bins)
        return [(pmf, kwargs)]
    elif verbose:
        print "DURATION {0:5d} ns".format(int(pmf_attrs["time"]))
    return False
