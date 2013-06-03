#!/usr/bin/python
desc = """association.py
    Functions for secondary analysis of amino acid analogue association
    Written by Karl Debiec on 12-08-15
    Last updated 13-05-12"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import numpy as np
from   scipy.optimize import curve_fit
################################################## INTERNAL FUNCTIONS ##################################################
def _concentration(n, volume): return n / 6.0221415e23 / volume
def _P_bound_to_Ka(P_bound, C_res1_total, C_res2_total):
    C_complex       = P_bound * np.min(C_res1_total, C_res2_total)
    C_res1_unbound  = C_res1_total - C_complex
    C_res2_unbound  = C_res2_total - C_complex
    return            C_complex / (C_res1_unbound * C_res2_unbound)
def _P_bound_se_to_Ka_se(P_bound, C_res1_total, C_res2_total, P_bound_se):
    return np.sqrt((((C_res2_total - C_res1_total * P_bound ** 2) * P_bound_se) /
                    ((P_bound - 1) ** 2 * (C_res1_total * P_bound - C_res2_total) ** 2)) ** 2)
def _block(data):
    full_size       = data.size
    sizes           = np.array(sorted(list(set([full_size / x for x in range(1, full_size)]))), dtype = np.int)[:-1]
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
def _shape_association(self, shapes, **kwargs):
    self.datakwargs["n_residue_1"]  = shapes[0,1]
    self.datakwargs["n_residue_2"]  = shapes[0,2]
    return np.array([np.sum(shapes, axis = 0)[0], shapes[0,1]])
def _process_association(self, data, **kwargs):
    return np.min(data, axis = 2)
################################################## ANALYSIS FUNCTIONS ##################################################
def com(hdf5_file, **kwargs):
    """ Calculates Ka, kon, and koff of <n_residue_1> molecules of type 1 and <n_residue_2> molecules of type 2 in cubic
        box of <side length> with the bound state defined as center of mass distance below <cutoff> Angstrom. Follows
        the protocol of Piana, S., Lindorff-Larsen, K., Shaw, D.E. How Robust Are Protein Folding Simulations with
        Respect to Force Field Parameterization? Biophys J. 2011. 100. L47-L49. Error is estimated using the blocking
        method of Flyvbjerg, H., and Petersen, H. G. Error Estimates on Averages of Correlated Data. J Phys Chem. 1989.
        91. 461-466. """
    verbose         = kwargs.get("verbose",     False)      # Print output to terminal
    cutoff          = kwargs.get("cutoff",      4.5)        # Bound/unbound cutoff (A)
    n_res1          = kwargs.get("n_residue_1", 1)          # Number of residues of type 1
    n_res2          = kwargs.get("n_residue_2", 1)          # Number of residues of type 2
    volume          = kwargs.get("volume")      * 1e-27     # System volume (L)
    time            = hdf5_file.data["*/log"]["time"]
    com_dist        = hdf5_file.data["*/association_com"]
    dt              = time[1] - time[0]
    C_res1_total    = _concentration(n_res1, volume)
    C_res2_total    = _concentration(n_res2, volume)

    bound                       = np.zeros(com_dist.shape)
    bound[com_dist < cutoff]    = 1
    P_bound                     = np.zeros(time.size)
    total_bound                 = np.sum(bound, axis = 1)
    for i in range(1, time.size): P_bound[i] = (P_bound[i-1] + total_bound[i])
    P_bound                    /= np.arange(n_res1, bound.size + 1, n_res1)
    Ka                          = _P_bound_to_Ka(P_bound, C_res1_total, C_res2_total)

    block_size, Pbound_se       = _block(total_bound / n_res1)
    min_asym, max_asym, poi, k, Pbound_se_fit   = _fit_sigmoid(block_size, Pbound_se)
    block_duration              = np.array(block_size, dtype = np.float32) * dt
    poi                        *= dt
    Ka_se                       = _P_bound_se_to_Ka_se(P_bound[-1], C_res1_total, C_res2_total, max_asym)

    formed                      = np.zeros(n_res1)
    starts                      = [[] for i in range(n_res1)]
    ends                        = [[] for i in range(n_res2)]
    for i, frame in enumerate(time):
        for j in range(n_res1):
            if not formed[j] and bound[i,j]:
                formed[j]   = 1
                starts[j]  += [frame]
            elif formed[j] and not bound[i,j]:
                formed[j]   = 0
                ends[j]    += [frame]
    for i, s in enumerate(starts):  starts[i]   = np.array(s)
    for i, e in enumerate(ends):    ends[i]     = np.array(e)
    for i, f in enumerate(formed):
        if f:   starts[i] = starts[i][:-1]
    fpt_on      = np.concatenate([starts[i][1:] - ends[i][:-1] for i in range(n_res1)])
    fpt_off     = np.concatenate([ends[i]       - starts[i]    for i in range(n_res1)])
    kon_sim     = 1 / np.mean(fpt_on)
    koff_sim    = 1 / np.mean(fpt_off)
    kon_sim_se  = (kon_sim  ** 2 * np.std(fpt_on))  / np.sqrt(fpt_on.size)
    koff_sim_se = (koff_sim ** 2 * np.std(fpt_off)) / np.sqrt(fpt_off.size)
    kon         = kon_sim  / (C_res1_total * C_res2_total)
    koff        = koff_sim /  C_res1_total
    kon_se      = kon  * (kon_sim_se  / kon_sim)
    koff_se     = koff * (koff_sim_se / koff_sim)

    Ka              = np.array([tuple(frame) for frame in np.column_stack((time, Ka))],
                        np.dtype([("time", "f"), ("Ka", "f")]))
    Ka_attrs        = {"time":  "ns", "Ka":   "M-1", "Ka se":   Ka_se}
    block           = np.array([tuple(frame) for frame in np.column_stack((block_duration, Pbound_se, Pbound_se_fit))],
                        np.dtype([("block duration", "f"), ("Pbound se", "f"), ("Pbound se fit", "f")]))
    block_attrs     = {"block length": "ns", "minimum asymptote": min_asym, "maximum asymptote": max_asym,
                       "point of inflection": poi, "k": k}
    fpt_attrs       = {"units": "M-1", "Ka":   kon / koff, "Ka se": kon/koff*np.sqrt((kon_se/kon)**2+(koff_se/koff)**2)}
    fpt_on_attrs    = {"units": "ns",  "kon":  kon,        "kon se":  kon_se,  "kon units":  "M-2 ns-1"}
    fpt_off_attrs   = {"units": "ns",  "koff": koff,       "koff se": koff_se, "koff units": "M-1 ns-1"}

    if verbose:
        print "DURATION  {0:5d} ns CUTOFF {1:3.1f} A".format(int(time.size * dt), cutoff)
        print "Ka          {0:>6.3f} M-1     ".format(float(Ka["Ka"][-1]))
        print "Ka se       {0:>6.3f} M-1     ".format(Ka_attrs["Ka se"])
        print "kon         {0:>6.0f} M-2 ns-1".format(fpt_on_attrs["kon"])
        print "kon se      {0:>6.0f} M-2 ns-1".format(fpt_on_attrs["kon se"])
        print "koff        {0:>6.0f} M-1 ns-1".format(fpt_off_attrs["koff"])
        print "koff se     {0:>6.0f} M-1 ns-1".format(fpt_off_attrs["koff se"])
        print "kon/koff    {0:>6.3f} M-1     ".format(fpt_attrs["Ka"])
        print "kon/koff se {0:>6.3f} M-1     ".format(fpt_attrs["Ka se"])
    return  [("association/com/Ka",         Ka),
             ("association/com/Ka",         Ka_attrs),
             ("association/com/block",      block),
             ("association/com/block",      block_attrs),
             ("association/com/fpt/on",     fpt_on),
             ("association/com/fpt/on",     fpt_on_attrs),
             ("association/com/fpt/off",    fpt_off),
             ("association/com/fpt/off",    fpt_off_attrs),
             ("association/com/fpt",        fpt_attrs),
             ("association/com",            {"cutoff": cutoff, "time": time[-1]})]
def _check_com(hdf5_file, force = False, **kwargs):
    def _load_association():
        hdf5_file.load("*/association_com", shaper = _shape_association,  processor = _process_association, **kwargs)
        kwargs["volume"]        = np.mean(hdf5_file.data["*/log"]["volume"])
        kwargs["n_residue_1"]   = hdf5_file.datakwargs.pop("n_residue_1")
        kwargs["n_residue_2"]   = hdf5_file.datakwargs.pop("n_residue_2")
    verbose     = kwargs.get("verbose", False)
    expected    = ["association/com/Ka", "association/com/block", "association/com/fpt/on", "association/com/fpt/off"]

    hdf5_file.load("*/log", type = "table")

    if (force
    or not(expected in hdf5_file)):
        _load_association()
        return [(com, kwargs)]
    attrs   = hdf5_file.attrs("association/com")
    cutoff  = kwargs.get("cutoff",  4.5)

    if (cutoff                              != attrs["cutoff"]
    or  hdf5_file.data["*/log"]["time"][-1] != attrs["time"]):
        _load_association()
        return [(com, kwargs)]
    elif verbose:
        Ka              = hdf5_file["association/com/Ka"]
        Ka_attrs        = hdf5_file.attrs("association/com/Ka")
        fpt_attrs       = hdf5_file.attrs("association/com/fpt")
        fpt_on_attrs    = hdf5_file.attrs("association/com/fpt/on")
        fpt_off_attrs   = hdf5_file.attrs("association/com/fpt/off")
        print "DURATION  {0:5d} ns CUTOFF {1:3.1f} A".format(int(attrs["time"]), float(attrs["cutoff"]))
        print "Ka          {0:>6.3f} M-1     ".format(float(Ka["Ka"][-1]))
        print "Ka se       {0:>6.3f} M-1     ".format(Ka_attrs["Ka se"])
        print "kon         {0:>6.0f} M-2 ns-1".format(fpt_on_attrs["kon"])
        print "kon se      {0:>6.0f} M-2 ns-1".format(fpt_on_attrs["kon se"])
        print "koff        {0:>6.0f} M-1 ns-1".format(fpt_off_attrs["koff"])
        print "koff se     {0:>6.0f} M-1 ns-1".format(fpt_off_attrs["koff se"])
        print "kon/koff    {0:>6.3f} M-1     ".format(fpt_attrs["Ka"])
        print "kon/koff se {0:>6.3f} M-1     ".format(fpt_attrs["Ka se"])
    return False

def _shape_pmf(self, shapes, bins, **kwargs):
    return np.array([shapes.shape[0], bins.size - 1])
def _process_pmf(self, data, bins, **kwargs):
    processed       = np.zeros((1, bins.size -1))
    processed[0, :] = np.histogram(data, bins)[0]
    return processed
def _postprocess_pmf(self, data, **kwargs):
    return  np.sum(data, axis = 0)

def pmf(hdf5_file, **kwargs):
    """ Calculates potential of mean force along a progress coordinate """
    verbose         = kwargs.get("verbose",     False)      # Print output to terminal
    time            = hdf5_file.data["*/log"]["time"]
    com_dist        = hdf5_file.data["*/association_com"]
    dt              = time[1] - time[0]

def _check_pmf(hdf5_file, force = False, **kwargs):
    def _load_association():
        hdf5_file.load("*/association_" + pcoord, shaper = _shape_pmf,  processor = _process_pmf, 
                       postprocessor = _postprocess_pmf, bins = bins, **kwargs)
        print hdf5_file.data["*/association_" + pcoord]
    verbose     = kwargs.get("verbose", False)
    pcoord      = kwargs.get("pcoord",  "com")
    bins        = kwargs.get("bins",    np.linspace(0., 50, 100))
    T           = kwargs.get("T",       298)
    expected    = ["association/" + pcoord + "/pmf"]

    hdf5_file.load("*/log", type = "table")
    if (force
    or not(expected in hdf5_file)):
        _load_association()
#        return [(pmf, kwargs)]
#    pmf     = hdf5_file["association/" + pcoord + "/pmf"]
#    attrs   = hdf5_file.attrs("association/" + pcoord + "/pmf")

#    if (bins                                != pmf["bins"]
#    or  T                                   != attrs["T"]
#    or  hdf5_file.data["*/log"]["time"][-1] != attrs["time"]):
#        _load_association_com()
#        return [(com, kwargs)]
#    elif verbose:
#        print "DURATION  {0:5d} ns CUTOFF {1:3.1f} A".format(int(attrs["time"]), float(attrs["cutoff"]))
#        print "Ka          {0:>6.3f} M-1     ".format(float(Ka["Ka"][-1]))
    return False

