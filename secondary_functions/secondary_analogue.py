#!/usr/bin/python

desc = """secondary_analogue.py
    Functions for analysis of amino acid analogue datasets
    Written by Karl Debiec on 12-08-15
    Last updated 13-02-07"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   scipy.optimize import curve_fit
from   hdf5_functions import shape_default, process_default, postprocess_default
################################################## INTERNAL FUNCTIONS ##################################################
def _concentration(side_length, n): return (n / 6.0221415e23) / ((side_length / 1e9) ** 3)
def _P_bound_to_Ka(P_bound, C_alg1_total, C_alg2_total):
    C_gnd_act       = P_bound * np.min(C_alg1_total, C_alg2_total)
    C_act_unbound   = C_alg1_total - C_gnd_act
    C_gnd_unbound   = C_alg2_total - C_gnd_act
    return            C_gnd_act / (C_act_unbound * C_gnd_unbound)
def _P_bound_SE_to_Ka_SE(P_bound, C_alg1_total, C_alg2_total, P_bound_SE):
    return np.sqrt((((C_alg2_total - C_alg1_total * P_bound ** 2) * P_bound_SE) /
                    ((P_bound - 1) ** 2 * (C_alg1_total * P_bound - C_alg2_total) ** 2)) ** 2)
def _block(data):
    size            = data.size
    lengths         = np.array(sorted(list(set([size / x for x in range(1, size)]))), dtype = np.int)[:-1]
    SDs             = np.zeros(lengths.size)
    n_blocks        = size // lengths
    for i, length in enumerate(lengths):
        resized     = np.resize(data, (size // length, length))
        means       = np.mean(resized, axis = 1)
        SDs[i]      = np.std(means)
    SEs             = SDs / np.sqrt(n_blocks)
    return lengths, SEs
def _fit_sigmoid(x, y):
    def model_function(x, min_asym, max_asym, poi, k):  return max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    min_asym, max_asym, poi, k  = curve_fit(model_function, x, y)[0]
    y_fit                       = max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    return min_asym, max_asym, poi, k, y_fit
################################################ PRIMARY DATA FUNCTIONS ################################################
def shape_association(shapes):      return np.array([np.sum(shapes, axis = 0)[0], shapes[0,1]])
def process_association(new_data):  return np.min(new_data, axis = 2)
################################################## ANALYSIS FUNCTIONS ##################################################
def association_com(primary_data, arguments, n_cores):
    side_length, n_alg1, n_alg2, cutoff = arguments
    time            = primary_data['*/time']
    com_dist        = primary_data['*/association_com']
    C_alg1_total    = _concentration(side_length, n_alg1)
    C_alg2_total    = _concentration(side_length, n_alg2)
    bound                       = np.zeros(com_dist.shape)
    bound[com_dist < cutoff]    = 1
    P_bound                     = np.zeros(time.size)
    total_bound                 = np.sum(bound, axis = 1)
    for length in range(1, time.size):
        P_bound[length]         = (P_bound[length-1] + total_bound[length])
    P_bound                    /= np.arange(n_alg1, bound.size + 1, n_alg1)
    Ka                          = _P_bound_to_Ka(P_bound, C_alg1_total, C_alg2_total)
    block_lengths, block_SEs    = _block(total_bound / n_alg1)
    min_asym, max_asym, poi, k, block_SEs_fit   = _fit_sigmoid(block_lengths, block_SEs)
    Ka_SE                       = _P_bound_SE_to_Ka_SE(P_bound[-1], C_alg1_total, C_alg2_total, max_asym)
    print Ka[-1], Ka_SE
    formed                      = np.zeros(n_alg1)
    starts                      = [[] for i in range(n_alg1)]
    ends                        = [[] for i in range(n_alg2)]
    for i, frame in enumerate(time):
        for j in range(n_alg1):
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
    fpt_on      = np.concatenate([starts[i][1:] - ends[i][:-1] for i in range(n_alg1)])
    fpt_off     = np.concatenate([ends[i]       - starts[i]    for i in range(n_alg1)])
    kon_sim     = 1 / np.mean(fpt_on)
    koff_sim    = 1 / np.mean(fpt_off)
    kon_sim_SE  = (kon_sim  ** 2 * np.std(fpt_on))  / np.sqrt(fpt_on.size)
    koff_sim_SE = (koff_sim ** 2 * np.std(fpt_off)) / np.sqrt(fpt_off.size)
    kon         = kon_sim  / (C_alg1_total * C_alg2_total)
    koff        = koff_sim /  C_alg1_total
    kon_SE      = kon  * (kon_sim_SE  / kon_sim)
    koff_SE     = koff * (koff_sim_SE / koff_sim)
    print kon, kon_SE, koff, koff_SE, kon / koff
    return  [("/association_com/P_bound/Ka",                np.array([Ka[-1], Ka_SE])),
             ("/association_com/P_bound/Ka",                {'units': 'M-1'}),
             ("/association_com/P_bound/Ka_convergence",    Ka),
             ("/association_com/P_bound/Ka_convergence",    {'units': 'M-1'}),
             ("/association_com/P_bound/time",              time),
             ("/association_com/P_bound/time",              {'units': 'ns'}),
             ("/association_com/P_bound/block_length",      block_lengths),
             ("/association_com/P_bound/block_SE",          block_SEs),
             ("/association_com/P_bound/block_SE_fit",      block_SEs_fit),
             ("/association_com/P_bound/block_SE_fit",      {'min_asymptote': min_asym, 'max_asymptote': max_asym,
                                                             'poi':           poi,      'k':             k}),
             ("/association_com/P_bound",                   {'time': time[-1]}),
             ("/association_com/fpt/kon",                   np.array([kon,  kon_SE])),
             ("/association_com/fpt/kon",                   {'units': 'M-2 ns-1'}),
             ("/association_com/fpt/koff",                  np.array([koff, koff_SE])),
             ("/association_com/fpt/koff",                  {'units': 'M-1 ns-1'}),
             ("/association_com/fpt/fpt_on",                fpt_on),
             ("/association_com/fpt/fpt_on",                {'units': 'ns'}),
             ("/association_com/fpt/fpt_off",               fpt_off),
             ("/association_com/fpt/fpt_off",               {'units': 'ns'}),
             ("/association_com/fpt",                       {'time': time[-1]})]
def path_association_com():
    return [('*/time',              (shape_default,     process_default,     postprocess_default)),
            ('*/association_com',   (shape_association, process_association, postprocess_default))]
def check_association_com(arguments):
    return [(association_com, arguments)]

