#!/usr/bin/python

desc = """analogue.py
    Functions for analysis of amino acid analogue datasets
    Written by Karl Debiec on 12-08-15
    Last updated 13-02-03"""

########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   scipy.optimize import curve_fit
from   hdf5_functions import *
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
################################################## INTERNAL FUNCTIONS ##################################################
def _concentration(side_length, n_alg1, n_alg2):
    return (n_alg1 / 6.0221415e23) / ((side_length / 1e9) ** 3), (n_alg2 / 6.0221415e23) / ((side_length / 1e9) ** 3)
def _P_bound_to_Ka(P_bound, side_length, n_alg1, n_alg2):
    alg1_total, alg2_total    = concentration(side_length, n_alg1, n_alg2)
    gnd_act                 = P_bound * np.min(alg1_total, alg2_total)
    act_unbound             = alg1_total - gnd_act
    gnd_unbound             = alg2_total - gnd_act
    return                    gnd_act / (act_unbound * gnd_unbound)
def _P_bound_SE_to_Ka_SE(P_bound, side_length, n_alg1, n_alg2, P_bound_SE):
    alg1_total, alg2_total    = concentration(side_length, n_alg1, n_alg2)
    return                    np.sqrt((((alg2_total - alg1_total * P_bound ** 2) * P_bound_SE) /
                                      ((P_bound - 1) ** 2 * (alg1_total * P_bound - alg2_total) ** 2)) ** 2)
def _block(data):
    """ Applies the blocking method of calculating standard error to a 1D array """
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
################################################## ANALYSIS FUNCTIONS ##################################################
def fpt(time, com_dist, side_length, n_alg1, n_alg2, cutoff):
    alg1_total, alg2_total      = concentration(side_length, n_alg1, n_alg2)
    bound                       = np.zeros(com_dist.shape)
    bound[com_dist < cutoff]    = 1
    formed                      = np.zeros(n_alg1)
    start                       = [[] for i in range(n_alg1)]
    end                         = [[] for i in range(n_alg2)]
    for frame in np.column_stack((time, bound)):
        time    = frame[0]
        bound   = frame[1:]
        for i in range(bound.size):
            if not formed[i] and bound[i]:
                formed[i]   = 1
                start[i]   += [time]
            elif formed[i] and not bound[i]:
                formed[i]   = 0
                end[i]     += [time]
    for i, s in enumerate(start):   start[i]    = np.array(s)
    for i, e in enumerate(end):     end[i]      = np.array(e)
    for i, f in enumerate(formed):
        if f:   start[i] = start[i][:-1]
    fpt_on      = np.concatenate([start[i][1:] - end[i][:-1] for i in range(n_alg1)])
    fpt_off     = np.concatenate([end[i]       - start[i]    for i in range(n_alg1)])
    kon_sim     =  1 / np.mean(fpt_on)
    koff_sim    =  1 / np.mean(fpt_off)
    kon_sim_SE  = (kon_sim  ** 2 * np.std(fpt_on))  / np.sqrt(fpt_on.size)
    koff_sim_SE = (koff_sim ** 2 * np.std(fpt_off)) / np.sqrt(fpt_off.size)
    kon         = kon_sim  / (alg1_total * alg2_total)
    koff        = koff_sim / alg1_total
    kon_SE      = kon  * (kon_sim_SE  / kon_sim)
    koff_SE     = koff * (koff_sim_SE / koff_sim)
    print kon,      kon_SE,     koff,       koff_SE,        kon / koff
    return  [("/association_com/fpt/kon",     np.array([kon,  kon_SE])),
             ("/association_com/fpt/kon",     {'units': 'M-2 ns-1'}),
             ("/association_com/fpt/koff",    np.array([koff, koff_SE])),
             ("/association_com/fpt/koff",    {'units': 'M-1 ns-1'}),
             ("/association_com/fpt/fpt_on",  fpt_on),
             ("/association_com/fpt/fpt_on",  {'units': 'ns'}),
             ("/association_com/fpt/fpt_off", fpt_off),
             ("/association_com/fpt/fpt_off", {'units': 'ns'}),
             ("/association_com/fpt",         {'time': time})]

def P_bound(time, com_dist, side_length, n_alg1, n_alg2, cutoff):
    bound                       = np.zeros(com_dist.shape)
    bound[com_dist < cutoff]    = 1
    P_bound_convergence         = np.zeros(com_dist.shape[0])
    total_bound                 = np.sum(bound, axis = 1)
    for length in range(1, bound.shape[0]):
        P_bound_convergence[length]             = (P_bound_convergence[length-1] + total_bound[length])
    P_bound_convergence                        /= np.arange(bound.shape[1], bound.size + 1, bound.shape[1])
    Ka_convergence                              = P_bound_to_Ka(P_bound_convergence, side_length, n_alg1, n_alg2)
    block_lengths, block_SEs                    = block(total_bound / bound.shape[1])
    min_asym, max_asym, poi, k, block_SEs_fit   = fit_sigmoid(block_lengths, block_SEs)
    Ka                                          = Ka_convergence[-1]
    Ka_SE                                       = P_bound_SE_to_Ka_SE(P_bound_convergence[-1], side_length,
                                                                      n_alg1, n_alg2, max_asym)
    print Ka, Ka_SE
    return  [("/association_com/P_bound/Ka",                np.array([Ka, Ka_SE])),
             ("/association_com/P_bound/Ka",                {'units': 'M-1'}),
             ("/association_com/P_bound/Ka_convergence",    Ka_convergence),
             ("/association_com/P_bound/Ka_convergence",    {'units': 'M-1'}),
             ("/association_com/P_bound/time",              time),
             ("/association_com/P_bound/time",              {'units': 'ns'}),
             ("/association_com/P_bound/block_length",      block_lengths),
             ("/association_com/P_bound/block_SE",          block_SEs),
             ("/association_com/P_bound/block_SE_fit",      block_SEs_fit),
             ("/association_com/P_bound/block_SE_fit",      {'min_asymptote': min_asym, 'max_asymptote': max_asym,
                                                             'poi':           poi,      'k':             k}),
             ("/association_com/P_bound",                   {'time': time[-1]})]

def pmf(state, bins, T):
    k                   = 0.0019872041
    hist, bins          = np.histogram(state, bins)
    centers             = (bins[:-1] + bins[1:]) / 2.
    hist                = np.array(hist, dtype = np.float)    
    hist               /= np.size(state)
    hist               /= centers
    hist               /= centers
    hist[hist == 0.]    = 'nan'
    pmf                 = np.log(hist) * -1 * k * T
    pmf                -= pmf[250]
    return pmf, centers













def com(arguments):
    """ Calculates center of mass distance between two residue types; assumes cubic box and pbc """
    jobstep, topology, trajectory, alg1, alg2 = arguments

    return  [("/" + jobstep + "/association_com",   distances),
             ("/" + jobstep + "/association_com",   {'units': "A"})]
def check_com(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    alg1, alg2                          = arguments
    if not os.path.isfile(topology):    return False
    if not os.path.isfile(trajectory):  return False
    if not jobstep + "/association_com" in hierarchy:   return [(com, (jobstep, topology, trajectory, alg1, alg2))]
    else:                                               return False







