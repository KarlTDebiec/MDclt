#!/usr/bin/python

desc = """analysis_com.py
    Calculates amino acid analogue association constants and rates
    Written by Karl Debiec on 12-08-15
    Last updated 13-02-03"""

########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import h5py
import numpy as np
from scipy.optimize import curve_fit
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
#################################################### HDF5 FUNCTIONS ####################################################
def process_default(new_data):  return new_data
def process_mindist(new_data):  return np.min(new_data, axis = 2)
def shape_default(shapes):
    if shapes.shape[1] == 1:    return np.sum(shapes)
    else:                       return np.array([np.sum(shapes, axis = 0)[0]] +  list(shapes[0,1:]))
def shape_mindist(shapes):      return np.array([np.sum(shapes, axis = 0)[0], shapes[0,1]])
def path_to_index(address):     return "[\'{0}\']".format('\'][\''.join(address.strip('/').split('/')))
def process_hdf5_data(hdf5_file, task_list):
    data        = {}
    hdf5_file   = h5py.File(hdf5_file)
    jobsteps    = sorted([j for j in hdf5_file if is_num(j)])
    try:
        complete_datasets   = [task for task in task_list.keys() if task[0] != '*']
        split_datasets      = [task for task in task_list.keys() if task[0] == '*']
        for complete_dataset in complete_datasets:
            data[complete_dataset]  = np.array(eval("hdf5_file{0}".format(path_to_index(complete_dataset))))
        for split_dataset in split_datasets:
            path        = path_to_index(split_dataset[2:])
            if    task_list[split_dataset] != '':   processor, shaper   = task_list[split_dataset]
            else:                                   processor, shaper   = process_default, shape_default
            shapes      = np.array([eval("hdf5_file['{0}']{1}.shape".format(jobstep, path)) for jobstep in jobsteps])
            total_shape = shaper(shapes)
            data[split_dataset] = np.zeros(total_shape)
            i           = 0
            for jobstep in jobsteps:
                new_data    = processor(np.array(eval("hdf5_file['{0}']{1}".format(jobstep, path))))
                data[split_dataset][i:i + new_data.shape[0]]    =  new_data
                i          += new_data.shape[0]
        for key in hdf5_file.attrs: del hdf5_file.attrs[key]
    finally:
        hdf5_file.flush()
        hdf5_file.close()
        return data
def add_data(hdf5_file, new_data):
    hdf5_file   = h5py.File(hdf5_file)
    try:
        for item in new_data:
            address, item   = item
            address         = [a for a in address.split('/') if a != '']
            name            = address.pop()
            group           = hdf5_file
            for subgroup in address:
                if   (subgroup in dict(group)): group = group[subgroup]
                else:                           group = group.create_group(subgroup)
            if (type(item) == dict):
                for key, value in item.iteritems(): group[name].attrs[key] = value
            else:
                if (name in dict(group)): del group[name]
                try:    group.create_dataset(name, data = item, compression = 'gzip')
                except: raise
    finally:
        hdf5_file.flush()
        hdf5_file.close()
################################################## SPECIAL FUNCTIONS ###################################################
def concentration(side_length, n_alg1, n_alg2):
    return (n_alg1 / 6.0221415e23) / ((side_length / 1e9) ** 3), (n_alg2 / 6.0221415e23) / ((side_length / 1e9) ** 3)
def P_bound_to_Ka(P_bound, side_length, n_alg1, n_alg2):
    alg1_total, alg2_total    = concentration(side_length, n_alg1, n_alg2)
    gnd_act                 = P_bound * np.min(alg1_total, alg2_total)
    act_unbound             = alg1_total - gnd_act
    gnd_unbound             = alg2_total - gnd_act
    return                    gnd_act / (act_unbound * gnd_unbound)
def P_bound_SE_to_Ka_SE(P_bound, side_length, n_alg1, n_alg2, P_bound_SE):
    alg1_total, alg2_total    = concentration(side_length, n_alg1, n_alg2)
    return                    np.sqrt((((alg2_total - alg1_total * P_bound ** 2) * P_bound_SE) /
                                      ((P_bound - 1) ** 2 * (alg1_total * P_bound - alg2_total) ** 2)) ** 2)
def block(data):
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
def fit_sigmoid(x, y):
    def model_function(x, min_asym, max_asym, poi, k):  return max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    min_asym, max_asym, poi, k  = curve_fit(model_function, x, y)[0]
    y_fit                       = max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    return min_asym, max_asym, poi, k, y_fit

def calculate_fpt(time, com_dist, side_length, n_alg1, n_alg2, cutoff):
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

def calculate_P_bound(time, com_dist, side_length, n_alg1, n_alg2, cutoff):
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

def calculate(hdf5_file, side_length, n_alg1, n_alg2, cutoff):
    """ Calculates FPT, Pbound, Ka, Kon, Koff, and associated errors """
    print hdf5_file
    data        = process_hdf5_data(hdf5_file, {'*/time': '', '*/association_com': (process_mindist, shape_mindist)})
    time        = data['*/time']
    com_dist    = data['*/association_com']
    print time.shape, com_dist.shape
    add_data(hdf5_file, calculate_P_bound(time, com_dist, side_length, n_alg1, n_alg2, cutoff))
    add_data(hdf5_file, calculate_fpt(time, com_dist, side_length, n_alg1, n_alg2, cutoff))

######################################################### MAIN #########################################################
if __name__ == '__main__':

    hdf5_folder = os.getcwd() + "/hdf5/"

#    calculate(hdf5_folder + '/AMBER99SBILDN_TIP3P.hdf5',                    56.5276184082031, 2, 100, 4.5)
#    calculate(hdf5_folder + '/AMBER99SBILDN_TIP4PEW.hdf5',                  56.7948532104492, 2, 100, 4.5)
#    calculate(hdf5_folder + '/ANTECHAMBER1_TIP3P.hdf5',                     56.5193328857422, 2, 100, 4.5)
#    calculate(hdf5_folder + '/ANTECHAMBER2_TIP3P.hdf5',                     56.5453300476074, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM22DER2_SPCE.hdf5',                      56.7451362609863, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM22DER2_TIP4PEW.hdf5',                   56.7719192504883, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM22DER2_TIPS3P.hdf5',                    56.6824417114258, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM22STAR_SPCE.hdf5',                      56.7521057128906, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM22STAR_TIP4PEW.hdf5',                   56.7406463623047, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM22STAR_TIPS3P.hdf5',                    56.6874084472656, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM27_SPCE.hdf5',                          56.7388916015625, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM27_TIP4PEW.hdf5',                       56.7892761230469, 2, 100, 4.5)
#    calculate(hdf5_folder + '/CHARMM27_TIPS3P.hdf5',                        56.6779098510742, 2, 100, 4.5)
    calculate(hdf5_folder + '/desmond/CHARMM22STAR_TIPS3P_GSE_SINGLE.hdf5', 56.6874084472656, 2, 100, 4.5)
    calculate(hdf5_folder + '/desmond/CHARMM22STAR_TIPS3P_PME_SINGLE.hdf5', 56.6874084472656, 2, 100, 4.5)

