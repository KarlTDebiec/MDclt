#!/usr/bin/python

desc = """hdf5_functions.py
    Functions for transferring data in and out of HDF5 files
    Written by Karl Debiec on 13-02-03
    Last updated 13-02-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import h5py
import numpy as np
#################################################### HDF5 FUNCTIONS ####################################################
def get_hierarchy(hdf5_file):
    hdf5_file = h5py.File(hdf5_file)
    hierarchy = {}
    def get_hierarchy(x, y): hierarchy[x] = y
    hdf5_file.visititems(get_hierarchy)
    hdf5_file.close()
    return sorted([str(k) for k in hierarchy.keys()])

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

def add_data(hdf5_file, path, data):
    path    = [p for p in path.split('/') if p != '']
    name    = path.pop()
    group   = hdf5_file
    for subgroup in path:
        if   (subgroup in dict(group)): group = group[subgroup]
        else:                           group = group.create_group(subgroup)
    if (type(data) == dict):
        for key, value in data.iteritems(): group[name].attrs[key] = value
    else:
        if (name in dict(group)): del group[name]
        group.create_dataset(name, data = data, compression = 'gzip')
        print "    '{0}' added".format('/'.join(path + [name]))


