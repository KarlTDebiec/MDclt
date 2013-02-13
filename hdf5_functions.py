#!/usr/bin/python

desc = """hdf5_functions.py
    Functions for transferring data in and out of HDF5 files
    Written by Karl Debiec on 13-02-03
    Last updated 13-02-08"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import h5py
import numpy as np
from   standard_functions import is_num
#################################################### HDF5 FUNCTIONS ####################################################
def get_hierarchy(hdf5_filename):
    """ Returns sorted list of paths present in <hdf5_filename> """
    hdf5_file = h5py.File(hdf5_filename)
    hierarchy = {}
    def get_hierarchy(x, y): hierarchy[x] = y
    hdf5_file.visititems(get_hierarchy)
    return sorted([str(k) for k in hierarchy.keys()])

def shape_default(shapes):
    if shapes.shape[1] == 1:    return np.sum(shapes)
    else:                       return np.array([np.sum(shapes, axis = 0)[0]] +  list(shapes[0,1:]))
def process_default(new_data):  return new_data
def postprocess_default(data):  return data
def path_to_index(address):     return "[\'{0}\']".format('\'][\''.join(address.strip('/').split('/')))
def load_data(hdf5_file, paths):
    """ Loads data stored at <paths> from <hdf5_file> """
    data        = {}
    segments    = sorted([s for s in hdf5_file if is_num(s)])
    for path, functions in paths:
        if path.startswith('*'): data[path] = load_split_dataset(hdf5_file, segments, path_to_index(path[2:]), functions)
        else:                    data[path] = load_complete_dataset(hdf5_file, path_to_index(path), functions)
    return data
def load_complete_dataset(hdf5_file, index, processor):
    """ Loads data stored at <index> from <hdf5_file> and processes with <processor> """
    return  processor(np.array(eval("hdf5_file{0}[...])".format(index))))
def load_split_dataset(hdf5_file, segments, index, functions):
    """ Loads data stored at <index> from <segments> of <hdf5_file> and processes with <functions> """
    shaper, processor, postprocessor = functions
    shapes          = np.array([eval("hdf5_file['{0}']{1}.shape".format(segment, index)) for segment in segments])
    total_shape     = shaper(shapes)
    data            = np.zeros(total_shape)
    i               = 0
    for segment in segments:
        new_data                    = processor(eval("np.array(hdf5_file['{0}']{1}[...])".format(segment, index)))
        data[i:i+new_data.shape[0]] = new_data
        i                          += new_data.shape[0]
    return postprocessor(data)

def add_data(hdf5_file, path, data):
    """ Adds <data> to <hdf5_file> at <path> """
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


