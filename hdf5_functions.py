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

class HDF5_File:
    def __init__(self, filename):
        self.file       = h5py.File(filename)
        self.data       = {}
        self._hierarchy()
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        self.file.flush()
        self.file.close()
    def __contains__(self, items):
        if type(items) == list:
            for item in list(items):
                if not item in self.hierarchy:  return False
        elif not (items in self.hierarchy):     return False
        return True

    def _hierarchy(self):
        hierarchy = {}
        def get_hierarchy(x, y): hierarchy[x] = y
        self.file.visititems(get_hierarchy)
        self.hierarchy  = sorted([str(k) for k in hierarchy.keys()])
    def _segments(self):    return sorted([s for s in self.hierarchy if is_num(s)])
    def _load_complete(self, index, **kwargs):
        processor   = kwargs.get("processor", self._process_default)
        return  processor(np.array(eval("self.file{0}[...])".format(index))))
    def _load_split(self, index, **kwargs):
        shaper          = kwargs.get("shaper",        self._shape_default)
        processor       = kwargs.get("processor",     self._process_default)
        postprocessor   = kwargs.get("postprocessor", self._postprocess_default)
        shapes          = np.array([eval("self.file['{0}']{1}.shape".format(seg, index)) for seg in self._segments()])
        total_shape     = shaper(shapes)
        data            = np.zeros(total_shape)
        i               = 0
        for segment in self._segments():
            new_data                    = processor(eval("np.array(self.file['{0}']{1}[...])".format(segment, index)))
            data[i:i+new_data.shape[0]] = new_data
            i                          += new_data.shape[0]
        return postprocessor(data)
    def _path_to_index(self, address):      return "[\'{0}\']".format('\'][\''.join(address.strip('/').split('/')))
    def _shape_default(self, shapes):
        if shapes.shape[1] == 1:            return np.sum(shapes)
        else:                               return np.array([np.sum(shapes, axis = 0)[0]] +  list(shapes[0,1:]))
    def _process_default(self, new_data):   return new_data
    def _postprocess_default(self, data):   return data

    def add(self, path, data):
        path    = [p for p in path.split('/') if p != '']
        name    = path.pop()
        group   = self.file
        for subgroup in path:
            if   (subgroup in dict(group)): group = group[subgroup]
            else:                           group = group.create_group(subgroup)
        if (type(data) == dict):
            for key, value in data.iteritems():
                try:        group[name].attrs[key]              = value
                except:     group.create_group(name).attrs[key] = value
        else:
            if (name in dict(group)): del group[name]
            group.create_dataset(name, data = data, compression = 'gzip')
            print "    '{0}' added".format('/'.join(path + [name]))
    def load(self, path, **kwargs):
        if path.startswith('*'):    self.data[path] = self._load_split(self._path_to_index(path[2:]), **kwargs)
        else:                       self.data[path] = self._load_complete(self._path_to_index(path),  **kwargs)
#################################################### HDF5 FUNCTIONS ####################################################



