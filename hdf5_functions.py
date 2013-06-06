#!/usr/bin/python
desc = """hdf5_functions.py
    Class for simplifying interaction with HDF5 files
    Written by Karl Debiec on 13-02-03
    Last updated 13-06-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import h5py
import numpy as np
from   collections import OrderedDict
from   standard_functions import is_num, Function_to_Method_Wrapper
######################################################## CLASS #########################################################
class HDF5_File:
    def __init__(self, filename):
        self.filename   = os.path.abspath(filename)
        self.file       = None
        self.data       = {}
        try:              self._quick_hierarchy()
        except:           self._open_file()
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        del self.hierarchy
        if  self.file:
            self.file.flush()
            self.file.close()
    def __contains__(self, items):
        if type(items) == list:
            for item in map(self._strip_path, items):
                if not item in self.hierarchy:                  return False
        elif not (self._strip_path(items) in self.hierarchy):   return False
        return True
    def __getitem__(self, path):
        if self._strip_path(path) in self:
            return np.array(self.hierarchy[self._strip_path(path)][...])
        else:
            raise Exception("Inappropriate input to HDF5_File.__getitem__(path):", self._strip_path(path))
    def _open_file(self):
        self.file   = h5py.File(self.filename)
        self._hierarchy()
    def _quick_hierarchy(self):
        command         = "h5ls -r {0} | cut -b2- | awk '{{print $1}}'".format(self.filename)
        self.hierarchy  = set(commands.getoutput(command).split())
    def _hierarchy(self):
        hierarchy = {}
        def get_hierarchy(x, y): hierarchy[x] = y
        self.file.visititems(get_hierarchy)
        self.hierarchy  = hierarchy
    def _strip_path(self, path):    return path.replace("//", "/").strip("/")
    def _segments(self):            return sorted([s for s in self.hierarchy if is_num(s)])
    def _load_split_array(self, path, **kwargs):
        shapes      = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in self._segments()])
        if shapes.shape[1] == 1:
            shape   = np.sum(shapes)
        else:
            shape   = np.array([np.sum(shapes, axis = 0)[0]] + list(shapes[0,1:]))
        data        = np.zeros(shape, np.float32)
        i           = 0
        for segment in self._segments():
            new_data                    = self[segment + "/" + path[2:]]
            data[i:i+new_data.shape[0]] = new_data
            i                          += new_data.shape[0]
        return data
    def _load_whole_array(self, path, **kwargs):
        return self[path]
    def _load_split_table(self, path, **kwargs):
        dtype   = self.hierarchy[self._segments()[0] + "/" + path[2:]].dtype
        shape   = np.sum(np.array([self.hierarchy[segment + "/" + path[2:]].shape[0] for segment in self._segments()]))
        data    = np.zeros(shape, dtype)
        i       = 0
        for segment in self._segments():
            new_data                    = np.array(self.hierarchy[segment + "/" + path[2:]], dtype)
            data[i:i+new_data.size]     = new_data
            i                          += new_data.shape[0]
        return data
    def _load_whole_table(self, path, **kwargs):
        return np.array(self[path], self[path].dtype)

    def add(self, path, data, data_kwargs = {"compression": "lzf"}, verbose = True, **kwargs):
        if not self.file:   self._open_file()
        path    = self._strip_path(path).split("/")
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
            group.create_dataset(name, data = data, **data_kwargs)
            if verbose:
                print "    {0:25} added".format("/".join(path + [name]))
    def load(self, path, type = "array", **kwargs):
        if not self.file: self._open_file()
        if   "loader" in kwargs:                       loader = Function_to_Method_Wrapper(self, kwargs.get("loader"))
        elif type == "array" and path.startswith("*"): loader = self._load_split_array
        elif type == "array":                          loader = self._load_whole_array
        elif type == "table" and path.startswith("*"): loader = self._load_split_table
        elif type == "table":                          loader = self._load_whole_table
        self.data[path] = loader(self._strip_path(path), **kwargs)
    def attrs(self, path, **kwargs):
        if not self.file: self._open_file()
        return dict(self.hierarchy[self._strip_path(path)].attrs)
