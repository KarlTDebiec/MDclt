#!/usr/bin/python
desc = """hdf5_functions.py
    Class for simplifying interaction with HDF5 files
    Written by Karl Debiec on 13-02-03
    Last updated 13-05-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import h5py
import numpy as np
from   collections import OrderedDict
from   standard_functions import is_num, Function_to_Method_Wrapper
######################################################## CLASS #########################################################
class HDF5_File:
    def __init__(self, filename):
        self.file       = h5py.File(filename)
        self.data       = {}
        self._hierarchy()
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        del self.hierarchy
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
    def _hierarchy(self):
        hierarchy = {}
        def get_hierarchy(x, y): hierarchy[x] = y
        self.file.visititems(get_hierarchy)
        self.hierarchy  = OrderedDict(sorted(hierarchy.items(), key=lambda item: item[0]))
    def _strip_path(self, path):    return path.replace("//", "/").strip("/")
    def _segments(self):            return sorted([s for s in self.hierarchy if is_num(s)])
    def _load_complete(self, path, **kwargs):
        processor   = kwargs.get("processor", self._process_default)
        return  processor(self[path])
    def _load_split(self, path, **kwargs):
        if ("shaper"        in kwargs): shaper          = Function_to_Method_Wrapper(self, kwargs.get("shaper"))
        else:                           shaper          = self._shape_default
        if ("processor"     in kwargs): processor       = Function_to_Method_Wrapper(self, kwargs.get("processor"))
        else:                           processor       = self._process_default
        if ("postprocessor" in kwargs): postprocessor   = Function_to_Method_Wrapper(self, kwargs.get("postprocessor"))
        else:                           postprocessor   = self._postprocess_default
        shapes          = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in self._segments()])
        total_shape     = shaper(shapes)
        data            = np.zeros(total_shape)
        i               = 0
        for segment in self._segments():
            new_data                    = processor(self[segment + "/" + path[2:]])
            data[i:i+new_data.shape[0]] = new_data
            i                          += new_data.shape[0]
        return postprocessor(data)
    def _load_split_table(self, path, **kwargs):
        dtype   = self.hierarchy[self._segments()[0] + "/" + path[2:]].dtype

        shapes  = np.array([self.hierarchy[segment + "/" + path[2:]].shape[0] for segment in self._segments()])
        data    = np.zeros((np.sum(shapes)), dtype)
        i       = 0
        for segment in self._segments():
            new_data                    = np.array(self.hierarchy[segment + "/" + path[2:]], dtype)
            data[i:i+new_data.size]     = new_data
            i                          += new_data.shape[0]
        return data
    def _shape_default(self, shapes):
        if shapes.shape[1] == 1:            return np.sum(shapes)
        else:                               return np.array([np.sum(shapes, axis = 0)[0]] +  list(shapes[0,1:]))
    def _process_default(self, new_data):   return new_data
    def _postprocess_default(self, data):   return data

    def add(self, path, data, **kwargs):
        verbose     = kwargs.get("verbose",     True)
        data_kwargs = kwargs.get("data_kwargs", {"compression": "gzip"})
        path        = self._strip_path(path).split("/")
        name        = path.pop()
        group       = self.file
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
    def load(self, path, **kwargs):
        type    =  kwargs.get("type", "array")
        if   type == "array":
            if path.startswith("*"):    self.data[path] = self._load_split(self._strip_path(path),          **kwargs)
            else:                       self.data[path] = self._load_complete(self._strip_path(path),       **kwargs)
        elif type == "table":
            if path.startswith("*"):    self.data[path] = self._load_split_table(self._strip_path(path),    **kwargs)
            else:                       raise Exception("self_load_complete_table() is not yet implemented")
    def attrs(self, path, **kwargs):
        return  dict(self.hierarchy[self._strip_path(path)].attrs)
