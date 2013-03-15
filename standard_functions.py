#!/usr/bin/python

desc = """standard_functions.py
    Standard functions
    Written by Karl Debiec on 13-02-03
    Last updated 13-03-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, subprocess, sys
import numpy as np
################################################ SEGMENT LIST FUNCTIONS ################################################
class Segment:
    def __init__(self, number, path, topology, trajectory, files):
        self.number     = number
        self.path       = path
        self.topology   = topology
        self.trajectory = trajectory
        self.files      = files
    def __float__(self):        return float(self.number)
    def __str__(self):          return self.number
    def __repr__(self):         return self.number
    def __add__(self, other):   return str(self.number) + other
    def __radd__(self, other):  return other + str(self.number)
    def file_of_type(self, end):
        temp    = [f for f in self.files if f.endswith(end)]
        if len(temp) == 1:  return temp[0]
        else:               raise
def segments_standard(path):
    """ Lists segment folders, topologies, and trajectories  at <path>, assuming the format ####/####.* """
    segments = []
    for directory in sorted([f for f in os.listdir(path) if is_num(f)]):
        files       = ["{0}/{1}/{2}".format(path, directory, f) for f in os.listdir("{0}/{1}/".format(path, directory))]
        segments   += [Segment(number       = directory,
                               path         = "{0}/{1}/".format(path, directory),
                               topology     = "{0}/{1}/{1}_solute.pdb".format(path, directory),
                               trajectory   = "{0}/{1}/{1}_solute.xtc".format(path, directory),
                               files        = files)]
    return segments
################################################## GENERAL FUNCTIONS ###################################################
def is_num(test):
    try:    float(test)
    except: return False
    return  True
def month(string):
    month = {'jan':  1, 'feb':  2, 'mar':  3, 'apr':  4, 'may':  5, 'jun':  6,
             'jul':  7, 'aug':  8, 'sep':  9, 'oct': 10, 'nov': 11, 'dec': 12}
    try:    return month[string.lower()]
    except: return None
def _string_to_function(module_function):
    module, function = module_function.split('.')
    return getattr(sys.modules[module], function)
def _shell_executor(command):
    pipe    = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    for line in iter(pipe.stdout.readline, ''):
        yield line.rstrip().replace('\n', ' ')
    pipe.wait()
################################################## ANALYSIS FUNCTIONS ##################################################
def _contact_1D_to_2D_map(contact_1D):
    """ Converts a 1D (sparse) contact map <contact_1D> to a 2D (complete) contact map """
    n_res       = int(1 + np.sqrt(1 + 8 * contact_1D.size)) / 2
    indexes     = contact_1D_to_2D_indexes(n_res)
    contact_2D  = np.zeros((n_res, n_res), dtype = np.int8)
    contact_2D[indexes[:,0], indexes[:,1]]  = contact_1D
    contact_2D[indexes[:,1], indexes[:,0]]  = contact_1D
    contact_2D[range(n_res), range(n_res)]  = 1
    return contact_2D
def _contact_2D_to_1D_indexes(n_res):
    """ Generates indexes for conversion of 2D (complete) contact map to a 1D (sparse) contact map of <n_res> """
    indexes = np.zeros((n_res,n_res), dtype = np.int)
    i       = 0
    for j in range(n_res-1):
        for k in range(j+1,n_res):
            indexes[j, k]   = i
            i              += 1
    return indexes
def _contact_1D_to_2D_indexes(n_res):
    """ Generates indexes for conversion of 1D (sparse) contact map to a 2D (complete) contact map of <n_res> """
    indexes = np.zeros(((n_res**2-n_res)/2,2), dtype = np.int)
    i       = 0
    for j in range(n_res-1):
        for k in range(j+1,n_res):
            indexes[i]  = [j, k]
            i          += 1
    return indexes


