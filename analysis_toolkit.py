#!/usr/bin/python
desc = """analysis_toolkit.py
    Toolkit for analysis of molecular dynamics simulations
    Written by Karl Debiec on 12-02-12
    Last updated 13-05-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import inspect, os, sys
import h5py
import numpy as np
from   multiprocessing import Pool
from   importlib import import_module
from   hdf5_functions import HDF5_File
from   standard_functions import _string_to_function
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
#################################################### CORE FUNCTIONS ####################################################
def _primary_make_task_list(hdf5_file, segments, analyses):
    check_functions = {}
    task_list       = []
    for module_function, kwargs in analyses:
        module, function                    = module_function.split(".")
        check_functions[module_function]    = getattr(sys.modules["primary.{0}".format(module)], '_check_' + function)
    for segment in segments:
        if not os.path.isfile(segment.topology):    continue
        if not os.path.isfile(segment.trajectory):  continue
        for module_function, kwargs in analyses:
            check_function  = check_functions[module_function]
            check           = check_function(hdf5_file, segment, **kwargs)
            if check:
                task_list  += check
    return task_list

def _primary_complete_tasks(hdf5_file, task_list, n_cores = 1):
    pool    = Pool(n_cores)
    for result in pool.imap_unordered(_primary_pool_director, task_list):
        if  not result:             continue
        for path, data in result:   hdf5_file.add(path, data)
    pool.close()
    pool.join()

def _primary_pool_director(task):
    function, segment, kwargs = task
    return function(segment, **kwargs)

def analyze_primary(hdf5_filename, path, segment_lister, analyses, n_cores = 1):
    """ Performs primary analysis (analysis of trajectory directly)
        1) Builds list of trajectory segments at <path> using function <segment_lister>
        2) Builds task list based on requested <analyses>, data present in <hdf5_filename>, and segments at <path>
        3) Distributes tasks across <n_cores> and writes results to <hdf5_filename> """
    print "Analyzing trajectory at {0}".format(path.replace('//','/'))
    import_module(segment_lister.split(".")[0])
    for module_name in set([m.split(".")[0] for m in [a[0] for a in analyses]]):
        import_module("primary.{0}".format(module_name))
    segments        = _string_to_function(segment_lister)(path)
    with HDF5_File(hdf5_filename) as hdf5_file:
        task_list   = _primary_make_task_list(hdf5_file, segments, analyses)
        print "{0} tasks to be completed for {1} segments using {2} cores".format(len(task_list), len(segments),
          n_cores)
        _primary_complete_tasks(hdf5_file, task_list, n_cores)

def _secondary_make_task_list(hdf5_file, analyses):
    task_list       = []
    for module_function, kwargs in analyses:
        module, function    = module_function.split(".")
        check_function      = getattr(sys.modules["secondary.{0}".format(module)], '_check_' + function)
        check               = check_function(hdf5_file, **kwargs)
        if check:
            task_list      += check
    return task_list

def _secondary_complete_tasks(hdf5_file, task_list, n_cores = 1):
    for function, kwargs in task_list:
        results = function(hdf5_file, n_cores, **kwargs)
        if  not results:            continue
        for path, data in results:  hdf5_file.add(path, data)

def analyze_secondary(hdf5_filename, analyses, n_cores = 1):
    """ Performs secondary analysis (analysis of primary analysis)
        1) Builds task list based on <analyses>, loads required data from <hdf5_filename>
        2) Completes tasks in serial, using <n_cores> for each task if implemented, and saves results to <hdf5_file>"""
    print "Analyzing file {0}".format(hdf5_filename)
    for module_name in set([m.split('.')[0] for m in [a[0] for a in analyses]]):
        import_module("secondary.{0}".format(module_name))
    with HDF5_File(hdf5_filename) as hdf5_file:
        task_list    = _secondary_make_task_list(hdf5_file, analyses)
        _secondary_complete_tasks(hdf5_file, task_list, n_cores)



