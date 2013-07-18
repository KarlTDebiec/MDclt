#!/usr/bin/python
desc = """analysis_toolkit.py
    Toolkit for analysis of molecular dynamics simulations
    Written by Karl Debiec on 12-02-12
    Last updated 13-07-18"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import inspect, os, sys, types
import h5py
import numpy as np
from   multiprocessing import Pool
from   importlib import import_module
from   hdf5_functions import HDF5_File
from   standard_functions import _string_to_function
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
#################################################### CORE FUNCTIONS ####################################################
def _primary_pool_director(task):
    function, segment, kwargs = task
    return function(segment, **kwargs)

def analyze_primary(hdf5_filename, path, segment_lister, analyses, n_cores = 1):
    """ Performs primary analysis (analysis of trajectories directly)
        1) Builds list of trajectory segments at <path> using function <segment_lister>
        2) Builds task list based on requested <analyses>, data present in <hdf5_filename>, and listed segments
        3) Distributes tasks across <n_cores> and writes results to <hdf5_filename> """
    print "Analyzing trajectory at {0}".format(path.replace("//","/"))
    for module in set([m.split(".")[0] for m in [a[0] for a in analyses]]): import_module("primary." + module)
    if   isinstance(segment_lister, types.FunctionType):
        segments    = segment_lister(path)
    elif isinstance(segment_lister, types.StringTypes):
        import_module(segment_lister.split(".")[0])
        segments    = _string_to_function(segment_lister)(path)
    else:
        raise Exception("segment_lister must be a function or string in format of 'module.function'")

    with HDF5_File(hdf5_filename) as hdf5_file:
        check_functions = {}
        task_list       = []
        for module_function, kwargs in analyses:
            module, function                    = module_function.split(".")
            check_functions[module_function]    = getattr(sys.modules["primary." + module], "_check_" + function)
        for segment in segments:
            if not os.path.isfile(segment.topology):    continue
            if not os.path.isfile(segment.trajectory):  continue
            for module_function, kwargs in analyses:
                check_function  = check_functions[module_function]
                new_tasks       = check_function(hdf5_file, segment, **kwargs)
                if new_tasks:
                    task_list  += new_tasks

        print "{0} tasks to be completed for {1} segments using {2} cores".format(len(task_list),len(segments),n_cores)
        pool    = Pool(n_cores)
        for results in pool.imap_unordered(_primary_pool_director, task_list):
            if not results: continue
            for result in results:
                if   len(result)  == 2: hdf5_file.add(result[0], result[1])
                else:                   hdf5_file.add(result[0], result[1], **result[2])
        pool.close()
        pool.join()

def analyze_primary_x(hdf5_filename, path, segment_lister, analyses, n_cores = 1):
    """ Performs primary cross-segment analysis (analysis of trajectories directly)
        1) Builds list of trajectory segments at <path> using function <segment_lister>
        2) Builds task list based on requested <analyses>, data present in <hdf5_filename>, and listed segments
        3) Completes tasks in serial, using <n_cores> for each task (if implemented), and writes results to
           <hdf5_filename> """
    print "Analyzing trajectory at {0}".format(path.replace("//","/"))
    for module in set([m.split(".")[0] for m in [a[0] for a in analyses]]): import_module("primary_x." + module)
    if   isinstance(segment_lister, types.FunctionType):
        segments    = segment_lister(path)
    elif isinstance(segment_lister, types.StringTypes):
        import_module(segment_lister.split(".")[0])
        segments        = _string_to_function(segment_lister)(path)
    else:
        raise Exception("segment_lister must be a function or string in format of 'module.function'")

    with HDF5_File(hdf5_filename) as hdf5_file:
        task_list   = []
        for module_function, kwargs in analyses:
            module, function    = module_function.split(".")
            check_function      = getattr(sys.modules["primary_x." + module], "_check_" + function)
            new_tasks           = check_function(hdf5_file, segments, **kwargs)
            if new_tasks:
                task_list      += new_tasks

        print "{0} tasks to be completed for {1} segments using {2} cores".format(len(task_list), len(segments), n_cores)
        for function, kwargs in task_list:
            kwargs["n_cores"]   = n_cores
            for result in function(segments, **kwargs):
                if   len(result)  == 2: hdf5_file.add(result[0], result[1], **kwargs)
                else:                   hdf5_file.add(result[0], result[1], **result[2])

def analyze_secondary(hdf5_filename, analyses, n_cores = 1):
    """ Performs secondary analysis (analysis of primary analysis)
        1) Builds task list based on <analyses>, concurrently loads required data from <hdf5_filename>
        2) Completes tasks in serial, using <n_cores> for each task (if implemented), and writes results to
           <hdf5_filename>"""
    print "Analyzing file {0}".format(hdf5_filename.replace("//","/"))
    for module in set([m.split(".")[0] for m in [a[0] for a in analyses]]): import_module("secondary." + module)

    with HDF5_File(hdf5_filename) as hdf5_file:
        task_list   = []
        for module_function, kwargs in analyses:
            module, function    = module_function.split(".")
            check_function      = getattr(sys.modules["secondary." + module], "_check_" + function)
            new_tasks           = check_function(hdf5_file, **kwargs)
            if new_tasks:
                task_list      += new_tasks

        for function, kwargs in task_list:
            kwargs["n_cores"]   = n_cores
            results = function(hdf5_file, **kwargs)
            if  not results:            continue
            for result in results:
                if   len(result)  == 2: hdf5_file.add(result[0], result[1], **kwargs)
                else:                   hdf5_file.add(result[0], result[1], **result[2])


