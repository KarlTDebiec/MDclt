#!/usr/bin/python

desc = """analysis_toolkit.py
    Toolkit for analysis of molecular dynamics simulations
    Written by Karl Debiec on 12-02-12
    Last updated 13-02-08"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import inspect, os, sys
import h5py
import numpy as np
from   multiprocessing import Pool
from   importlib import import_module
module_path     = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path       += [module_path + "/primary_functions"]
sys.path       += [module_path + "/secondary_functions"]
sys.path       += [module_path + "/cython_functions"]
from   hdf5_functions import get_hierarchy, load_data, add_data
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
#################################################### CORE FUNCTIONS ####################################################
def _primary_make_task_list(hierarchy, segments, analyses):
    check_functions = {}
    task_list       = []
    for analysis, module, function in [[a] + a.split('.') for a in analyses]:
        check_function              = getattr(sys.modules[module], 'check_' + function)
        check_functions[analysis]   = check_function
    for segment in segments:
        for analysis, arguments in analyses.iteritems():
            check   = check_functions[analysis](hierarchy, segment, arguments)
            if check: task_list  += check
    return task_list
def _primary_complete_tasks(hdf5_file, task_list, n_cores = 1):
    pool    = Pool(n_cores)
    for result in pool.imap_unordered(_primary_pool_director, task_list):
        if  not result:             continue
        for path, data in result:   add_data(hdf5_file, path, data)
    pool.close()
    pool.join()
def _primary_pool_director(task):
    function, arguments = task
    return function(arguments)
def analyze_primary(hdf5_filename, path, segment_lister, analyses, n_cores = 1):
    """ Performs primary analysis (analysis of trajectory directly)
        1) Checks data already present in <hdf5_filename>
        2) Builds list of trajectory segments at <path> using function <segment_lister>
        3) Builds task list based on requested <analyses>, data present in <hdf5_filename>, and segments at <path>
        4) Distributes tasks across <n_cores> and writes results to <hdf5_filename> """
    sys.stderr.write("Analyzing trajectory at {0}.\n".format(path.replace('//','/')))
    for module_name in set([m.split('.')[0] for m in analyses.keys() + [segment_lister]]):  import_module(module_name)
    hierarchy       = get_hierarchy(hdf5_filename)
    segments        = getattr(sys.modules[segment_lister.split('.')[0]], segment_lister.split('.')[1])(path)
    task_list       = _primary_make_task_list(hierarchy, segments, analyses)
    sys.stderr.write("{0} tasks to be completed for {1} segments using {2} cores.\n".format(len(task_list),
      len(segments), n_cores))
    hdf5_file       = h5py.File(hdf5_filename)
    _primary_complete_tasks(hdf5_file, task_list, n_cores)
    hdf5_file.flush()
    hdf5_file.close()

def _secondary_make_path_list(analyses):
    path_list = []
    for analysis, arguments in analyses.iteritems():
        module, function    = analysis.split('.')
        path_function       = getattr(sys.modules[module], 'path_' + function)
        path_list          += path_function(arguments)
    return list(set(path_list))
def _secondary_make_task_list(analyses):
    check_functions = {}
    task_list       = []
    for analysis, module, function in [[a] + a.split('.') for a in analyses]:
        check_function              = getattr(sys.modules[module], 'check_' + function)
        check_functions[analysis]   = check_function
    for analysis, arguments in analyses.iteritems():
        check   = check_functions[analysis](arguments)
        if check: task_list  += check
    return task_list
def _secondary_complete_tasks(hdf5_file, primary_data, task_list, n_cores = 1):
    for function, arguments in task_list:
        results = function(primary_data, arguments, n_cores)
        if  not results:            continue
        for path, data in results:  add_data(hdf5_file, path, data)
def analyze_secondary(hdf5_filename, analyses, n_cores = 1):
    """ Performs secondary analysis (analysis of primary analysis)
        1) Builds list of data required from <hdf5_filename> by <analyses>
        2) Loads required data from <hdf5_filename>
        3) Builds task list based on <analyses>
        4) Completes tasks in serial, using <n_cores> for each task if implemented, and saves results to <hdf5_file>"""
    sys.stderr.write("Analyzing file {0}.\n".format(hdf5_filename))
    for module_name in set([m.split('.')[0] for m in analyses.keys()]):  import_module(module_name)
    path_list       = _secondary_make_path_list(analyses)
    hdf5_file       = h5py.File(hdf5_filename)
    primary_data    = load_data(hdf5_file, path_list)
    task_list       = _secondary_make_task_list(analyses)
    _secondary_complete_tasks(hdf5_file, primary_data, task_list, n_cores)
    hdf5_file.flush()
    hdf5_file.close()


