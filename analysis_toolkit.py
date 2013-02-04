#!/usr/bin/python

desc = """analysis_toolkit.py
    Toolkit for analysis of molecular dynamics simulations
    From trajectory location, hdf5 database, and list of analyses constructs task list
    Splits tasks among designated number of cores and saves results to hdf5
    Written by Karl Debiec on 12-02-12
    Last updated 13-02-04"""
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
from   hdf5_functions import get_hierarchy, add_data
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
#################################################### CORE FUNCTIONS ####################################################
def make_task_list(hierarchy, segments, analyses):
    checkers    = {}
    task_list   = []
    for analysis, module, function in [[a] + a.split('.') for a in analyses]:
        checkers[analysis]  = getattr(sys.modules[module], 'check_' + function)
    for segment in segments[:20]:
        for analysis, arguments in analyses.iteritems():
            check   = checkers[analysis](hierarchy, segment, arguments)
            if check: task_list  += check
    return task_list
def complete_tasks(hdf5_file, task_list, n_cores = 1):
    pool    = Pool(n_cores)
    try:
        hdf5_file   = h5py.File(hdf5_file)
        for result in pool.imap_unordered(pool_director, task_list):
            if  not result: continue
            for path, data in result:
                add_data(hdf5_file, path, data)
    finally:
        hdf5_file.flush()
        hdf5_file.close()
        pool.close()
        pool.join()
def pool_director(task):
    function, arguments = task
    return function(arguments)

def analyze_primary(hdf5_file, path, segment_lister, analyses, n_cores = 1):
    sys.stderr.write("Analyzing trajectory at {0}.\n".format(path.replace('//','/')))
    for module_name in set([m.split('.')[0] for m in analyses.keys() + [segment_lister]]):  import_module(module_name)
    hierarchy       = get_hierarchy(hdf5_file)
    segments        = getattr(sys.modules[segment_lister.split('.')[0]], segment_lister.split('.')[1])(path)
    task_list       = make_task_list(hierarchy, segments, analyses)
    sys.stderr.write("{0} tasks to be completed for {1} segments using {2} cores.\n".format(len(task_list),
      len(segments), n_cores))
    complete_tasks(hdf5_file, task_list, n_cores)

######################################################### MAIN #########################################################
if __name__ == '__main__':
    pass


