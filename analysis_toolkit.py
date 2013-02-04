#!/usr/bin/python

desc = """analysis_toolkit.py
    Toolkit for analysis of simulations
    From trajectory location, hdf5 database, and list of analyses constructs task list
    Splits tasks among designated number of cores and saves results to hdf5
    Written by Karl Debiec on 12-02-12
    Last updated 13-02-03"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import importlib, inspect, os, sys
import h5py as h5
import numpy as np
from   multiprocessing import Pool
module_path     = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path       += [module_path + "/primary_functions"]
sys.path       += [module_path + "/secondary_functions"]
sys.path       += [module_path + "/cython_functions"]
from   hdf5_functions import get_hierarchy
np.set_printoptions(precision = 3, suppress = True, linewidth = 120)
#################################################### CORE FUNCTIONS ####################################################
def make_task_list(hierarchy, jobsteps, analyses, modules):
    checkers    = {}
    task_list   = []
    for analysis in analyses:
        checkers[analysis]  = getattr(modules[analysis.split('.')[0]], 'check_' + analysis.split('.')[1])
    for jobstep in jobsteps:
        for analysis, arguments in analyses.iteritems():
            check   = checkers[analysis](hierarchy, jobstep, arguments)
            if check:   task_list  += check
    return task_list
def complete_tasks(n_cores, hdf5_file, task_list):
    pool    = Pool(n_cores)
    try:
        hdf5_file   = h5.File(hdf5_file)
        for result_i, result in enumerate(pool.imap_unordered(pool_director, task_list)):
            if result is None: continue
            for subresult in result:
                path, data = subresult
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
                    try:
                        group.create_dataset(name, data = data, compression = 'gzip')
                        print "    '{0}' added".format('/'.join(path + [name]))
                    except:
                        print "{0}\n{1}\n{2} FAILED TO ADD".format(group, name, data)
                        raise
    finally:
        hdf5_file.flush()
        hdf5_file.close()
        pool.close()
        pool.join()
def pool_director(task):
    function, arguments = task
    return function(arguments)

def analyze_trajectory(hdf5_file, path, package, analyses, n_cores = 1):
    sys.stderr.write("Analyzing {0} trajectory at {1}.\n".format(package, path.replace('//','/')))
    modules         = {}
    for m in set([m.split('.')[0] for m in analyses.keys()] + [package]):
        print m
        modules[m]  = importlib.import_module(m)
    hierarchy   = get_hierarchy(hdf5_file)
    jobsteps    = modules[package].jobsteps(path)
    task_list   = make_task_list(hierarchy, jobsteps, analyses, modules)
    sys.stderr.write("{0} tasks to be completed for {1} jobsteps using {2} cores.\n".format(len(task_list),
      len(jobsteps), n_cores))
    complete_tasks(n_cores, hdf5_file, task_list)

def analyze_post(hdf5_file, analyses, n_cores = 1):
    sys.stderr.write("Analyzing dataset in {0}.\n".format(hdf5_file))
    modules         = {}
    for m in set([m.split('.')[0] for m in analyses.keys()]):
        modules[m]  = importlib.import_module(m)
    hierarchy   = get_hierarchy(hdf5_file)

######################################################### MAIN #########################################################
if __name__ == '__main__':
    pass
    



