#!/usr/bin/python

desc = """analyze_acemd.py
    Functions for analysis of ACEMD simulations
    Written by Karl Debiec on 12-11-30
    Last updated 12-11-30"""

########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
################################################## GENERAL FUNCTIONS ###################################################
def is_num(test):
    try:    float(test)
    except: return False
    return  True
#################################################### CORE FUNCTIONS ####################################################
def jobsteps(path):
    jobsteps = []
    for f in sorted([f for f in os.listdir(path) if is_num(f)]):
        jobsteps += [(f, "{0}/{1}/".format(path, f), "{0}/{1}/{1}_solute.pdb".format(path, f),
                      "{0}/{1}/{1}_solute.xtc".format(path, f))]
    return jobsteps
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ parses energy log, calculates time based on configuration file """
    jobstep, log, conf  = arguments
    data        = np.genfromtxt(log, skip_header = 1, invalid_raise = False)[1:,:]
    command     = "grep \"timestep                            \"  {0}".format(conf)
    timestep    = float(commands.getoutput(command).split()[1]) / 1000000
    return [("/" + jobstep + "/time",                   data[:, 0] * timestep + float(jobstep) + timestep),
            ("/" + jobstep + "/energy_total",           data[:, 9]),
            ("/" + jobstep + "/energy_potential",       data[:, 6]),
            ("/" + jobstep + "/energy_kinetic",         data[:, 7]),
            ("/" + jobstep + "/temperature",            data[:,10]),
            ("/" + jobstep + "/time",                   {'units': "ns"}),
            ("/" + jobstep + "/energy_total",           {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_potential",       {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_kinetic",         {'units': "kcal / mol"}),
            ("/" + jobstep + "/temperature",            {'units': "K"})]
def check_energy(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    if not (set([jobstep + "/time",             jobstep + "/energy_total",  jobstep + "/energy_potential",
                 jobstep + "/energy_kinetic",   jobstep + "/temperature"]).issubset(hierarchy)):
        log         = "{0}/{1}.log".format(path, jobstep)
        conf        = "{0}/{1}.conf".format(path, jobstep)
        return [(energy, (jobstep, log, conf))]
    else:   return False


