#!/usr/bin/python

desc = """analyze_anton.py
    Functions for analysis of Anton simulations
    Written by Karl Debiec on 12-11-29
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
def month(string):
    month = {'jan':  1, 'feb':  2, 'mar':  3, 'apr':  4, 'may':  5, 'jun':  6,
             'jul':  7, 'aug':  8, 'sep':  9, 'oct': 10, 'nov': 11, 'dec': 12}
    try:    return month[string.lower()]
    except: return None
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ parses energy log, either calculated on Anton or generated afterwards using vrun """
    jobstep, ene    = arguments
    try:    data    = np.loadtxt(ene)
    except: return None
    head            = commands.getoutput("head -n 1 {0}".format(ene)).split()
    if "vrun" in head:
        attributes          = {'type': 'vrun'}
    else:
        attributes          = {'type': 'ene'}
        attributes['date']  = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
        attributes['time']  = head[7]
    return [("/" + jobstep + "/time",               data[:,0] / 1000.),
            ("/" + jobstep + "/energy_total",       data[:,1]),
            ("/" + jobstep + "/energy_potential",   data[:,2]),
            ("/" + jobstep + "/energy_kinetic",     data[:,3]),
            ("/" + jobstep + "/temperature",        data[:,8]),
            ("/" + jobstep,                         attributes),
            ("/" + jobstep + "/time",               {'units': "ns"}),
            ("/" + jobstep + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + jobstep + "/temperature",        {'units': "K"})]
def check_energy(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    if not (set([jobstep + "/time",             jobstep + "/energy_total",  jobstep + "/energy_potential",
                 jobstep + "/energy_kinetic",   jobstep + "/temperature"]).issubset(hierarchy)):
        ene = "{0}/{1}.ene".format(path, jobstep)
        return [(energy, (jobstep, ene))]
    else:   return False


