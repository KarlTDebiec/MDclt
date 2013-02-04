#!/usr/bin/python

desc = """anton.py
    Functions for analysis of Anton trajectories
    Written by Karl Debiec on 12-11-29
    Last updated 13-02-03"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, importlib, os, sys
import numpy as np
from   standard_functions import is_num, month
#################################################### CORE FUNCTIONS ####################################################
def jobsteps(path):
    jobsteps = []
    for f in sorted([f for f in os.listdir(path) if is_num(f)]):
        jobsteps += [(f, "{0}/{1}/".format(path, f), "{0}/{1}/{1}_solute.pdb".format(path, f),
                      "{0}/{1}/{1}_solute.xtc".format(path, f))]
    return jobsteps
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ Parses energy log, either calculated on Anton or generated afterwards using vrun """
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


