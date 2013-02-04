#!/usr/bin/python

desc = """analyze_amber.py
    Functions for analysis of Amber simulations
    Written by Karl Debiec on 12-12-01
    Last updated 12-12-01"""

########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os
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
    """ parses energy log using provided format, including start time, assumes 1 ns jobstep length """
    jobstep, log        = arguments
    nstlim              = float(commands.getoutput("grep nstlim " + log).split()[2][:-1])
    ntpr                = float(commands.getoutput("grep ntpr   " + log).split()[2][:-1])
    dt                  = float(commands.getoutput("grep dt     " + log).split()[2][:-1])
    length              = nstlim / ntpr
    log                 = open(log, 'r')
    raw                 = [l.split() for l in log.readlines()]
    log.close()
    attributes          = {}
    time                = np.arange(dt, length * dt + dt, dt) + float(jobstep) * length * dt
    potential_energy    = np.zeros(length)
    kinetic_energy      = np.zeros(length)
    total_energy        = np.zeros(length)
    temperature         = np.zeros(length)
    i                   = 0
    for j, l in enumerate(raw):
        if   len(l) == 9 and (l[2], l[3]) == ('Total', 'wall'):
            attributes['duration']          = "{0}:{1}:{2}".format(int(l[5])//60//60, int(l[5])//60%60,int(l[5])%60)
        elif len(l) == 6 and (l[1], l[2]) == ('Run', 'on'):
            attributes['date']              = "{0}-{1}-{2}".format(l[3][8:], l[3][:2], l[3][3:5])
            attributes['time']              = l[5]
        elif len(l) == 12 and (l[0], l[3]) == ('NSTEP', 'TIME(PS)'):
            if i < length:
                temperature[i]              = float(l[8])
        elif len(l) == 9 and (l[0], l[3]) == ('Etot', 'EKtot'):
            if i < length:
                total_energy[i]             = float(l[2])
                potential_energy[i]         = float(l[8])
                kinetic_energy[i]           = float(l[5])
                i                          += 1
    return [("/" + jobstep + "/time",               time),
            ("/" + jobstep + "/energy_total",       total_energy),
            ("/" + jobstep + "/energy_potential",   potential_energy),
            ("/" + jobstep + "/energy_kinetic",     kinetic_energy),
            ("/" + jobstep + "/temperature",        temperature),
            ("/" + jobstep,                         attributes),
            ("/" + jobstep + "/time",               {'units': "ns"}),
            ("/" + jobstep + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + jobstep + "/temperature",        {'units': "K"})]
def check_energy(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    if not (set([jobstep + "/time",           jobstep + "/energy_total", jobstep + "/energy_potential",
                 jobstep + "/energy_kinetic", jobstep + "/temperature"]).issubset(hierarchy)):
        log     = "{0}/{1}.out".format(path, jobstep)
        return    [(energy, (jobstep, log))]
    else:   return False


