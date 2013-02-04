#!/usr/bin/python

desc = """amber.py
    Functions for analysis of AMBER trajectories
    Written by Karl Debiec on 12-12-01
    Last updated 13-02-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ parses energy log using provided format, including start time, assumes 1 ns segment length """
    segment, log        = arguments
    nstlim              = float(commands.getoutput("grep nstlim " + log).split()[2][:-1])
    ntpr                = float(commands.getoutput("grep ntpr   " + log).split()[2][:-1])
    dt                  = float(commands.getoutput("grep dt     " + log).split()[2][:-1])
    length              = nstlim / ntpr
    log                 = open(log, 'r')
    raw                 = [l.split() for l in log.readlines()]
    log.close()
    attributes          = {}
    time                = np.arange(dt, length * dt + dt, dt) + float(segment) * length * dt
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
    return [("/" + segment + "/time",               time),
            ("/" + segment + "/energy_total",       total_energy),
            ("/" + segment + "/energy_potential",   potential_energy),
            ("/" + segment + "/energy_kinetic",     kinetic_energy),
            ("/" + segment + "/temperature",        temperature),
            ("/" + segment,                         attributes),
            ("/" + segment + "/time",               {'units': "ns"}),
            ("/" + segment + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + segment + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + segment + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + segment + "/temperature",        {'units': "K"})]
def check_energy(hierarchy, segment, arguments):
    segment, path, topology, trajectory = segment
    if not (set([segment + "/time",           segment + "/energy_total", segment + "/energy_potential",
                 segment + "/energy_kinetic", segment + "/temperature"]).issubset(hierarchy)):
        log     = "{0}/{1}.out".format(path, segment)
        return    [(energy, (segment, log))]
    else:   return False


