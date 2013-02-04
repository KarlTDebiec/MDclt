#!/usr/bin/python

desc = """anton.py
    Functions for analysis of Anton trajectories
    Written by Karl Debiec on 12-11-29
    Last updated 13-02-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num, month
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ Parses energy log, either calculated on Anton or generated afterwards using vrun """
    segment, ene    = arguments
    try:    data    = np.loadtxt(ene)
    except: return None
    head            = commands.getoutput("head -n 1 {0}".format(ene)).split()
    if "vrun" in head:
        attributes          = {'type': 'vrun'}
    else:
        attributes          = {'type': 'ene'}
        attributes['date']  = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
        attributes['time']  = head[7]
    return [("/" + segment + "/time",               data[:,0] / 1000.),
            ("/" + segment + "/energy_total",       data[:,1]),
            ("/" + segment + "/energy_potential",   data[:,2]),
            ("/" + segment + "/energy_kinetic",     data[:,3]),
            ("/" + segment + "/temperature",        data[:,8]),
            ("/" + segment,                         attributes),
            ("/" + segment + "/time",               {'units': "ns"}),
            ("/" + segment + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + segment + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + segment + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + segment + "/temperature",        {'units': "K"})]
def check_energy(hierarchy, segment, arguments):
    segment, path, topology, trajectory = segment
    if not (set([segment + "/time",             segment + "/energy_total",  segment + "/energy_potential",
                 segment + "/energy_kinetic",   segment + "/temperature"]).issubset(hierarchy)):
        ene = "{0}/{1}.ene".format(path, segment)
        return [(energy, (segment, ene))]
    else:   return False


