#!/usr/bin/python

desc = """anton.py
    Functions for primary analysis of Anton trajectories
    Written by Karl Debiec on 12-11-29
    Last updated 13-03-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num, month
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(segment, **kwargs):
    """ Parses <ene>; may be either calculated on Anton or generated afterwards using vrun """
    ene                 = segment.file_of_type(".ene")
    head                = commands.getoutput("head -n 1 {0}".format(ene)).split()
    if "vrun" in head:
        attrs           = {'type': 'vrun'}
    else:
        attrs           = {'type': 'ene'}
        attrs['date']   = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
        attrs['time']   = head[7]
    data                = np.loadtxt(ene)
    return [("/" + segment + "/time",               data[:,0] / 1000.),
            ("/" + segment + "/energy_total",       data[:,1]),
            ("/" + segment + "/energy_potential",   data[:,2]),
            ("/" + segment + "/energy_kinetic",     data[:,3]),
            ("/" + segment + "/temperature",        data[:,8]),
            ("/" + segment,                         attrs),
            ("/" + segment + "/time",               {'units': "ns"}),
            ("/" + segment + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + segment + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + segment + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + segment + "/temperature",        {'units': "K"})]
def _check_energy(hdf5_file, segment, **kwargs):
    if not ([segment + "/time",
             segment + "/energy_total",
             segment + "/energy_potential",
             segment + "/energy_kinetic",
             segment + "/temperature"] in hdf5_file):
            return [(energy, segment, kwargs)]
    else:   return False


