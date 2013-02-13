#!/usr/bin/python

desc = """desmond.py
    Functions for primary analysis of Desmond trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-02-08"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num, month
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ parses <ene> for <segment> """
    segment, ene    = arguments
    data            = np.loadtxt(ene)
    head            = commands.getoutput("head -n 3 {0}".format(ene)).split('\n')[2].split()
    date            = "{0:02d}-{1:02d}-{2:02d}".format(int(head[8][2:]), int(month(head[5])), int(head[6]))
    start_time      = head[7]
    return [("/" + segment + "/time",               data[1:,0] / 1000.),
            ("/" + segment + "/energy_total",       data[1:,1]),
            ("/" + segment + "/energy_potential",   data[1:,2]),
            ("/" + segment + "/energy_kinetic",     data[1:,3]),
            ("/" + segment + "/temperature",        data[1:,9]),
            ("/" + segment,                         {'date': date,  'time': start_time}),
            ("/" + segment + "/time",               {'units': "ns"}),
            ("/" + segment + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + segment + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + segment + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + segment + "/temperature",        {'units': "K"})]
def check_energy(hierarchy, segment, arguments):
    segment, path, topology, trajectory = segment
    if not os.path.isfile(topology):    return False
    if not os.path.isfile(trajectory):  return False
    if not (set([segment + "/time",             segment + "/energy_total",  segment + "/energy_potential",
                 segment + "/energy_kinetic",   segment + "/temperature"]).issubset(hierarchy)):
        ene = "{0}/{1}.ene".format(path, segment)
        return [(energy, (segment, ene))]
    else:   return False


