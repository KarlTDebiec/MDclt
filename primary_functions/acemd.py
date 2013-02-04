#!/usr/bin/python

desc = """acemd.py
    Functions for analysis of ACEMD trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-02-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num
################################################# ANALYSIS FUNCTIONS ##################################################
def energy(arguments):
    """ parses energy log, calculates time based on configuration file """
    segment, log, conf  = arguments
    data        = np.genfromtxt(log, skip_header = 1, invalid_raise = False)[1:,:]
    command     = "grep \"timestep                            \"  {0}".format(conf)
    timestep    = float(commands.getoutput(command).split()[1]) / 1000000
    return [("/" + segment + "/time",                   data[:, 0] * timestep + float(segment) + timestep),
            ("/" + segment + "/energy_total",           data[:, 9]),
            ("/" + segment + "/energy_potential",       data[:, 6]),
            ("/" + segment + "/energy_kinetic",         data[:, 7]),
            ("/" + segment + "/temperature",            data[:,10]),
            ("/" + segment + "/time",                   {'units': "ns"}),
            ("/" + segment + "/energy_total",           {'units': "kcal / mol"}),
            ("/" + segment + "/energy_potential",       {'units': "kcal / mol"}),
            ("/" + segment + "/energy_kinetic",         {'units': "kcal / mol"}),
            ("/" + segment + "/temperature",            {'units': "K"})]
def check_energy(hierarchy, segment, arguments):
    segment, path, topology, trajectory = segment
    if not (set([segment + "/time",             segment + "/energy_total",  segment + "/energy_potential",
                 segment + "/energy_kinetic",   segment + "/temperature"]).issubset(hierarchy)):
        log         = "{0}/{1}.log".format(path, segment)
        conf        = "{0}/{1}.conf".format(path, segment)
        return [(energy, (segment, log, conf))]
    else:   return False


