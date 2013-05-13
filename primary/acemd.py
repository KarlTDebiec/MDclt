#!/usr/bin/python

desc = """acemd.py
    Functions for primary analysis of ACEMD trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-03-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num
################################################# ANALYSIS FUNCTIONS ##################################################
def energy(segment, **kwargs):
    """ Parses <log> and <conf> for <segment> """
    log         = segment.file_of_type(".log")
    conf        = segment.file_of_type(".conf")
    data        = np.genfromtxt(log, skip_header = 1, invalid_raise = False)[1:,:]
    command     = "grep \"timestep                            \"  {0}".format(conf)
    timestep    = float(commands.getoutput(command).split()[1]) / 1000000
    return [(segment + "/time",                   data[:, 0] * timestep + float(segment) + timestep),
            (segment + "/energy_total",           data[:, 9]),
            (segment + "/energy_potential",       data[:, 6]),
            (segment + "/energy_kinetic",         data[:, 7]),
            (segment + "/temperature",            data[:,10]),
            (segment + "/time",                   {'units': "ns"}),
            (segment + "/energy_total",           {'units': "kcal / mol"}),
            (segment + "/energy_potential",       {'units': "kcal / mol"}),
            (segment + "/energy_kinetic",         {'units': "kcal / mol"}),
            (segment + "/temperature",            {'units': "K"})]
def _check_energy(hdf5_file, segment, **kwargs):
    segment, path, topology, trajectory = segment
    if not ([segment + "/time",
             segment + "/energy_total",
             segment + "/energy_potential",
             segment + "/energy_kinetic",
             segment + "/temperature"] in hdf5_file):
            return [(energy, segment, kwargs)]
    else:   return False
