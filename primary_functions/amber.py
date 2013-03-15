#!/usr/bin/python

desc = """amber.py
    Functions for primary analysis of AMBER trajectories
    Written by Karl Debiec on 12-12-01
    Last updated 13-03-06"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(segment, **kwargs):
    """ Parses <out>"""
    out                 = segment.file_of_type(".out")
    time_offset         = kwargs.get("time_offset", 0.0)
    nstlim              = float(commands.getoutput("grep nstlim " + out).split()[2][:-1])
    ntpr                = float(commands.getoutput("grep ntpr   " + out).split()[2][:-1])
    dt                  = float(commands.getoutput("grep dt     " + out).split()[2][:-1])
    length              = nstlim / ntpr
    attrs               = {}
    time                = np.zeros(length)
    potential_energy    = np.zeros(length)
    kinetic_energy      = np.zeros(length)
    total_energy        = np.zeros(length)
    temperature         = np.zeros(length)
    i                   = 0
    with open(out, 'r') as out:
        raw             = [line.strip() for line in out.readlines()]
    for line in raw:
        if line.startswith("NSTEP") and i < length:
            line                = line.split()
            time[i]             = float(line[5]) / 1000 + time_offset
            temperature[i]      = float(line[8])
        elif line.startswith("Etot") and i < length:
            line                = line.split()
            total_energy[i]     = float(line[2])
            potential_energy[i] = float(line[8])
            kinetic_energy[i]   = float(line[5])
            i                  += 1
        elif line.startswith("|  Master Total wall time"):
            seconds             = int(line.split()[5])
            attrs['duration']   = "{0}:{1:02d}:{2:02d}".format(seconds // 60 // 60, seconds // 60 % 60, seconds % 60)
        elif line.startswith("| Run on"):
            line                = line.split()
            date                = line[3]
            attrs['date']       = "{0}-{1}-{2}".format(date[8:], date[:2], date[3:5])
            attrs['time']       = line[5]
        elif line.startswith("| Running AMBER/MPI version on"):
            line                = line.split()
            attrs['n_cores']    = int(line[5])
    return [("/" + segment + "/time",               time),
            ("/" + segment + "/energy_total",       total_energy),
            ("/" + segment + "/energy_potential",   potential_energy),
            ("/" + segment + "/energy_kinetic",     kinetic_energy),
            ("/" + segment + "/temperature",        temperature),
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
             segment + "/temperature"]  in hdf5_file):
            return [(energy, segment, kwargs)]
    else:   return False


