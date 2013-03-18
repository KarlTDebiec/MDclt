#!/usr/bin/python

desc = """gromacs.py
    Functions for primary analysis of GROMACS trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-03-17"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   standard_functions import is_num, month
################################################## ANALYSIS FUNCTIONS ##################################################
def energy(segment, **kwargs):
    """ parses <log>  of <logtype> for <segment> """
    log                 = segment.file_of_type(".log")
    logtype             = kwargs.get("logtype",    "standard")
    start_time          = kwargs.get("start_time", 0.0)
    nsteps              = float(commands.getoutput("grep nsteps " + log).split()[2])
    nstlog              = float(commands.getoutput("grep nstlog " + log).split()[2])
    length              = nsteps / nstlog
    with open(log, "r") as log:
        raw             = [line.rstrip() for line in log.readlines()]
    attrs               = {"decomposition": np.zeros((2, 3), dtype = np.int)}
    time                = np.zeros(length)
    potential_energy    = np.zeros(length)
    kinetic_energy      = np.zeros(length)
    total_energy        = np.zeros(length)
    temperature         = np.zeros(length)
    pressure            = np.zeros(length)
    i                   = -1
    for j, line in enumerate(raw):
        if   line.startswith("Domain decomposition grid"):
            line                        = line.split()
            attrs["decomposition"][0]   = int(line[3]), int(line[5]), int(line[7][:-1])
        elif line.startswith("PME domain decomposition"):
            line                        = line.split()
            attrs["decomposition"][1]   = int(line[3]), int(line[5]), int(line[7])
        elif line.startswith("       Time:"):
            seconds                     = int(float(line.split()[2]))
            attrs['duration']           = "{0}:{1:02d}:{2:02d}".format(seconds//60//60, seconds//60%60, seconds%60)
        elif line.startswith("Finished mdrun on"):
            line                        = line.split()
            attrs['date']               = "{0}-{1:02d}-{2:02d}".format(line[9][-2:], month(line[6]), int(line[7]))
            attrs['time']               = line[8]
        elif line.startswith("           Step           Time         Lambda"):
            time[i]                     = float(raw[j + 1].split()[1]) / 1000.
        elif i < length:
            if   (logtype == "standard"):
                if   line.startswith("      Potential    Kinetic En.   Total Energy    Temperature Pres. DC (bar)"):
                    line                    = raw[j + 1].split()
                    potential_energy[i]     = float(line[0])
                    kinetic_energy[i]       = float(line[1])
                    total_energy[i]         = float(line[2])
                    temperature[i]          = float(line[3])
                elif line.startswith(" Pressure (bar)"):
                    line                    = raw[j + 1].split()
                    pressure[i]             = float(line[0])
                    i                      += 1
            elif (logtype == "charmm"):
                if   line.startswith("   Coul. recip.      Potential    Kinetic En.   Total Energy    Temperature"):
                    line                    = raw[j + 1].split()
                    potential_energy[i]     = float(line[1])
                    kinetic_energy[i]       = float(line[2])
                    total_energy[i]         = float(line[3])
                    temperature[i]          = float(line[4])
                elif line.startswith(" Pres. DC (bar) Pressure (bar)   Constr. rmsd"):
                    line                    = raw[j + 1].split()
                    pressure[i]             = float(line[1])
                    i                      += 1
    return [("/" + segment + "/time",               time             + start_time),
            ("/" + segment + "/energy_total",       total_energy     * 0.239005736),
            ("/" + segment + "/energy_potential",   potential_energy * 0.239005736),
            ("/" + segment + "/energy_kinetic",     kinetic_energy   * 0.239005736),
            ("/" + segment + "/temperature",        temperature),
            ("/" + segment + "/pressure",           pressure),
            ("/" + segment,                         attrs),
            ("/" + segment + "/time",               {'units': "ns"}),
            ("/" + segment + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + segment + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + segment + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + segment + "/temperature",        {'units': "K"}),
            ("/" + segment + "/pressure",           {'units': "Bar"})]
def _check_energy(hdf5_file, segment, **kwargs):
    if not ([segment + "/time",
             segment + "/energy_total",
             segment + "/energy_potential",
             segment + "/energy_kinetic",
             segment + "/temperature",
             segment + "/pressure"] in hdf5_file):
            kwargs["start_time"] = float(segment)
            return [(energy, segment, kwargs)]
    else:   return False

#def mindist(arguments):
#    segment, prefix, path = arguments
#    tpr     = "{0}/{1}.tpr".format(path, prefix)
#    xtc     = "{0}/{1}.xtc".format(path, prefix)
#    xvg     = "{0}/mindist.xvg".format(path, segment)
#    command = "echo 1 | g_mindist -f {0} -s {1} -od {2} -pi".format(xtc, tpr, xvg)
#    commands.getoutput(command)
#    data    = np.loadtxt(xvg, skiprows = 24)
#    os.remove(xvg)
#    return [("/" + segment + "/mindist/min_pbc",    data[:,1] * 10.),
#            ("/" + segment + "/mindist/max_int",    data[:,2] * 10.),
#            ("/" + segment + "/mindist/min_pbc",    {'units': "A"}),
#            ("/" + segment + "/mindist/max_int",    {'units': "A"})]


