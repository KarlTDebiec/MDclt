#!/usr/bin/python

desc = """gromacs.py
    Functions for analysis of Gromacs trajectories
    Written by Karl Debiec on 12-11-30
    Last updated 13-02-03"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import commands, os, sys
import numpy as np
from   general import is_num, month
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
    jobstep, log, logtype = arguments
    nsteps              = float(commands.getoutput("grep nsteps    " + log).split()[2])
    nstlog              = float(commands.getoutput("grep nstlog    " + log).split()[2])
    delta_t             = float(commands.getoutput("grep delta_t   " + log).split()[2])
    nstxtcout           = float(commands.getoutput("grep nstxtcout " + log).split()[2])
    length              = nsteps / nstlog
    dt                  = delta_t * nstxtcout / 1000.
    log                 = open(log, 'r')
    raw                 = [l.split() for l in log.readlines()]
    log.close()
    attributes          = {'decomposition': np.zeros((2, 3), dtype = np.int)}
    time                = np.arange(dt, length * dt + dt, dt) + float(jobstep)
    potential_energy    = np.zeros(length)
    kinetic_energy      = np.zeros(length)
    total_energy        = np.zeros(length)
    temperature         = np.zeros(length)
    pressure            = np.zeros(length)
    i                   = -1
    for j, l in enumerate(raw):
        if   (len(l) == 12 and (l[0], l[1]) == ('Domain', 'decomposition')):
            attributes['decomposition'][0]  = np.array([l[3], l[5], l[7][:-1]], dtype = np.int)
        elif (len(l) ==  8 and (l[0], l[1]) == ('PME', 'domain')):
            attributes['decomposition'][1]  = np.array([l[3], l[5], l[7]],      dtype = np.int)
        elif (len(l) ==  4 and (l[0])       == ('Time:')):
            attributes['duration']          = raw[j + 1][0].replace('h', ':')
        elif (len(l) == 10 and (l[0], l[1]) == ('Finished', 'mdrun')):
            attributes['date']              = "{0:02d}-{1:02d}-{2:02d}".format(int(l[9][-2:]), month(l[6]), int(l[7]))
            attributes['time']              = l[8]
        elif (logtype == "standard"):
            if   (len(l) == 9 and (l[0], l[1]) == ('Potential', 'Kinetic')):
                if -1 < i < length:
                    potential_energy[i]     = float(raw[j + 1][0])
                    kinetic_energy[i]       = float(raw[j + 1][1])
                    total_energy[i]         = float(raw[j + 1][2])
                    temperature[i]          = float(raw[j + 1][3])
            elif (len(l) == 4 and l[0]      == 'Pressure'):
                if -1 < i < length:
                    pressure[i]             = float(raw[j + 1][0])
                i                          += 1
        elif (logtype == "charmm"):
            if   (len(l) == 8 and l[0]      == 'Coul.'):
                if -1 < i < length:
                    potential_energy[i]     = float(raw[j + 1][1])
                    kinetic_energy[i]       = float(raw[j + 1][2])
                    total_energy[i]         = float(raw[j + 1][3])
                    temperature[i]          = float(raw[j + 1][4])
            elif (len(l) == 7 and (l[0], l[1]) == ('Pres.', 'DC')):
                if -1 < i < length:
                    pressure[i]             = float(raw[j + 1][1])
                i                          += 1
    return [("/" + jobstep + "/time",               time),
            ("/" + jobstep + "/energy_total",       total_energy     * 0.239005736),
            ("/" + jobstep + "/energy_potential",   potential_energy * 0.239005736),
            ("/" + jobstep + "/energy_kinetic",     kinetic_energy   * 0.239005736),
            ("/" + jobstep + "/temperature",        temperature),
            ("/" + jobstep + "/pressure",           pressure),
            ("/" + jobstep,                         attributes),
            ("/" + jobstep + "/time",               {'units': "ns"}),
            ("/" + jobstep + "/energy_total",       {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_potential",   {'units': "kcal / mol"}),
            ("/" + jobstep + "/energy_kinetic",     {'units': "kcal / mol"}),
            ("/" + jobstep + "/temperature",        {'units': "K"}),
            ("/" + jobstep + "/pressure",           {'units': "Bar"})]
def check_energy(hierarchy, jobstep, arguments):
    jobstep, path, topology, trajectory = jobstep
    logtype                             = arguments
    if not (set([jobstep + "/time",           jobstep + "/energy_total", jobstep + "/energy_potential",
                 jobstep + "/energy_kinetic", jobstep + "/temperature",  jobstep + "/pressure"]).issubset(hierarchy)):
        log     = "{0}/{1}.log".format(path, jobstep)
        return    [(energy, (jobstep, log, logtype))]
    else:   return False

#def mindist(arguments):
#    jobstep, prefix, path = arguments
#    tpr     = "{0}/{1}.tpr".format(path, prefix)
#    xtc     = "{0}/{1}.xtc".format(path, prefix)
#    xvg     = "{0}/mindist.xvg".format(path, jobstep)
#    command = "echo 1 | g_mindist -f {0} -s {1} -od {2} -pi".format(xtc, tpr, xvg)
#    commands.getoutput(command)
#    data    = np.loadtxt(xvg, skiprows = 24)
#    os.remove(xvg)
#    return [("/" + jobstep + "/mindist/min_pbc",    data[:,1] * 10.),
#            ("/" + jobstep + "/mindist/max_int",    data[:,2] * 10.),
#            ("/" + jobstep + "/mindist/min_pbc",    {'units': "A"}),
#            ("/" + jobstep + "/mindist/max_int",    {'units': "A"})]

