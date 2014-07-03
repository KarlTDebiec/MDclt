#!/usr/bin/python
#   MDclt.primary.amber.py
#   Written by Karl Debiec on 12-12-01, last updated by Karl Debiec on 14-07-03
"""
Functions for primary analysis of AMBER trajectories
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt.primary import Primary_Analysis
####################################################### CLASSES ########################################################
class Log(Primary_Analysis):
    """
    Parses AMBER simulation logs

    .. todo:
        - Properly support appending data rather than reloading complete dataset
        - Move to amber submodule
    """

    @classmethod
    def add_parser(self, subparsers, *args, **kwargs):
        """
        Adds subparser arguments and argument groups to an argument parser

        **Arguments:**
            :*subparsers*: argparse subparsers object to add subparser
            :*args*:       Passed to subparsers.add_parser(...)
            :*kwargs*:     Passed to subparsers.add_parser(...)
        """
        super(Log, self).add_parser(subparsers, name = "log",
          help = "Load AMBER logs")
        self.parser.set_defaults(analysis = Log)
        self.parser_input.add_argument("-start_time", type = float, required = False,
          help = "Desired time of first frame (optional)")

    def check(self, infiles, h5_file, destination, frames_per_file = None, force = False, *args, **kwargs):
        """
        Determines whether or not to run analysis

        **Arguments:**
            :*infiles*:         Text infiles
            :*h5_file*:         Target h5 file
            :*destination*:     Intended location of new dataset within h5 file
            :*frames_per_file*: Number of frames of data in each infile (optional)
            :*force*:           Force analysis to run even if data already present

        **Returns:**
            :*run*:             True if analysis should be run, false otherwise

        .. todo:
            - Implement
        """
        return True

    def run(self, infiles, destination, start_time = 0.001, *args, **kwargs):
        """
        Runs analysis and returns new datasets

        **Arguments:**
            :*infiles*:     Text infile(s)
            :*destination*: Destination of dataset within h5 file
            :*start_time*:  Time of first frame (optional)

        **Returns:**
            :*new_data*:    List of tuples of new data in form of [(path1, dataset1), (path2, dataset2), ...]
        """

        # Load raw data from each infile
        data = {}
        for infile in infiles:
            with open(infile, "r") as infile:
                raw = [line.strip() for line in infile.readlines()]
            if not raw[-1].startswith("|  Master Total wall time:"):
                break
            i = 0
            while i < len(raw):
                line    = raw[i]
                if   line.startswith("A V E R A G E S"):
                    break
                elif line.startswith("NSTEP"):
                    while True:
                        line = raw[i]
                        if line.startswith("------------------------------------------------------------------------------"):
                            break
                        else:
                            line = line.split("=")
                            for j in range(1, len(line)):
                                if j == 1:            key = line[j-1].strip()
                                else:
                                    key                   = line[j-1].split()
                                    if len(key) == 1: key = key[0]
                                    else:             key = " ".join(key[1:])
                                value = float(line[j].split()[0])
                                if key in data: data[key] += [value]
                                else:           data[key]  = [value]
                            i += 1
                i += 1

        # Format and return data
        data["time"] = np.array(data["TIME(PS)"])  / 1000.
        if start_time is not None:
            data["time"] -= (data["time"][0] - start_time)
        new_keys = ["time"]
        dtype    = [("time", "f4")]
        attrs    = {"time units": "ns"}
        fields   = [("Etot",                    "total",                         "kcal mol-1"),
                    ("EPtot",                   "potential",                     "kcal mol-1"),
                    ("EKtot",                   "kinetic",                       "kcal mol-1"),
                    ("BOND",                    "bond",                          "kcal mol-1"),
                    ("ANGLE",                   "angle",                         "kcal mol-1"),
                    ("DIHED",                   "dihedral",                      "kcal mol-1"),
                    ("EELEC",                   "coulomb",                       "kcal mol-1"),
                    ("1-4 EEL",                 "coulomb 1-4",                   "kcal mol-1"),
                    ("VDWAALS",                 "van der Waals",                 "kcal mol-1"),
                    ("1-4 NB",                  "van der Waals 1-4",             "kcal mol-1"),
                    ("EHBOND",                  "hydrogen bond",                 "kcal mol-1"),
                    ("RESTRAINT",               "position restraint",            "kcal mol-1"),
                    ("EKCMT",                   "center of mass motion kinetic", "kcal mol-1"),
                    ("VIRIAL",                  "virial",                        "kcal mol-1"),
                    ("EPOLZ",                   "polarization",                  "kcal mol-1"),
                    ("TEMP(K)",                 "temperature",                   "K"),
                    ("PRESS",                   "pressure",                      "bar"),
                    ("Dipole convergence: rms", "dipole convergence rms",        None),
                    ("iters",                   "dipole convergence iterations", None)]
        for raw_key, new_key, units in fields:
            if raw_key in data:
                data[new_key] = np.array(data[raw_key])
                new_keys     += [new_key]
                dtype        += [(new_key, "f4")]
                if units is not None:
                    attrs[new_key + " units"] = units
        new_data = np.zeros(len(data["time"]), dtype)
        for key in new_keys:
            new_data[key] = data[key]
        return [(destination, new_data)]


