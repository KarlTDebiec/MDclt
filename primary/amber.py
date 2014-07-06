#!/usr/bin/python
#   MDclt.primary.amber.py
#   Written by Karl Debiec on 12-12-01, last updated by Karl Debiec on 14-07-05
"""
Class for transfer of AMBER simulation logs to h5
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt.primary import Primary_Analysis
####################################################### CLASSES ########################################################
class Log(Primary_Analysis):
    """
    Class for transfer of AMBER simulation logs to h5

    .. todo:
        - Better support appending data rather than reloading complete dataset
    """

    @classmethod
    def add_parser(cls, subparsers, *args, **kwargs):
        """
        Adds subparser for this analysis to a nascent argument parser

        **Arguments:**
            :*subparsers*: argparse subparsers object to add subparser
            :*args*:       Passed to subparsers.add_parser(...)
            :*kwargs*:     Passed to subparsers.add_parser(...)

        .. todo:
            - Implement nested subparser (should be 'amber log', not just 'log')
        """
        subparser = super(Log, cls).add_parser(subparsers, name = "log",
          help = "Load AMBER logs")
        arg_groups = {ag.title:ag for ag in subparser._action_groups}
        arg_groups["input"].add_argument("-frames_per_file", type = int, required = False,
          help = "Number of frames in each file; used to check if new data is present (optional)")
        arg_groups["input"].add_argument("-start_time", type = float, required = False,
          help = "Desired time of first frame (ns) (optional)")
        subparser.set_defaults(analysis = cls.command_line)

    class block_generator(Primary_Analysis.block_generator):
        """
        Generator class that yields blocks of analysis

        This is a class rather than a function, because it needs to perform initialization before the first call to
        next().
        """

        fields = [("Etot",                    "total",                         "kcal mol-1"),
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

        def __init__(self, address, infiles, frames_per_file = None, **kwargs):
            """
            Initializes block_generator; temporarily opens h5 file to examine current contents; stores attrs passed
            from the command line

            **Arguments:**
                :*h5_file*:         Filename of h5 file in which data will be stored
                :*address*:         Address of dataset within h5 file
                :*infiles*:         List of infiles
                :*frames_per_file*: Number of frames in each infile (optional)
                :*dimensions*:      Additional dimensions in dataset; if multidimensional (optional)
                :*attrs*:           Attributes to be added to dataset
                :*force*:           Run analysis even if all data is already present

            .. todo:
                - Intelligently break lists of infiles into blocks larger than 1?
                - If data is being appended, do not determine dtype again
            """

            # Store necessary information in instance variables
            self.address         = address
            self.infiles         = infiles
            self.frames_per_file = frames_per_file
            self.expected_shape  = [len(self.infiles) * self.frames_per_file]
            self.current_index   = 0

            # Adjust start time, if applicable
            self.get_time_offset(**kwargs)

            # Determine structure of input data
            self.get_dataset_format(**kwargs)

            # Disregard last infile, if applicable
            self.cut_incomplete_infiles(**kwargs)

            # Complete initialization 
            dataset_kwargs = dict(chunks = True, compression = "gzip", maxshape = [None])
            self._initialize(address = address, dataset_kwargs = dataset_kwargs, **kwargs)

        def next(self):
            """
            Prepares and yields next block of analysis; calculates appropriate slice indices for storage
            """
            if len(self.infiles) == 0:
                raise StopIteration()
            else:
                slice_start         = self.current_index
                slice_end           = self.current_index + self.frames_per_file
                self.current_index += self.frames_per_file
                return Log(infiles     = [self.infiles.pop(0)],
                           raw_keys    = self.raw_keys,
                           new_keys    = self.new_keys,
                           dtype       = self.dtype,
                           address     = self.address,
                           slc         = slice(slice_start, slice_end, 1),
                           time_offset = self.time_offset)

        def get_time_offset(self, start_time = None, **kwargs):
            """
            Calculates time adjustment based on desired and actual time of first frame

            **Arguments:**
                :*start_time*: Desired time of first frame (ns); typically 0.001
            """
            import subprocess

            if start_time is None:
                self.time_offset = 0
            else:
                with open(os.devnull, "w") as fnull:
                    command = "cat {0} | grep -m 1 'TIME(PS)' | awk '{{print $6}}'".format(self.infiles[0])
                    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = fnull, shell = True)
                    result  = process.stdout.read()
                self.time_offset = float(result) / - 1000 + start_time
 
        def get_dataset_format(self, **kwargs):
            """
            Determines format of dataset
            """

            # Determine fields present in infile
            raw_keys = []
            breaking = False
            with open(self.infiles[0], "r") as infile:
                raw_text = [line.strip() for line in infile.readlines()]
            for i in xrange(len(raw_text)):
                if breaking:  break
                if raw_text[i].startswith("NSTEP"):
                    while True:
                        if raw_text[i].startswith("----------"):
                            breaking = True
                            break
                        for j, field in enumerate(raw_text[i].split("=")[:-1]):
                            if j == 0:
                                raw_keys += [field.strip()]
                            else:
                                raw_keys += [" ".join(field.split()[1:])]
                        i += 1

            # Determine appropriate dtype of new data
            self.raw_keys = ["TIME(PS)"]
            self.new_keys = ["time"]
            self.dtype    = [("time", "f4")]
            self.attrs    = {"time units": "ns"}
            for raw_key, new_key, units in self.fields:
                if raw_key in raw_keys:
                    self.raw_keys  += [raw_key]
                    self.new_keys  += [new_key]
                    self.dtype     += [(new_key, "f4")]
                    if units is not None:
                        self.attrs[new_key + " units"] = units

        def cut_incomplete_infiles(self, **kwargs):
            """
            Checks last infile; if the log is not complete, removes from list of infiles to analyze
            """
            import subprocess

            with open(os.devnull, "w") as fnull:
                command = "tail -n 1 {0}".format(self.infiles[-1])
                process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = fnull, shell = True)
                result  = process.stdout.read()
            if not result.startswith("|  Master Total wall time:"):
                self.infiles.pop(-1)


    def __init__(self, infiles, raw_keys, new_keys, dtype, address, slc, time_offset = 0, *args, **kwargs):
        """
        Initializes a block of analysis; prepares instance variable to store dataset

        **Arguments:**
            :*infiles*:  List of infiles
            :*address*:  Address of dataset within h5 file
            :*slc*:      Slice within dataset at which this block will be stored
        """
        super(Log, self).__init__(*args, **kwargs)
        self.infiles     = infiles
        self.raw_keys    = raw_keys
        self.new_keys    = new_keys
        self.datasets    = [dict(address = address, slc = slc, attrs = {},
                            data = np.empty(slc.stop - slc.start, dtype))]
        self.time_offset = time_offset

    def analyze(self, **kwargs):
        """
        Runs this block of analysis; stores resulting data in an instance variable
        """

        # Load raw data from each infile
        raw_data = {raw_key:[] for raw_key in self.raw_keys}
        for infile in self.infiles:
            with open(infile, "r") as infile:
                raw_text = [line.strip() for line in infile.readlines()]
            i = 0
            while i < len(raw_text):
                if raw_text[i].startswith("A V E R A G E S"): break
                if raw_text[i].startswith("NSTEP"):
                    while True:
                        if raw_text[i].startswith("----------"): break
                        line = raw_text[i].split("=")
                        for j, field in enumerate(line[:-1]):
                            if j == 0:
                                raw_key = field.strip()
                            else:
                                raw_key = " ".join(field.split()[1:])
                            value = line[j+1].split()[0]
                            if raw_key in self.raw_keys:
                                raw_data[raw_key] += [value]
                        i += 1
                i += 1

        # Copy from raw_data to new_data
        self.datasets[0]["data"]["time"] = (np.array(raw_data["TIME(PS)"], np.float) / 1000) + self.time_offset
        for raw_key, new_key in zip(self.raw_keys[1:], self.new_keys[1:]):
            self.datasets[0]["data"][new_key] = np.array(raw_data[raw_key])


