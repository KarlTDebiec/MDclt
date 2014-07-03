#!/usr/bin/python
#   MDclt.primary.raw.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-03
"""
Class for analysis of raw text
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt.primary import Primary_Analysis
####################################################### CLASSES ########################################################
class Raw(Primary_Analysis):
    """
    Class for analysis of raw text

    .. todo:
        - Support arbitrary shape
        - Properly support appending data rather than reloading complete dataset
        - Look into improving speed
    """

    @classmethod
    def add_parser(self, subparsers):
        """
        Adds subparser arguments and argument groups to an argument parser

        **Arguments:**
            :*subparsers*: argparse subparsers object to add subparser
            :*args*:       Passed to subparsers.add_parser(...)
            :*kwargs*:     Passed to subparsers.add_parser(...)
        """
        super(Raw, self).add_parser(subparsers, name = "raw",
          help = "Load raw text files")
        self.parser_input.add_argument("-frames_per_file", type = int, required = False,
          help = "Number of frames in each file; used to check for new data (optional)")
        self.parser_input.add_argument("-width_2D", type = int, required = False,
          help = "Number of columns in dataset, if two-dimensional (optional)")
        self.parser.set_defaults(analysis = Raw)

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
        """
        import h5py

        if force:
            return True
        elif frames_per_file is None:
            return True
        else:
            expected_size = len(infiles) * frames_per_file
            with h5py.File(h5_file) as h5_file:
                if destination in h5_file:
                    current_size = h5_file[destination].shape[0]
                else:
                    current_size = 0
            if expected_size != current_size:
                return True
            else:
                return False

    def run(self, infiles, destination, width_2D = None, *args, **kwargs):
        """
        Runs analysis and returns new datasets

        **Arguments:**
            :*infiles*:     Text infile(s)
            :*destination*: Destination of dataset within h5 file
            :*width_2D*:    Number of columns in dataset (optional)

        **Returns:**
            :*new_data*:    List of tuples of new data in form of [(path1, dataset1), (path2, dataset2), ...]
        """
        import subprocess

        # Load raw data into numpy using shell commands
        # There may be a faster way to do this; but this is at least faster than np.loadtxt(...)
        command      = "cat {0} | sed ':a;N;$!ba;s/\\n//g'".format(" ".join(infiles))
        process      = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
        input_bytes  = bytearray(process.stdout.read())
        output_array = np.array(np.frombuffer(input_bytes, dtype = "S8",
                        count = int((len(input_bytes) -1) / 8)), np.float32)

        # Reshape if necessary
        if width_2D is not None:
            output_array = output_array.reshape((output_array.size / width_2D, width_2D))

        return [(destination, output_array)]



