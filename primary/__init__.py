#!/usr/bin/python
#   MDclt.primary.__init__.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-03
"""
Classes and functions for primary analysis of molecular dynamics simulations
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
####################################################### CLASSES ########################################################
class Primary_Analysis(object):
    """
    Base class for direct analyses of molecular dynamics trajectory, logs, or other output, including output from other
    analysis programs

    .. todo:
        - Rethink structure of information sent to add(...), list of tuples too limited
        - Should support receiving blocks of data rather than full datasets, as well as their position within the full
          dataset
        - Accept datasets and attrs at the same time
        - Accept kwargs to send to create_dataset(...)
        - Consider implementing generator version of check(...) (as a classmethod?) that yields blocks of analysis in
          the form of additional Primary_Analysis subclass objects
        - Analysis functions should be written to act on arbitrary number of infiles (i.e. write one function for serial
          or parallel blocks)
        - Need some more intelligent way of operating on h5 files to support use of multiple cores
        - Primary_Analysis subclasses may yield new data as well as function to add new data; once a 'central' function
          that is not tied to a particular Primary_Analysis object receives the data it adds it using the function
        - Likely requires function to presize dataset in h5 file, run before 
    """

    @classmethod
    def add_parser(self, subparsers, *args, **kwargs):
        """
        Adds subparser arguments and argument groups to an argument parser

        **Arguments:**
            :*subparsers*: argparse subparsers object to add subparser
            :*args*:       Passed to subparsers.add_parser(...)
            :*kwargs*:     Passed to subparsers.add_parser(...)

        .. todo:
            - Should -attrs be moved to subclasses? May not always be clear where these attrs should go
        """
        self.parser = subparsers.add_parser(*args, **kwargs)

        self.parser_input = self.parser.add_argument_group("input")
        self.parser_input.add_argument("-infiles", type = str, required = True, action = "append", nargs = "*",
          help = "Input filenames")

        self.parser_action = self.parser.add_argument_group("action")

        self.parser_output = self.parser.add_argument_group("output")
        self.parser_output.add_argument("-h5_file", type = str, required = True,
          help = "H5 file in which to output data")
        self.parser_output.add_argument("-destination", type = str, required = True,
          help = "Location of dataset within h5 file")
        self.parser_output.add_argument("-attrs", type = str, required = False, action = "append", nargs = "*",
          metavar = "KEY VALUE", 
          help = "Attributes to add to dataset")
        self.parser_output.add_argument("--force", action = "store_true",
          help = "Overwrite data if already present")

    def command_line(self, *args, **kwargs):
        """
        Provides basic command line functionality
        """
        if self.check(*args, **kwargs):
            new_data = self.run(*args,   **kwargs)
            self.add(new_data = new_data, *args,   **kwargs)

    def check(self, *args, **kwargs):
        """
        Determines whether or not to run analysis
        """
        return True

    def run(self, *args, **kwargs):
        """
        Runs analysis and returns new datasets
        """
        return []

    def add(self, h5_file, new_data, attrs = {}, *args, **kwargs):
        """
        Adds new datasets and attributes to h5 file

        .. todo:
            - Support better handling of attributes
        """
        import types
        import h5py

        with h5py.File(h5_file) as h5_file:
            for destination, dataset in new_data:
                if (destination in dict(h5_file)):
                    del h5_file[destination]
                new_dataset = h5_file.create_dataset(name = destination, data = dataset,
                            dtype = dataset.dtype, compression = "lzf")
                print("Dataset '{0}' of shape '{1}' added to {2}".format(destination, new_dataset.shape,
                  h5_file.filename))
            if attrs is not None:
                for key, value in attrs.items():
                    new_dataset.attrs[key] = value


