#!/usr/bin/python
# -*- coding: utf-8 -*-
#   MDclt.secondary.__init__.py
#   Written by Karl Debiec on 14-07-06, last updated by Karl Debiec on 14-09-30
"""
Classes and functions for secondary analysis of molecular dynamics simulations
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import Block_Generator
################################## FUNCTIONS ###################################
def add_parser(subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: <argparse._SubParsersAction> to which to add
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    from MDclt import overridable_defaults

    subparser = subparsers.add_parser(**kwargs)
    arg_groups = {"input":  subparser.add_argument_group("input"),
                  "action": subparser.add_argument_group("action"),
                  "output": subparser.add_argument_group("output")}

    arg_groups["action"].add_argument(
      "-n_cores",
      type     = int,
      required = False,
      default  = 1,
      help     = "Number of cores on which to carry out analysis (default: 1)")

    arg_groups["output"].add_argument(
      "-attrs",
      type     = str,
      required = False,
      nargs    = "*",
      metavar  = "KEY VALUE",
      help     = "Attributes to add to dataset (optional)")
    arg_groups["output"].add_argument(
      "--force",
      action   = "store_true",
      help     = "Overwrite data if already present")

    return subparser

################################### CLASSES ####################################
class Secondary_Block_Generator(Block_Generator):
    """
    Generator class that prepares blocks of analysis
    """
    def __init__(self, force = False, **kwargs):
        """
        Initializes generator

        Analyzes inputs, checking which simulation frames each input
        dataset corresponds to, and setting the expected final dataset
        slice to the frames shared by all. Analyzes outputs, checking
        which simulation frames each previously-existing dataset
        corresponds to, and setting the expected incoming dataset to
        those available from the input and not present in the output.

        **Arguments:**
            :*force*:   Overwrite data even if already present
        """
        from warnings import warn

        # Input
        self.get_final_slice(**kwargs)

        # Output
        self.get_preexisting_slice(force = force, **kwargs)

        # Action
        self.get_incoming_slice(**kwargs)

        super(Secondary_Block_Generator, self).__init__(force=force, **kwargs)

    def get_final_slice(self, debug = False, **kwargs):
        """
        """
        from warnings import warn
        from h5py import File as h5

        if debug: print(self.inputs)

        in_starts   = []
        in_stops    = []
        for in_path, in_address in self.inputs:
            with h5(in_path) as in_h5:
                attrs = dict(in_h5[in_address].attrs)
                if "slice" in attrs:
                    in_starts += [eval(attrs["slice"]).start]
                    in_stops  += [eval(attrs["slice"]).stop]
                else:
#                    warn("'slice' not found in dataset " +
#                      "'{0}:{1}' attributes; ".format(in_path, in_address) +
#                      "assuming first dimension is time")
                    in_starts += [0]
                    in_stops  += [in_h5[in_address].shape[0]]
        in_start = np.max(in_starts)
        in_stop  = np.min(in_stops)
        self.final_slice = slice(in_start, in_stop, 1)

    def get_incoming_slice(self, **kwargs):
        """
        """
        if self.preexisting_slice is None:
            self.incoming_slice = self.final_slice
        elif self.final_slice.stop > self.preexisting_slice.stop:
            self.incoming_slice = slice(self.preexisting_slice.stop,
              self.final_slice.stop, 1)
        elif self.final_slice.stop < self.preexisting_slice.stop:
            warning = "Preexisting outfile(s) appear to correspond to " + \
              "a larger slice of frames " + \
              "({0}) ".format(self.preexisting_slice) + \
              "than available in infile(s) " + \
              "({0})".format(self.final_slice)
            self.incoming_slice = None
        else:
            self.incoming_slice = None

