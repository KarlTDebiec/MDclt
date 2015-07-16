# -*- coding: utf-8 -*-
#   MDclt.primary.__init__.py
#
#   Copyright (C) 2012-2015 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
"""
Classes and functions for primary analysis of molecular dynamics
simulations

.. todo:
    - Support alternatives to -infiles, such as specifying a path and
      function to list segments
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import Block_Generator
################################## FUNCTIONS ###################################
def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: <argparse._SubParsersAction> to which to add
        :*\*args*:     Passed to *subparsers*.add_parser(...)
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    from MDclt import h5_default_path, overridable_defaults

    subparser  = subparsers.add_parser(*args, **kwargs)
    arg_groups = {"input":  subparser.add_argument_group("input"),
                  "action": subparser.add_argument_group("action"),
                  "output": subparser.add_argument_group("output")}

    arg_groups["input"].add_argument(
      "-infiles",
      type     = str,
      required = True,
      nargs    = "*",
      help     = "Input filenames")

    arg_groups["action"].add_argument(
      "-n_cores",
      type     = int,
      required = False,
      default  = 1,
      help     = "Number of cores on which to carry out analysis")

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
class Primary_Block_Generator(Block_Generator):
    """
    Generator class that prepares blocks of analysis
    """

    def __init__(self, force = False, **kwargs):
        """
        Initialize generator

        **Arguments:**
            :*force*:   Overwrite data even if already present

        .. todo:
            - Make more general
            - Currently assumes constant frames_per_file
        """
        from warnings import warn
        from h5py import File as h5

        # Input
        self.get_final_slice(**kwargs)

        # Output
        self.get_preexisting_slice(force = force, **kwargs)

        if self.preexisting_slice is None:
            self.start_index = 0
        elif self.final_slice.stop > self.preexisting_slice.stop:
            self.infiles = self.infiles[
              int(self.preexisting_slice.stop / self.frames_per_file):]
            self.start_index = self.preexisting_slice.stop
        elif self.final_slice.stop < self.preexisting_slice.stop:
            warning  = "Fewer infiles provided " + \
              "({0}) ".format(len(infiles)) + \
              "than would be expected based on  " + \
              "preexisting shape of dataset {0} ".format(current_shape) + \
              "and specified number of frames per " + \
              "file ({0})".format(frames_per_file)
            warn(warning)
            self.infiles = []
        else:
            self.infiles = []

        super(Primary_Block_Generator, self).__init__(force = force, **kwargs)

    def get_final_slice(self, **kwargs):
        self.final_slice = slice(0, len(self.infiles) * self.frames_per_file, 1)

