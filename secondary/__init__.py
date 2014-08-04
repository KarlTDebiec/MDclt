#!/usr/bin/python
#   MDclt.secondary.__init__.py
#   Written by Karl Debiec on 14-07-06, last updated by Karl Debiec on 14-08-04
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
        :*\*args*:     Passed to *subparsers*.add_parser(...)
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    subparser = subparsers.add_parser(**kwargs)
    arg_groups = {"input":  subparser.add_argument_group("input"),
                  "action": subparser.add_argument_group("action"),
                  "output": subparser.add_argument_group("output")}

    arg_groups["input"].add_argument(
      "-log",
      type     = str,
      required = True,
      nargs    = 2,
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to load simulation log")

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
class Secondary_Block_Generator(Block_Generator):
    """
    Generator class that yields blocks of analysis
    """
    def __init__(self, inputs, output, force = False, **kwargs):
        from warnings import warn
        from h5py import File as h5

        # Input
        in_sizes = []
        for in_path, in_address in inputs:
            with h5(in_path) as in_h5:
                in_sizes += [in_h5[in_address].shape[0]]
        in_sizes = np.array(in_sizes)
        if np.unique(np.array(in_sizes)).size != 1:
            warn("Size of input datasets ({0}) inconsistent; ".format(
              in_shapes) + "using smallest size")
        self.final_index = np.min(in_sizes)

        # Output
        out_path, out_address = output
        with h5(out_path) as out_h5:
            if force or not out_address in out_h5:
                self.start_index       = 0
                self.preexisting_slice = None
            else:
                dataset                = out_h5[out_address]
                self.preexisting_slice = eval(dataset.attrs["slice"])
                self.start_index       = self.preexisting_slice.stop

            if self.start_index == self.final_index:
                self.incoming_slice = None
            else:
                self.incoming_slice = slice(self.start_index,
                                            self.final_index, 1)

        super(Secondary_Block_Generator, self).__init__(
          force = force, **kwargs)


