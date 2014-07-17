#!/usr/bin/python
#   MDclt.primary.__init__.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-10
"""
Classes and functions for primary analysis of molecular dynamics simulations

.. todo:
    - Support alternatives to -infiles, such as specifying a path and function to list segments
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
      "-output",
      type     = str,
      required = True,
      nargs    = 2,
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address at which to output data")
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
class Block_Generator(Block_Generator):
    """
    Generator class that yields blocks of analysis
    """

    def __init__(self, output, attrs = {}, force = False, **kwargs):
        """
        Initialize generator

        **Arguments:**
            :*output*:          List including path to h5 file and
                                address within h5 file
            :*dataset_kwargs*:  Keyword arguments to be passed to
                                create_dataset(...)
            :*attrs*:           Attributes to be added to dataset
            :*force*:           Run analysis even if all data is
                                already present

        .. todo:
            - Make more general
            - Support multiple datasets with multiple addresses,
              probably using syntax similar to block_storer
            - Allow specification of where attrs should be stored, so
              that it is not necessary to override this method for
              this basic feature
            - Currently assumes constant frames_per_file
        """
        from h5py import File as h5
        from warnings import warn

        # Requires: dtype, final_shape, dataset_kwargs, frames_per_file
        #           infiles
        # Modifies: infiles
        # Provides: out_path, out_address, start_index

        # Output
        self.out_path    = output[0]
        self.out_address = output[1]

        with h5(self.out_path) as out_h5:
            if self.out_address in out_h5:
                if force:
                    # Dataset exists but will be overwritten

                    del out_h5[self.out_address]
                    out_h5.create_dataset(self.out_address,
                      data = np.empty(self.final_shape, self.dtype),
                      **self.dataset_kwargs)
                    self.start_index  = 0
                else:
                    dataset           = out_h5[self.out_address]
                    preexisting_shape = dataset.shape
                    if self.final_shape[0] > preexisting_shape[0]:
                        # Dataset exists, and will be extended

                        dataset.resize(size = self.final_shape)
                        self.infiles = self.infiles[
                          int(preexisting_shape[0] / self.frames_per_file):]
                        self.start_index  = preexisting_shape[0]
                    elif self.final_shape[0] < preexisting_shape[0]:
                        # Dataset is longer than expected; do nothing

                        warning  = "Fewer infiles provided ({0}) ".format(
                          len(self.infiles))
                        warning += "than would be expected based on  "
                        warning += "preexisting shape of dataset {0} ".format(
                          current_shape)
                        warning += "and specified number of frames per "
                        warning += "file ({0})".format(self.frames_per_file)
                        warn(warning)
                        self.infiles = []
                    else:
                        # Dataset is already expected size

                        self.infiles = []
            else:
                # Dataset did not previously exist

                out_h5.create_dataset(self.out_address,
                  data = np.empty(self.final_shape, self.dtype),
                  **self.dataset_kwargs)
                self.start_index  = 0
            for key, value in attrs.items():
                out_h5[self.out_address].attrs[key] = value


