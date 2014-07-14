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

    def __init__(self, output, dataset_kwargs = dict(chunks = True,
                 compression = "gzip"), attrs = {}, force = False,
                 **kwargs):
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
        """
        from h5py import File as h5

        out_path, out_address = output

        with h5(out_path) as out_h5:
            if out_address in out_h5:
                if force:
                    del out_h5[out_address]
                    out_h5.create_dataset(out_address,
                      data = np.empty(self.expected_shape, self.dtype),
                      **dataset_kwargs)
                else:
                    dataset       = out_h5[out_address]
                    current_shape = dataset.shape
                    if self.expected_shape != current_shape:
                        dataset.resize(size = self.expected_shape)
                        self.infiles        = self.infiles[int(current_shape[0]
                                            / self.frames_per_file):]
                        self.current_index  = current_shape[0]
                    else:
                        self.infiles = []
            else:
                out_h5.create_dataset(out_address,
                  data = np.empty(self.expected_shape, self.dtype),
                  **dataset_kwargs)
            for key, value in attrs.items():
                out_h5[out_address].attrs[key] = value


