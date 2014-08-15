#!/usr/bin/python
#   MDclt_bridge.py
#   Written by Karl Debiec on 14-08-14, last updated by Karl Debiec on 14-08-14
"""
Command Line Tool to convert analysis conducted with previous version
of scripts (i.e. divided into segments) into current format (i.e.
single large datasets)
"""
################################### MODULES ####################################
from __future__ import division, print_function
import argparse, operator, os, sys
import numpy as np
from MDclt import Block, Block_Acceptor
################################## FUNCTIONS ###################################
def bridge(input, output, input2 = None, **kwargs):
    """
    """
    def load_input(in_path, in_address, table = False, **kwargs):
        """
        """
        from MD_toolkit.HDF5_File import HDF5_File as h5

        with h5(in_path) as in_h5:
            if table: dataset = in_h5.load(in_address, type = "table")
            else:     dataset = in_h5.load(in_address, type = "array")
            attrs = in_h5.attrs("0000/{0}".format(in_address[2:]))
        print("Dataset loaded from {0}[{1}]".format(in_path, in_address))
        return dataset, attrs

    # Load data
    dataset, attrs = load_input(in_path = input[0], in_address = input[1], **kwargs)
    if input2 is not None:
        dataset2, _ = load_input(in_path = input2[0], in_address = input2[1],
          **kwargs)
        stacked = np.column_stack((dataset, dataset2))
        minimum = np.min(stacked , axis = 1)
        dataset = minimum[:, np.newaxis]

    block = Bridge_Block(dataset = dataset, attrs = attrs, address = output[1])
        

    block_acceptor = Block_Acceptor(out_path = output[0])
    block_acceptor.next()
    block_acceptor.send(block)
    block_acceptor.close()

################################### CLASSES ####################################
class Bridge_Block(Block):
    """
    """
    def __init__(self, dataset, attrs, address, **kwargs):
        """
        """
        self.datasets = {address: dict(
          data  = dataset,
          attrs = attrs)}

    def __call__(self, **kwargs):
        """
        """
        pass
##################################### MAIN #####################################
if __name__ == "__main__":

    # Prepare argument parser
    parser            = argparse.ArgumentParser(
      description     = __doc__,
      formatter_class = argparse.RawTextHelpFormatter)
    arg_groups = {"input":  parser.add_argument_group("input"),
                  "output": parser.add_argument_group("output")}
    arg_groups["input"].add_argument(
      "-input",
      type     = str,
      required = True,
      nargs    = 2,
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to load dataset; " + 
                 "address in form of '*/DATASET_NAME'")
    arg_groups["input"].add_argument(
      "-input2",
      type     = str,
      required = False,
      nargs    = 2,
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to second dataset; " + 
                 "Output dataset will be minimum of input and input2; " +
                 "address in form of '*/DATASET_NAME'")
    arg_groups["input"].add_argument(
      "--table",
      action   = "store_true",
      help     = "Dataset is a table (numpy record array)")
    arg_groups["output"].add_argument(
      "-output",
      type     = str,
      required = True,
      nargs    = 2,
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address at which to output dataset")

    # Parse arguments
    kwargs = vars(parser.parse_args())

    # Run selected analysis
    bridge(**kwargs)


