#!/usr/bin/python
#   MDclt.secondary.__init__.py
#   Written by Karl Debiec on 14-07-06, last updated by Karl Debiec on 14-07-10
"""
Classes and functions for secondary analysis of molecular dynamics simulations
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
################################## FUNCTIONS ###################################
def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: <argparse._SubParsersAction> to which to add
        :*\*args*:     Passed to *subparsers*.add_parser(...)
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    subparser = subparsers.add_parser(*args, **kwargs)
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


