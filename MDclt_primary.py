#!/usr/bin/python
#   MDclt_primary.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-05
"""
Command Line Tool to manage primary analysis of molecular dynamics simulations

.. todo:
    - Automatically add analysis functions (i.e. do not hardcode)
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import argparse, operator, os, sys
import numpy as np
######################################################### MAIN #########################################################
if __name__ == "__main__":

    # Prepare argument parser
    parser = argparse.ArgumentParser(description     = __doc__,
                                     formatter_class = argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest = "package", description = "")

    from MDclt.primary.raw import Raw
    from MDclt.primary.amber import Log
    Raw.add_parser(subparsers)
    Log.add_parser(subparsers)

    # Parse arguments
    kwargs = vars(parser.parse_args())
    if kwargs["infiles"] is not None:
        kwargs["infiles"] = reduce(operator.add, kwargs["infiles"])
    if kwargs["attrs"] is not None:
        attrs = reduce(operator.add, kwargs["attrs"])
        kwargs["attrs"] = {key: value for key, value in zip(*[iter(attrs)] * 2)}
    else:
        kwargs["attrs"] = {}

    # Run selected analysis
    analysis = kwargs.pop("analysis")(**kwargs)


