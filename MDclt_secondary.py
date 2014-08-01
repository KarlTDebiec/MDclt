#!/usr/bin/python
#   MDclt_secondary.py
#   Written by Karl Debiec on 14-07-06, last updated by Karl Debiec on 14-07-31
"""
Command Line Tool to manage secondary analysis of molecular dynamics simulations

.. todo:
    - Automatically add analysis functions (i.e. do not hardcode)
"""
################################### MODULES ####################################
from __future__ import division, print_function
import argparse, operator, os, sys
import numpy as np
##################################### MAIN #####################################
if __name__ == "__main__":

    # Prepare argument parser
    parser = argparse.ArgumentParser(
               description     = __doc__,
               formatter_class = argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest = "package", description = "")

    from MDclt.secondary import pmf
    from MDclt.secondary import association
    pmf.add_parser(subparsers)
    association.add_parser(subparsers)

    # Parse arguments
    kwargs = vars(parser.parse_args())
    if kwargs["attrs"] is not None:
        kwargs["attrs"] = {key: value
                            for key, value in zip(*[iter(kwargs["attrs"])] * 2)}
    else:
        kwargs["attrs"] = {}

    # Run selected analysis
    analysis = kwargs.pop("analysis")(**kwargs)


