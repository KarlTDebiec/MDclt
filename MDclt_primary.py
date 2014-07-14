#!/usr/bin/python
#   MDclt_primary.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-09
"""
Command Line Tool to manage primary analysis of molecular dynamics simulations

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
    parser     = argparse.ArgumentParser(
                   description     = __doc__,
                   formatter_class = argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(
                   dest            = "package",
                   description     = "")

    from MDclt.primary import raw
    from MDclt.primary.amber import log
    raw.add_parser(subparsers)
    log.add_parser(subparsers)

    # Parse arguments
    kwargs = vars(parser.parse_args())
    if kwargs["attrs"] is not None:
        kwargs["attrs"] = {key: value for key, value
          in zip(*[iter(kwargs["attrs"])] * 2)}
    else:
        kwargs["attrs"] = {}

    # Run selected analysis
    analysis = kwargs.pop("analysis")(**kwargs)
