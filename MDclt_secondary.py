#!/usr/bin/python
# -*- coding: utf-8 -*-
#   MDclt_secondary.py
#
#   Copyright (C) 2012-2015 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
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
    parser            = argparse.ArgumentParser(
      description     = __doc__,
      formatter_class = argparse.RawTextHelpFormatter)
    tool_subparsers   = parser.add_subparsers(
      dest            = "tool",
      description     = "")

    from MDclt.secondary import pdist
    pdist.add_parser(tool_subparsers)
    from MDclt.secondary import assign
    assign.add_parser(tool_subparsers)
    from MDclt.secondary import stateprobs
    stateprobs.add_parser(tool_subparsers)

    # Parse arguments
    kwargs = vars(parser.parse_args())
    if kwargs["attrs"] is not None:
        kwargs["attrs"] = {k: v for k, v in zip(*[iter(kwargs["attrs"])] * 2)}
    else:
        kwargs["attrs"] = {}

    # Run selected analysis
    kwargs.pop("analysis")(**kwargs)


