# -*- coding: utf-8 -*-
#   MDclt.secondary.stateprobs.py
#
#   Copyright (C) 2012-2015 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
"""
Classes for calculation of state probabilities
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import secondary, pool_director
from MDclt import Block, Block_Accumulator, Block_Acceptor
from MDclt.FP_Block_Averager import FP_Block_Averager
################################## FUNCTIONS ###################################
def add_parser(tool_subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*tool_subparsers*: <argparse._SubParsersAction> to which to add
        :*\*\*kwargs*:      Passed to *tool_subparsers*.add_parser(...)
    """
    from MDclt import h5_default_path, overridable_defaults

    tool_subparser = tool_subparsers.add_parser(
      name     = "stateprobs",
      help     = "Calculates state probabilities")
    mode_subparsers = tool_subparser.add_subparsers(
      dest        = "mode",
      description = "")

    pdist_subparser = secondary.add_parser(mode_subparsers,
      name = "from_pdist",
      help = "Calculates state probabilities using the probability "
             "density function")
    assign_subparser  = secondary.add_parser(mode_subparsers,
      name = "from_assignment",
      help = "Calculates state probabilities using per-frame state " +
             "assignments")

    arg_groups = {
        pdist_subparser:
          {ag.title: ag for ag in pdist_subparser._action_groups},
        assign_subparser:
          {ag.title: ag for ag in assign_subparser._action_groups}}

    # Input
    arg_groups[pdist_subparser]["input"].add_argument(
      "-pdist",
      type     = str,
      required = True,
      nargs    = "+",
      action   = h5_default_path(),
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to load probability "
                 "density function")
    arg_groups[assign_subparser]["input"].add_argument(
      "-assign",
      type     = str,
      required = True,
      nargs    = "+",
      action   = h5_default_path(),
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to load state "
                 "assignments")

    # Action
    arg_groups[pdist_subparser]["action"].add_argument(
      "-states",
      type     = str,
      nargs    = "+",
      metavar  = "NAME:{<,>}CUTOFF",
      help     = "State definitions")

    # Output
    for mode_subparser in [pdist_subparser, assign_subparser]:
        arg_groups[mode_subparser]["output"].add_argument(
          "-output",
          type     = str,
          required = True,
          nargs    = "+",
          action   = overridable_defaults(nargs = 2, defaults = {1: "/"}),
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and optionally address in which to output data " +
                     "(default address: /)")

    pdist_subparser.set_defaults(analysis  = command_line(PDist_Analyzer))
    assign_subparser.set_defaults(analysis = command_line(Assign_Analyzer))

def command_line(analyzer_class, **kwargs):
    """
    Generates function for command line action

    **Arguments:**
        :*analyzer_class*: Class to be used for analysis
    """
    def func(**kwargs):
        """
        Function for command line action
        """
        analyzer = analyzer_class(**kwargs)
        analyzer()
        acceptor  = Block_Acceptor(outputs = analyzer.outputs, **kwargs)
        acceptor.send(analyzer)
        acceptor.close()
        print(analyzer)
    return func

################################### CLASSES ####################################
class StateProb_Analyzer(secondary.Secondary_Block_Generator):
    """
    Calculates state probabilities

    While this is a subclass of Secondary_Block_Generator, this is only
    to inherit the input and output management functions and this class
    does not actually function as a block generator.
    """
    def __init__(self, output, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*output*: List including path to output h5 file and
                       address of output within h5 file
        """
        from MDclt import clean_path

        # Output
        output[1]    = clean_path(output[1], strip = "stateprobs")
        self.outputs = [(output[0], output[1] + "stateprobs")]

        super(StateProb_Analyzer, self).__init__(**kwargs)

    def next(self):
        """
        This is invoked by the __init__ method of this class' base
        class, but serves no other function.
        """
        raise StopIteration()

    def __str__(self):
        """
        Prepares string representation of output data
        """
        if hasattr(self, "datasets"):
            stateprob_ds = np.array(self.datasets.values()[0]["data"])
            titles = ""
        else:
            from h5py import File as h5
            out_path, out_address = self.outputs[0]
            with h5(out_path) as out_h5:
                if out_address in out_h5:
                    stateprob_ds = np.array(out_h5[out_address])
                    titles = "Dataset present at {0}[{1}]\n".format(
                      out_path, out_address)
                else:
                    return "Dataset {0}[{1}] does not exist".format(
                      out_path, out_address)
        values = ""
        for i, field in enumerate(stateprob_ds.dtype.names):
            if i % 2 == 0:
                titles += "{0:>12} {1:<10}  ".format(field, "(se)")
                values += "{0:>12.4f} ".format(float(stateprob_ds[field]))
            else:
                values += "{0:<12}".format("({0:<.4f})".format(
                            float(stateprob_ds[field])))
        return "{0}\n{1}".format(titles, values)

class Assign_Analyzer(StateProb_Analyzer):
    """
    Calculates state probabilities from per-frame assignments
    """
    def __init__(self, assign, **kwargs):
        """
        Initializes
        """
        from MDclt import clean_path

        # Input
        assign[1] = clean_path(assign[1], strip = "assignment")
        self.assign_path, self.assign_address = assign
        self.inputs = [(self.assign_path, self.assign_address + "assignment")]

        super(Assign_Analyzer, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs analysis
        """
        from h5py import File as h5

        if self.incoming_slice is None:
            return

        # Load input data
        with h5(self.assign_path) as assign_h5:
            assignments = np.array(assign_h5[self.assign_address
                            + "assignment"])
            attrs       = dict(assign_h5[self.assign_address
                            + "assignment"].attrs)
        n_frames    = assignments.shape[0]
        n_states    = np.max(assignments) + 1
        state_names = ["unassigned"] + [name for name, _ in eval(
                        attrs["states"].replace("inf", "np.inf"))]

        # Calculate state probabilities
        Pstate             = np.zeros(n_states, np.float64)
        Pstate_se          = np.zeros(n_states, np.float64)
        assignments_for_FP = np.zeros((n_frames, n_states), np.float64)
        for i in range(n_states):
            Pstate[i] = float(assignments[assignments == i].size) / float(
                          assignments.size)
            if len(assignments.shape) > 1:
                assignments_for_FP[:,i] = (np.sum(assignments == i, axis = 1) 
                                            / assignments.shape[1])
            else:
                assignments_for_FP[:,i] = (assignments == i)

        # If all frames are assigned, remove state 0 (unassigned)
        if Pstate[0] == 0.0:
            n_states          -= 1
            Pstate             = Pstate[1:]
            Pstate_se          = Pstate_se[1:]
            assignments_for_FP = assignments_for_FP[:,1:]
            state_names        = state_names[1:]

        # Calculate standard error
        fp_block_averager = FP_Block_Averager(
          dataset    = assignments_for_FP,
          fieldnames = state_names)
        fp_block_averager()
        Pstate_se[:] = fp_block_averager.exp_fit_parameters[0]

        # Organize data
        dtype = [field for state in
                  [[("P {0}".format(name),"f4"),("P {0} se".format(name),"f4")]
                  for name in state_names] for field in state]
        stateprobs = np.zeros(1, dtype)
        for i in range(n_states):
            stateprobs[0]["P {0}".format(state_names[i])]    = Pstate[i]
            stateprobs[0]["P {0} se".format(state_names[i])] = Pstate_se[i]
        stateprobs_at = dict(
          states = attrs["states"],
          slice  = str(self.final_slice))
        self.datasets = {self.outputs[0]: dict(data = stateprobs,
                          attrs = stateprobs_at)}
        
class PDist_Analyzer(StateProb_Analyzer):
    """
    Calculates state probabilities by integrating probability distribution
    """
    def __init__(self, pdist, states, **kwargs):
        """
        Initializes
        """
        from MDclt import clean_path, parse_states

        # Input
        pdist[1]    = clean_path(pdist[1], strip = ["pdist", "blocks"])
        self.pdist_path, self.pdist_address = pdist
        self.inputs = [(self.pdist_path, self.pdist_address + "pdist"),
                       (self.pdist_path, self.pdist_address + "blocks")]

        # Action
        self.states = parse_states(states)

        super(PDist_Analyzer, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs analysis
        """
        from h5py import File as h5

        if self.incoming_slice is None:
            return

        # Load input data
        with h5(self.pdist_path) as pdist_h5:
            pdist     = np.array(pdist_h5[self.pdist_address + "pdist"])
            pdist_at  = dict(pdist_h5[self.pdist_address + "pdist"].attrs)
            blocks    = np.array(pdist_h5[self.pdist_address + "blocks"])
            blocks_at = dict(pdist_h5[self.pdist_address + "blocks"].attrs)
        n_states    = len(self.states)
        state_names = self.states.keys()

        # Calculate state probabilities
        Pstate       = np.zeros(n_states, np.float64)
        Pstate_se    = np.zeros(n_states)
        state_slices = []
        for i, state_name in enumerate(self.states):
            dimensions = self.states[state_name]
            for d, dimension in enumerate(dimensions):
                inner, outer = dimensions[dimension]

                if "lower bound" in pdist.dtype.names:
                    min_index = np.abs(
                      pdist["lower bound"] - inner).argmin()
                    if np.isinf(outer):
                        max_index = pdist["upper bound"].size
                    else:
                        max_index = np.abs(
                          pdist["upper bound"] - outer).argmin() + 1
                    state_slice   = slice(min_index, max_index, 1)
                    state_slices += [state_slice]
                    Pstate[i] = float(
                                  np.nansum(pdist["count"][state_slice])) / \
                                float(
                                  np.nansum(pdist["count"]))
                else:
                    min_index = np.abs(
                      pdist["center"] - inner).argmin()
                    if np.isinf(outer):
                        max_index = pdist["center"].size
                    else:
                        max_index = np.abs(
                          pdist["center"] - outer).argmin() + 1
                    state_slice   = slice(min_index, max_index, 1)
                    state_slices += [state_slice]
                    Pstate[i] = float(
                                  np.nansum(pdist["kde"][state_slice])) / \
                                float(
                                  np.nansum(pdist["kde"]))

        # Calculate standard error
        fp_block_averager = PDist_FP_Block_Averager(
          dataset      = blocks[:-1],
          full_length  = blocks["stop"][-2],
          state_slices = state_slices,
          n_fields     = n_states,
          fieldnames   = state_names,
          factor       = blocks_at["block_size"])
        fp_block_averager()
        Pstate_se[:] = fp_block_averager.exp_fit_parameters[0]

        # Organize data
        dtype = [field for state in
                  [[("P {0}".format(name),"f4"),("P {0} se".format(name),"f4")]
                  for name in state_names] for field in state]
        stateprobs = np.zeros(1, dtype)
        for i in range(n_states):
            stateprobs[0]["P {0}".format(state_names[i])]    = Pstate[i]
            stateprobs[0]["P {0} se".format(state_names[i])] = Pstate_se[i]
        stateprobs_at = dict(
          states = str(self.states),
          slice  = str(self.final_slice))
        self.datasets = {self.outputs[0]: dict(data = stateprobs,
                          attrs = stateprobs_at)}

class PDist_FP_Block_Averager(FP_Block_Averager):
    """
    Class to manage estimation of standard error using the
    block-averaging method of Flyvbjerg and Petersen, from a
    probability distribution
    """
    def __init__(self, state_slices, **kwargs):
        """
        **Arguments:**
            :*state_slices*: List of slices from probability
                             distribution corresponding to each state
        """
        self.state_slices = state_slices
        super(PDist_FP_Block_Averager, self).__init__(**kwargs)

    def transform(self, block_length, n_blocks, total_length, **kwargs):
        """
        Prepares a block-transformed dataset of states from a
        probability distribution

        **Argument:**
            :*block_length*: Length of each block in transformed
                             assign
            :*n_blocks*:     Number of blocks in transformed assign 
            :*total_length*: Number of frames in transformed assign
        """
        blocks = self.dataset
        transformed = np.zeros((n_blocks, self.n_fields), np.float)
        for i in range(n_blocks):
            # Generate block-averaged probability distributions
            if "count" in blocks.dtype.names:
                pdist  = blocks["count"][np.logical_and(
                           blocks["start"] >=  i      * block_length,
                           blocks["stop"]  <= (i + 1) * block_length)]
                pdist  = np.array(np.sum(pdist, axis = 0), np.float64)
            else:
                pdist  = blocks["kde"][np.logical_and(
                           blocks["start"] >=  i      * block_length,
                           blocks["stop"]  <= (i + 1) * block_length)]
                pdist  = np.array(np.sum(pdist, axis = 0), np.float64)
            pdist /= np.sum(pdist)

            # Calcalate state probabilities in each block
            for j, state_slice in enumerate(self.state_slices):
                transformed[i, j] = np.sum(pdist[state_slice])
        return transformed


