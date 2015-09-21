#!/usr/bin/python
# -*- coding: utf-8 -*-
#   MDclt.secondary.assign.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-10-05
"""
Classes for state assignment

.. todo:
    - Support ignoring portion of dataset
    - Support frames_per_block and dataset extension
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import secondary, pool_director
from MDclt import Block, Block_Accumulator, Block_Acceptor
################################## FUNCTIONS ###################################
def add_parser(tool_subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*tool_subparsers*: <argparse._SubParsersAction> to which to add
        :*\*\*kwargs*:      Passed to *tool_subparsers*.add_parser(...)
    """
    from MDclt import h5_default_path, overridable_defaults

    tool_subparser = secondary.add_parser(tool_subparsers,
      name     = "assign",
      help     = "Assigns states")
    mode_subparsers = tool_subparser.add_subparsers(
      dest        = "mode",
      description = "")

    coord_subparser = secondary.add_parser(mode_subparsers,
      name = "from_coord",
      help = "Assigns states based on coordinates")
    assign_subparser  = secondary.add_parser(mode_subparsers,
      name = "from_assign",
      help = "Assigns states based on previous state assignments")

    arg_groups = {
        coord_subparser:
          {ag.title: ag for ag in coord_subparser._action_groups},
        assign_subparser:
          {ag.title: ag for ag in assign_subparser._action_groups}}

    # Input
    arg_groups[coord_subparser]["input"].add_argument(
      "-coord",
      type     = str,
      required = True,
      nargs    = "+",
      action   = h5_default_path(),
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to load coordinate")
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
    arg_groups[coord_subparser]["action"].add_argument(
      "-states",
      type     = str,
      required = True,
      nargs    = "+",
      metavar  = "NAME:{<,>}CUTOFF",
      help     = "State definitions")

    # Output
    for mode_subparser in [coord_subparser, assign_subparser]:
        arg_groups[mode_subparser]["output"].add_argument(
          "-output",
          type     = str,
          required = True,
          nargs    = "+",
          action   = overridable_defaults(nargs = 2, defaults = {1:"/"}),
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and optionally address in which to output data " +
                     "(default address: /)")

    coord_subparser.set_defaults(analysis  = coord_command_line)
    assign_subparser.set_defaults(analysis = assign_command_line)

def coord_command_line(n_cores = 1, **kwargs):
    """
    Function for command line action

    **Arguments:**
        :*n_cores*: Number of cores to use
    """
    generator = Assign_Block_Generator(**kwargs)
    acceptor  = Block_Acceptor(outputs = generator.outputs, **kwargs)

    if n_cores == 1:                # Serial
        for block in generator:
            block()
            acceptor.send(block)
    else:                           # Parallel (processes)
        from multiprocessing import Pool
        pool = Pool(n_cores)
        for block in pool.imap_unordered(pool_director, generator):
            acceptor.send(block)
        pool.close()
        pool.join()

    acceptor.close()

def assign_command_line(**kwargs):
    """
    Function for command line action

    **Arguments:**
        :*n_cores*: Number of cores to use
    """
    analyzer = Assign_Analyzer(**kwargs)
    analyzer()
    acceptor  = Block_Acceptor(outputs = analyzer.outputs, **kwargs)
    acceptor.send(analyzer)
    acceptor.close()

################################### CLASSES ####################################
class Assign_Block_Generator(secondary.Secondary_Block_Generator):
    """
    Generator class that prepares blocks of analysis
    """
    def __init__(self, coord, states, output, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*coord*:  Coordinates used to make assignments
            :*output*: Tuple including path to h5 file and address
                       within h5 file
            :*force*:  Run analysis even if no new data is present
        """
        from h5py import File as h5
        from MDclt import parse_states

        # Input
        # In the long term, it is probably appropriate to have some
        #   settings to control how multiple molecules are handled
        # Also necessary to handle multiple coordinate dimensions
        #   appropriately
        self.coord_path, self.coord_address = coord
        self.inputs  = [(self.coord_path, self.coord_address)]
        with h5(self.coord_path) as coord_h5:
            coord_shape           = coord_h5[self.coord_address].shape
        self.i = 0
        if len(coord_shape) > 1:
            self.n_molecule_1 = coord_shape[1]
        else:
            self.n_molecule_1 = 1
        self.j = 0
        if len(coord_shape) > 2:
            self.n_molecule_2 = coord_shape[2]
        else:
            self.n_molecule_2 = 1

        # Action
        self.frames_per_block = coord_shape[0] # Use whole trajectory
        self.states           = parse_states(states)

        # Output
        output[1]    = output[1].rstrip("assignment")
        self.outputs = [(output[0], os.path.normpath(output[1] + "//assignment"),
          coord_shape)]

        super(Assign_Block_Generator, self).__init__(inputs = self.inputs,
          outputs = self.outputs, **kwargs)

        # Does not yet support extension, must recalculate entire
        #   dataset
        if self.preexisting_slice != self.final_slice:
            self.incoming_slice = self.final_slice

        if self.incoming_slice is not None:
            self.current_start = self.incoming_slice.start
            self.current_stop  = self.incoming_slice.start + \
                                 self.frames_per_block
            self.final_stop    = self.final_slice.stop

    def next(self):
        """
        Prepares and returns next Block of analysis
        """
        from h5py import File as h5

        if (self.incoming_slice is None
        or  self.current_start >= self.final_stop):
            raise StopIteration()
        else:
            # Determine slice indexes
            if self.n_molecule_1 >= 1 and self.n_molecule_2 >= 1:
                block_slice = (slice(self.current_start, self.current_stop, 1),
                                self.i, self.j)
                attrs       = {"slice": str(block_slice[0])}
            else:
                block_slice = slice(self.current_start, self.current_stop, 1)
                attrs       = {"slice": str(block_slice)}

            # Load primary data from these indexes
            #   NOTE: It is necessary to round to the scaleoffset in
            #   order to ensure that the same results are obtained for
            #   fresh and extended datasets
            with h5(self.coord_path) as coord_h5:
                scaleoffset = coord_h5[self.coord_address].scaleoffset
                print(scaleoffset, self.coord_address, block_slice)
                block_coord = np.array(coord_h5[self.coord_address]
                                [block_slice])
                if scaleoffset is not None:
                    scaleoffset = int(str(scaleoffset)[0])
                    block_coord = np.round(block_coord, scaleoffset)

            # Iterate
            self.j     += 1
            if self.j  == self.n_molecule_2:
                self.i += 1
                self.j  = 0
            if self.i == self.n_molecule_1:
                self.current_start += self.frames_per_block
                self.current_stop   = min(self.current_start + 
                                        self.frames_per_block, self.final_stop)
                self.i  = 0
                self.j  = 0

            # Return new block
            return Assign_Block(
              coord   = block_coord,
              states  = self.states,
              slc     = block_slice,
              outputs = self.outputs,
              attrs   = attrs)

class Assign_Block(Block):
    """
    Independent block of analysis for state assignment
    """
    def __init__(self, coord, states, outputs, slc, attrs = {}, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:  Coordinates
            :*states*: States in which to classify coordinates
            :*output*: For each dataset, path to h5 file, address
                       within h5 file, and if appropriate final
                       shape of dataset; list of tuples
            :*slc*:    Slice of final dataset to which this block
                       corresponds
            :*attrs*:  Attributes to add to dataset
    
        """
        # Input
        self.coord    = coord
        self.n_frames = self.coord.shape[0]
        if   isinstance(slc, slice):
            self.i, self.j = 0, 0
        else:
            self.i, self.j = slc[1], slc[2]

        # Action
        self.states = states

        # Output
        self.outputs   = outputs
        self.datasets = {self.outputs[0]: dict(slc = slc, attrs = attrs)}

        super(Assign_Block, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs block of analysis
        """
        coord    = self.coord
        n_frames = self.n_frames
        states   = self.states
        n_states = len(self.states)

        # Assign to states (0 = unassigned)
        in_states      = np.zeros((n_frames, n_states + 1), np.int8)
        in_states[:,0] = 1

        for s, state in enumerate(states, 1):
            dimensions = states[state]
            in_state_s = np.zeros((n_frames, len(dimensions)), np.int8)
            for d, dimension in enumerate(dimensions):
                inner, outer = dimensions[dimension]
                in_state_s[np.logical_and(
                  coord >  inner,
                  coord <= outer), d] = 1
            in_state_s = np.all(in_state_s, axis = 1)
            in_states[in_state_s, s] = 1
            in_states[in_state_s, 0] = 0
        if not np.all(np.sum(in_states, axis = 1) == 1):
            raise ValueError("System in multiple states simultaneously")
        else:
            assigned = np.zeros(n_frames, np.int8)
            for s, state in enumerate(states, 1):
                assigned[np.where(in_states[:,s] == 1)] = s

        # Handle unassigned frames
        # Note: It may be preferable to handle this in an accumulator
        #       or acceptor; this would allow time series to be
        #       divided into segments to be analyzed by different
        #       cores
        color = True    # Stay in state until reaching next state;
        if color:       #   may add ability to do otherwise later

            # Locate transitions
            transitions = []
            for source in range(n_states + 1):
                for dest in range(n_states + 1):
                    if source == dest:
                        continue
                    indexes = np.where(np.logical_and(
                      in_states[ :-1, source] == 1,
                      in_states[1:,   dest]   == 1))[0] + 1
                    transitions += [np.column_stack((
                      np.ones(indexes.size, np.int) * (source),
                      np.ones(indexes.size, np.int) * (dest),
                      indexes))]
            transitions = np.concatenate(transitions)
            transitions = transitions[transitions[:,2].argsort()]

            # Assign unassigned frames to last state
            for i, transition in enumerate(transitions):
                source, dest, exit_0 = transition
                if source == 0:
                    source, dest, enter_0 = transitions[i - 1]
                    assigned[enter_0:exit_0] = source

        # Organize data
        self.datasets[self.outputs[0]]["data"] = assigned
        self.datasets[self.outputs[0]]["attrs"]["states"] = str(
          list(self.states.items()))

class Assign_Analyzer(secondary.Secondary_Block_Generator):
    """
    Assigns states based on prior state assignments

    While this is a subclass of Secondary_Block_Generator, this is only
    to inherit the input and output management functions and this class
    does not actually function as a block generator.
    """
    def __init__(self, assign, output, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*output*: List including path to output h5 file and
                       address of output within h5 file
        """
        from MDclt import clean_path

        # Input
        assign[1] = clean_path(assign[1], strip = "assignment")
        self.assign_path, self.assign_address = assign
        self.inputs = [(self.assign_path, self.assign_address + "assignment")]

        # Output
        output[1]    = clean_path(output[1], strip = "assignment")
        self.outputs = [(output[0], output[1] + "assignment")]

        super(Assign_Analyzer, self).__init__(**kwargs)

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
        return str(self)

    def __call__(self, **kwargs):
        """
        Runs analysis
        """
        from h5py import File as h5

        if self.incoming_slice is None:
            return

        # Load input data
        with h5(self.assign_path) as assign_h5:
            in_assign    = np.array(assign_h5[self.assign_address
                             + "assignment"])
            in_assign_at = dict(assign_h5[self.assign_address
                             + "assignment"].attrs)
        self.orig_states = eval(in_assign_at.pop("states").replace("inf",
                             "np.inf"))

        if  (self.orig_states[0][0] == "unbound"
        and  self.orig_states[1][0] == "bound"):
            out_assign  = np.array(np.sum(in_assign == 2, axis = 2), np.int8)+1
            self.states = [("unbound", self.orig_states[0][1])] + \
              [("{0} bound".format(i), self.orig_states[1][1]) 
              for i in range(1, np.max(out_assign + 1))]
        else:
            raise Exception("Unable to understand state descriptions, " +
              "necessary logic not yet implemented")
        out_assign_at = dict(
          slice  = in_assign_at.pop("slice"),
          states = str(self.states))
        self.datasets = {self.outputs[0]: dict(data = out_assign,
                          attrs = out_assign_at)}


