#!/usr/bin/python
#   MDclt.primary.raw.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-09-29
"""
Classes for transfer of data from raw text files to h5

.. todo:
    - Look into improving speed (np.loadtxt or np.genfromtxt may actually be
      preferable)
    - Alow scaleoffset to be specified
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import Block, Block_Acceptor, primary
################################## FUNCTIONS ###################################
def add_parser(tool_subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*tool_subparsers*: argparse subparsers object to add subparser
        :*args*:            Passed to tool_subparsers.add_parser(...)
        :*kwargs*:          Passed to tool_subparsers.add_parser(...)
    """
    from MDclt import overridable_defaults

    subparser  = primary.add_parser(tool_subparsers,
      name     = "raw",
      help     = "Load raw text files")
    arg_groups = {ag.title: ag for ag in subparser._action_groups}

    arg_groups["input"].add_argument(
      "-frames_per_file",
      type     = int,
      required = False,
      help     = "Number of frames in each file; used to check if new data " +
                 "is present")
    arg_groups["input"].add_argument(
      "-dimensions",
      type     = int,
      required = False,
      nargs    = "*",
      help     = "Additional dimensions in dataset; if multidimensional " +
                 "(optional)")

    arg_groups["output"].add_argument(
      "-output",
      type     = str,
      required = True,
      nargs    = "+",
      action   = overridable_defaults(nargs = 2, defaults = {1: "/dataset"}),
      help     = "H5 file and optionally address in which to output data " +
                 "(default address: /dataset)")

    subparser.set_defaults(analysis = command_line)

def command_line(n_cores = 1, **kwargs):
    """
    Provides command line functionality for this analysis

    **Arguments:**
        :*n_cores*: Number of cores to use

    .. todo:
        - Figure out syntax to get this into MDclt.primary
    """
    from multiprocessing import Pool
    from MDclt import pool_director

    block_generator = Raw_Block_Generator(**kwargs)
    block_acceptor  = Block_Acceptor(outputs = block_generator.outputs,
                        **kwargs)
    if n_cores == 1:                # Serial
        for block in block_generator:
            block()
            block_acceptor.send(block)
    else:                           # Parallel (processes)
        pool = Pool(n_cores)
        for block in pool.imap_unordered(pool_director, block_generator):
            block_acceptor.send(block)
        pool.close()
        pool.join()

    block_acceptor.close()

################################### CLASSES ####################################
class Raw_Block_Generator(primary.Primary_Block_Generator):
    """
    Generator class that prepares blocks of analysis
    """

    def __init__(self, infiles, dimensions, output, frames_per_file = None,
          **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*output*:          List including path to h5 file and
                                address within h5 file
            :*infiles*:         List of infiles
            :*frames_per_file*: Number of frames in each infile
            :*dimensions*:      Additional dimensions in dataset; if
                                multidimensional (optional)

        .. todo:
            - Intelligently break lists of infiles into blocks larger
              than 1
        """

        # Input
        self.infiles           = infiles
        self.frames_per_file   = frames_per_file
        self.infiles_per_block = 5
        if dimensions is None:
            self.dimensions = []
        else:
            self.dimensions = dimensions

        # Output
        self.outputs = [(output[0], os.path.normpath(output[1]))]

        # Action
        self.dtype = np.float32

        super(Raw_Block_Generator, self).__init__(**kwargs)

        # Output
        self.outputs = [(output[0], os.path.normpath(output[1]),
          tuple([self.final_slice.stop - self.final_slice.start]
            + self.dimensions))]

    def next(self):
        """
        Prepares and returns next Block of analysis
        """
        if len(self.infiles) == 0:
            raise StopIteration()
        else:
            block_infiles = self.infiles[:self.infiles_per_block]
            block_slice   = slice(self.start_index,
              self.start_index + len(block_infiles) * self.frames_per_file, 1)

            self.infiles      = self.infiles[self.infiles_per_block:]
            self.start_index += len(block_infiles) * self.frames_per_file

            return Raw_Block(infiles     = block_infiles,
                             output      = self.outputs[0],
                             slc         = block_slice, 
                             dimensions  = self.dimensions,
                             dtype       = self.dtype)

class Raw_Block(Block):
    """
    Independent block of analysis
    """
    def __init__(self, infiles, output, dtype, slc, dimensions = [],
          attrs = {}, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*infiles*:    List of infiles
            :*output*:     For each dataset, path to h5 file, address
                           within h5 file, and if appropriate final
                           shape of dataset; list of tuples
            :*dtype*:      Data type of nascent dataset
            :*slc*:        Slice within dataset at which this block
                           will be stored
            :*dimensions*: Additional dimensions in dataset; if
                           multidimensional (optional)
            :*attrs*:      Attributes to add to dataset
        """
        super(Raw_Block, self).__init__(**kwargs)

        self.infiles     = infiles
        self.dimensions  = dimensions
        self.output      = output
        self.datasets    = {self.output: dict(slc = slc, attrs = attrs, 
                             kwargs = dict(maxshape = [None] + dimensions,
                             scaleoffset = 4))}

    def __call__(self, **kwargs):
        """
        Runs this block of analysis
        """
        from subprocess import Popen, PIPE

        # Load raw data into numpy using shell commands; there may be a faster
        #   way to do this; but this seems faster than np.loadtxt()
        #   followed by np.concanenate() for multiple files
        command     = "cat {0} | sed ':a;N;$!ba;s/\\n//g'".format(
                        " ".join(self.infiles))
        process     = Popen(command, stdout = PIPE, shell = True)
        input_bytes = bytearray(process.stdout.read())
        dataset     = np.array(np.frombuffer(input_bytes, dtype = "S8",
                        count = int((len(input_bytes) -1) / 8)), np.float32)

        # np.loadtxt alternative; keep here for future testing
        # dataset = []
        # for infile in self.infiles:
        #     dataset += [np.loadtxt(infile)]
        # dataset = np.concatenate(dataset)

        # Reshape if necessary
        if len(self.dimensions) != 0:
            dataset = dataset.reshape(
              [dataset.size / np.product(self.dimensions)] + self.dimensions)

        # Store in instance variable
        self.datasets[self.output]["data"] = dataset


