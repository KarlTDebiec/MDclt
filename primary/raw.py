#!/usr/bin/python
#   MDclt.primary.raw.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-10
"""
Classes for transfer of data from raw text files to h5

.. todo:
    - Better support appending data rather than reloading complete dataset
    - Look into improving speed (np.loadtxt or np.genfromtxt may actually be preferable)
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import primary
from MDclt import Block, Block_Acceptor
###################################################### FUNCTIONS #######################################################
def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: argparse subparsers object to add subparser
        :*args*:       Passed to subparsers.add_parser(...)
        :*kwargs*:     Passed to subparsers.add_parser(...)
    """
    subparser = primary.add_parser(subparsers, name = "raw",
      help = "Load raw text files")
    arg_groups = {ag.title: ag for ag in subparser._action_groups}

    arg_groups["input"].add_argument("-frames_per_file", type = int, required = False,
      help = "Number of frames in each file; used to check if new data is present (optional)")
    arg_groups["input"].add_argument("-dimensions", type = int, required = False, nargs = "*",
      help = "Additional dimensions in dataset; if multidimensional (optional)")

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

    block_generator = Block_Generator(**kwargs)
    block_acceptor  = Block_Acceptor(out_path = kwargs["output"][0], **kwargs)
    block_acceptor.next()
    pool            = Pool(n_cores)

    for block in pool.imap_unordered(pool_director, block_generator):
        block_acceptor.send(block)

    pool.close()
    pool.join()
    block_acceptor.close()

####################################################### CLASSES ########################################################
class Block(Block):
    """
    Independent block of analysis
    """
    def __init__(self, infiles, address, slc, dimensions = None, *args, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*infiles*:    List of infiles
            :*address*:    Address of dataset within h5 file
            :*slc*:        Slice within dataset at which this block will be stored
            :*dimensions*: Additional dimensions in dataset; if multidimensional (optional)
        """
        super(Block, self).__init__(*args, **kwargs)
        self.infiles     = infiles
        self.dimensions  = dimensions
        self.datasets    = [dict(address = address, slc = slc, attrs = {})]

    def __call__(self, **kwargs):
        """
        Runs this block of analysis
        """
        import subprocess

        # Load raw data into numpy using shell commands
        # There may be a faster way to do this; but this is at least faster than np.loadtxt(...) (maybe)
        command     = "cat {0} | sed ':a;N;$!ba;s/\\n//g'".format(" ".join(self.infiles))
        process     = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
        input_bytes = bytearray(process.stdout.read())
        dataset     = np.array(np.frombuffer(input_bytes, dtype = "S8",
                        count = int((len(input_bytes) -1) / 8)), np.float32)

        # np.loadtxt alternative; keep here for future testing
        # dataset = []
        # for infile in self.infiles:
        #     dataset += [np.loadtxt(infile)]
        # dataset = np.concatenate(dataset)

        # Reshape if necessary
        if self.dimensions is not None:
            dataset = dataset.reshape([dataset.size / np.product(self.dimensions)] + self.dimensions)

        # Store in instance variable
        self.datasets[0]["data"] = dataset

class Block_Generator(primary.Block_Generator):
    """
    Generator class that yields blocks of analysis
    """

    def __init__(self, output, infiles, frames_per_file = None, dimensions = [], **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*output*:          List including path to h5 file and address within h5 file
            :*infiles*:         List of infiles
            :*frames_per_file*: Number of frames in each infile (optional)
            :*dimensions*:      Additional dimensions in dataset; if multidimensional (optional)

        .. todo:
            - Intelligently break lists of infiles into blocks larger than 1
        """
        out_path, out_address = output
        self.out_address      = out_address
        self.infiles          = infiles
        self.frames_per_file  = frames_per_file
        self.dimensions       = dimensions
        self.expected_shape   = [len(self.infiles) * self.frames_per_file] + self.dimensions
        self.current_index    = 0
        self.dtype            = np.float32

        dataset_kwargs = dict(chunks = True, compression = "gzip", maxshape = [None] + self.dimensions)

        super(Block_Generator, self).__init__(output, dataset_kwargs = dataset_kwargs, **kwargs)

    def next(self):
        """
        Prepares and yields next Block of analysis
        """
        if len(self.infiles) == 0:
            raise StopIteration()
        else:
            slice_start         = self.current_index
            slice_end           = self.current_index + self.frames_per_file
            self.current_index += self.frames_per_file
            return Block(infiles    = [self.infiles.pop(0)],
                       address    = self.out_address,
                       slc        = slice(slice_start, slice_end, 1),
                       dimensions = self.dimensions)


