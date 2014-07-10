#!/usr/bin/python
#   MDclt.secondary.Pmf.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-07-10
"""
Class for calculation of potential of mean force

.. todo:
    - Support bin expressions in the same variety of forms that WESTPA does
    - Support additional dimensions (change -coord from accepting 2 arguments to 2 * N arguments, same for -bins)
    - Support ignoring portion of dataset
"""
####################################################### MODULES ########################################################
import os, sys
import numpy as np
from MDclt import secondary
from MDclt import Block, Block_Generator, Block_Accumulator, Block_Acceptor, pool_director
###################################################### FUNCTIONS #######################################################
def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: <argparse._SubParsersAction> to which to add
        :*\*args*:     Passed to *subparsers*.add_parser(...)
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    subparser = secondary.add_parser(subparsers, name = "pmf",
      help = "Calculates potential of mean force")
    arg_groups = {ag.title:ag for ag in subparser._action_groups}

    arg_groups["input"].add_argument("-coord", type = str, required = True, nargs = 2,
      metavar = ("H5_FILE", "ADDRESS"),
      help = "H5 file and address from which to load coordinate")

    arg_groups["action"].add_argument("-bins", type = str, required = True,
      metavar = "BINEXPR",
      help = "Python expression used to generate bins")
    arg_groups["action"].add_argument("-zero_point", type = str, required = False,
      help = "Point at which to adjust PMF to 0; alternatively range of points (e.g. 10-12) at which to adjust " +
             "average to 0 (optional)")
    arg_groups["action"].add_argument("-temperature", type = str, default = 298.0,
      help = "System temperature (default %(default)s)")

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

    block_generator   = Block_Generator(**kwargs)
    block_accumulator = Block_Accumulator(**kwargs)
    block_accumulator.next()
    pool              = Pool(n_cores)

    for block in pool.imap_unordered(pool_director, block_generator):
        block_accumulator.send(block)

    pool.close()
    pool.join()

    block_accumulator.close()

    block_acceptor = Block_Acceptor(**kwargs)
    block_acceptor.next()
    block_acceptor.send(block_accumulator)
    block_acceptor.close()

####################################################### CLASSES ########################################################
class Block(Block):
    """
    Independent block of analysis
    """
    def __init__(self, coord, bins, out_address, slc, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:       Coordinates
            :*bins*:        Bins in which to classify coordinates
            :*out_address*: Address of dataset within h5 file
            :*slc*:         Indexes of simulation frames to which this block corresponds
        """
        self.coord      = coord
        self.bins       = bins
        self.datasets   = [dict(address = out_address, slc = slc, attrs = {})]

    def __call__(self, **kwargs):
        """
        Runs this block of analysis
        """
        hist, _ = np.histogram(self.coord, self.bins)
        self.datasets[0]["data"] = hist

class Block_Generator(Block_Generator):
    """
    Generator class that yields blocks of analysis
    """
    def __init__(self, log, coord, output, bins, force = False, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: List including path to h5 file and address within h5 file
            :*log*:    List of infiles
            :*force*:  Run analysis even if no new data is present
        """
        import warnings
        from h5py import File as h5

        # Unpack arguments
        log_path,   log_address   = log
        coord_path, coord_address = coord
        out_path,   out_address   = output

        # Check input and output files
        with h5(out_path) as out_h5, h5(log_path) as log_h5, h5(coord_path) as coord_h5:
            # Get previously-analyzed indexes from out_h5, use to set current_index unless force
            log_shape   = log_h5[log_address].shape
            coord_shape = coord_h5[coord_address].shape
            if log_shape[0] != coord_shape[0]:
                warning  = "Length of log dataset ({0}) ".format(log_shape[0])
                warning += "and coordinate dataset ({0}) ".format(coord_shape[0])
                warning += "do not match, using smaller of the two"
                warnings.warn(warning)
            self.final_index   = min(log_shape[0], coord_shape[0])
            if force:
                self.current_index = 0
            else:
                self.current_index = 0

        # Store necessary data in instance variables
        self.coord_path       = coord_path
        self.coord_address    = coord_address
        self.frames_per_block = 10000
        self.bins             = np.array(eval(bins))
        self.out_address      = out_address

    def next(self):
        """
        Prepares and yields next Block of analysis
        """
        from h5py import File as h5

        if self.current_index >= self.final_index:
            raise StopIteration()
        else:
            # Determine slice indexes
            slice_start = self.current_index
            slice_end   = min(self.current_index + self.frames_per_block, self.final_index)

            # Load primary data from these indexes (Consider keeping coord_h5 open from __init__ on...)
            with h5(self.coord_path) as coord_h5:
                coord = np.array(coord_h5[self.coord_address][slice_start:slice_end])

            # Iterate
            self.current_index += self.frames_per_block

            # Return new block
            return Block(
                     coord   = coord,
                     bins    = self.bins,
                     out_address = self.out_address,
                     slc     = slice(slice_start, slice_end, 1))

class Block_Accumulator(Block_Accumulator):
    """
    Coroutine class used to accumulate Blocks of data and perform analysis once the complete data is present;
    Also may act as a Block itself
    """
    def __init__(self, bins, temperature, zero_point, output, **kwargs):
        """
        Initializes wrapped function

        **Arguments:**
            :*bins*:        Bins in which to classify coordinates
            :*temperature*: Simulation temperature used to calculate pmf
            :*zero_point*:  Point to adjust pmf to zero, or range of points over which to adjust average to zero
            :*output*: List including path to h5 file and address within h5 file
        """
        self.func        = self.accumulate()
        self.bins        = np.array(eval(bins))
        self.temperature = temperature
        self.zero_point  = zero_point

        data = np.zeros(self.bins.size - 1,
                 dtype = [("lower bound", "f4"), ("center",      "f4"),
                          ("upper bound", "f4"), ("count",       "i4"),
                          ("probability", "f4"), ("free energy", "f4"),
                          ("pmf",         "f4")])
        data["lower bound"] =  self.bins[:-1]
        data["center"]      = (self.bins[:-1] + self.bins[1:]) / 2
        data["upper bound"] =  self.bins[1:]

        out_path, out_address = output
        self.datasets = [dict(address = out_address, attrs = {}, data = data)]

    def accumulate(self, **kwargs):
        """
        Accumulates received Blocks
        """
        while True:
            block = (yield)
            print(block.datasets[0]["slc"])
            self.datasets[0]["data"]["count"] += block.datasets[0]["data"]

    def close(self, *args, **kwargs):
        """
        Calculates free energy and potential of mean force
        """
        self.func.close(*args, **kwargs)

        # Calculate free energy and PMF
        center                          = self.datasets[0]["data"]["center"]
        count                           = self.datasets[0]["data"]["count"]
        probability                     = np.array(count, dtype = np.float32) / np.sum(count)
        probability[probability == 0.0] = np.nan
        free_energy                     = -1.0 * np.log(probability)
        pmf                             = probability / (center ** 2.0)
        pmf                            /= np.nansum(pmf)
        pmf                             = -1.0 * np.log(pmf) * 0.0019872041 * self.temperature

        # Zero PMF at selected point or over selected range
        zero_point = self.zero_point
        if zero_point:
            if isinstance(zero_point, types.StringTypes):
                if   ":" in zero_point: zero_start, zero_end = zero_point.split(":")
                elif "-" in zero_point: zero_start, zero_end = zero_point.split("-")
                else:                   zero_start, zero_end = zero_point.split()
                zero_start    = np.abs(bins - float(zero_start)).argmin()
                zero_end      = np.abs(bins - float(zero_end)).argmin()
                value_at_zero = np.mean(np.ma.MaskedArray(pmf[zero_start:zero_end], np.isnan(pmf[zero_start:zero_end])))
            else:
                value_at_zero = pmf[np.abs(center - zero_point).argmin()]
            pmf              -= value_at_zero
        
        else:
            zero_point        = "None"

        self.datasets[0]["data"]["probability"] = probability
        self.datasets[0]["data"]["free energy"] = free_energy
        self.datasets[0]["data"]["pmf"]         = pmf

        # Organize and return data
        self.datasets[0]["attrs"]["lower bound units"] = "A",
        self.datasets[0]["attrs"]["center units"]      = "A",
        self.datasets[0]["attrs"]["upper bound units"] = "A", 
        self.datasets[0]["attrs"]["free energy units"] = "kBT",
        self.datasets[0]["attrs"]["pmf units"]         = "kcal mol-1",
        self.datasets[0]["attrs"]["temperature"]       =  self.temperature


