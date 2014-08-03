#!/usr/bin/python
#   MDclt.secondary.Pmf.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-08-02
"""
Classes for calculation of potential of mean force

.. todo:
    - Support bin expressions in the same variety of forms that WESTPA does
    - Support additional dimensions (change -coord from accepting 2 arguments
      to 2 * N arguments, same for -bins)
    - Support ignoring portion of dataset
    - Calculate and print minimum, POI, desolvation barrier, etc.
"""
################################### MODULES ####################################
import os, sys
import numpy as np
from MDclt import secondary
from MDclt.secondary import Block_Generator
from MDclt import Block, Block_Accumulator, Block_Acceptor
from MDclt import pool_director
################################## FUNCTIONS ###################################
def add_parser(tool_subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: <argparse._SubParsersAction> to which to add
        :*\*args*:     Passed to *subparsers*.add_parser(...)
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    tool_subparser = tool_subparsers.add_parser(
      name     = "pmf",
      help     = "Calculates potential of mean force")
    pdf_subparsers = tool_subparser.add_subparsers(
      dest = "pdf",
      description = "")

    hist_subparser = secondary.add_parser(pdf_subparsers,
      name = "hist",
      help = "Estimates probability density function using a histogram")
    
    kde_subparser  = secondary.add_parser(pdf_subparsers,
      name = "kde",
      help = "Estimates probability density function using a kernal density estimate")

    arg_groups = {
        hist_subparser: {ag.title: ag for ag in hist_subparser._action_groups},
        kde_subparser:  {ag.title: ag for ag in kde_subparser._action_groups}}

    for pdf_subparser in [hist_subparser, kde_subparser]:
        arg_groups[pdf_subparser]["input"].add_argument(
          "-coord",
          type     = str,
          required = True,
          nargs    = 2,
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and address from which to load coordinate")

    arg_groups[hist_subparser]["action"].add_argument(
      "-bins",
      type     = str,
      required = True,
      metavar  = "BINEXPR",
      help     = "Python expression used to generate bins")
    arg_groups[kde_subparser]["action"].add_argument(
      "-grid",
      type     = str,
      required = True,
      metavar  = "GRIDEXPR",
      help     = "Python expression used to generate grid")
    arg_groups[kde_subparser]["action"].add_argument(
      "-bandwidth",
      type     = float,
      required = True,
      help     = "Bandwidth of kernel density estimate")

    for pdf_subparser in [hist_subparser, kde_subparser]:
        arg_groups[pdf_subparser]["action"].add_argument(
          "-zero_point",
          type     = str,
          required = False,
          help     = "Point at which to adjust PMF to 0; " + \
                     "alternatively range of points (e.g. 10-12) " + \
                     "at which to adjust average to 0 (optional)")
        arg_groups[pdf_subparser]["action"].add_argument(
          "-temperature",
          type     = str,
          default  = 298.0,
          help     = "System temperature (default %(default)s)")

    for pdf_subparser in [hist_subparser, kde_subparser]:
        arg_groups[pdf_subparser]["output"].add_argument(
          "-output",
          type     = str,
          required = True,
          nargs    = 1,
          metavar  = "H5_FILE",
          help     = "H5 file in which to output data")

    for pdf_subparser in [hist_subparser, kde_subparser]:
        pdf_subparser.set_defaults(analysis = command_line)

def command_line(n_cores = 1, **kwargs):
    """
    Provides command line functionality for this analysis

    **Arguments:**
        :*n_cores*: Number of cores to use
    """
    from multiprocessing import Pool

    block_generator   = Block_Generator(**kwargs)
    block_accumulator = Block_Accumulator(
                          preexisting_slice = block_generator.preexisting_slice,
                          incoming_slice    = block_generator.incoming_slice,
                          **kwargs)
    block_accumulator.next()

    # Serial
    # for block in block_generator:
    #     block()
    #     block_accumulator.send(block)

    # Parallel (processes)
    pool = Pool(n_cores)
    for block in pool.imap_unordered(pool_director, block_generator):
        block_accumulator.send(block)
    pool.close()
    pool.join()

    block_accumulator.close()

    block_acceptor = Block_Acceptor(**kwargs)
    block_acceptor.next()
    block_acceptor.send(block_accumulator)
    block_acceptor.close()

################################### CLASSES ####################################
class Block(Block):
    """
    Independent block of analysis
    """
    def __init__(self, coord, bins, out_address, slc, attrs = {}, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:       Coordinates
            :*bins*:        Bins in which to classify coordinates
            :*out_address*: Address of dataset within h5 file
            :*slc*:         Indexes of simulation frames to which this block
                            corresponds
        """
        from collections import OrderedDict

        # Input
        self.coord       = coord

        # Action
        self.bins        = bins

        # Output
        self.out_address = out_address
        self.datasets    = OrderedDict(
          {out_address: dict(slc = slc, attrs = attrs)})

    def __call__(self, **kwargs):
        """
        Runs this block of analysis
        """
        hist, _ = np.histogram(self.coord, self.bins)
        self.datasets[self.out_address]["count"] = hist

class Block_Generator(Block_Generator):
    """
    Generator class that yields blocks of analysis
    """
    def __init__(self, log, coord, output, bins, *args, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: List including path to h5 file and address within h5 file
            :*bins*:   Bins in which to classify coordinates
            :*force*:  Run analysis even if no new data is present
        """

        # Input
        self.log_path,   self.log_address   = log
        self.coord_path, self.coord_address = coord
        self.frames_per_block               = 10000

        # Action
        self.bins = np.squeeze(np.array(eval(bins)))

        # Output
        self.out_path, self.out_address = output

        super(Block_Generator, self).__init__(inputs = [log, coord],
          output = output, *args, **kwargs)

    def next(self):
        """
        Prepares and yields next Block of analysis
        """
        from h5py import File as h5

        if self.start_index >= self.final_index:
            raise StopIteration()
        else:
            # Determine slice indexes
            block_slice = slice(self.start_index,
                                min(self.start_index + self.frames_per_block,
                                    self.final_index), 1)

            # Load primary data from these indexes
            with h5(self.coord_path) as coord_h5:
                block_coord = np.array(
                                coord_h5[self.coord_address][block_slice])

            # Iterate
            self.start_index += self.frames_per_block

            # Return new block
            return Block(coord       = block_coord,
                         bins        = self.bins,
                         out_address = self.out_address,
                         slc         = block_slice)

class Block_Accumulator(Block_Accumulator):
    """
    Coroutine class used to accumulate Blocks of data and perform
    analysis once the complete data is present; also may act as a Block
    itself
    """
    def __init__(self, bins, temperature, zero_point, output,
                 preexisting_slice, incoming_slice, attrs = {}, **kwargs):
        """
        Initializes wrapped function

        **Arguments:**
            :*bins*:        Bins in which to classify coordinates
            :*temperature*: Simulation temperature used to calculate pmf
            :*zero_point*:  Point to adjust pmf to zero, or range of
                            points over which to adjust average to zero
            :*output*: List including path to h5 file and address
                            within h5 file
        """
        from h5py import File as h5
        from collections import OrderedDict

        super(Block_Accumulator, self).__init__(**kwargs)
        out_path, out_address = output

        self.bins              = np.squeeze(np.array(eval(bins)))
        self.temperature       = temperature
        self.zero_point        = zero_point
        self.out_address       = out_address
        self.preexisting_slice = preexisting_slice
        self.incoming_slice    = incoming_slice
        self.received_slices   = []

        data = np.zeros(self.bins.size - 1,
                 dtype = [("lower bound", "f4"), ("center",      "f4"),
                          ("upper bound", "f4"), ("count",       "i4"),
                          ("probability", "f4"), ("free energy", "f4"),
                          ("pmf",         "f4")])
        data["lower bound"] =  self.bins[:-1]
        data["center"]      = (self.bins[:-1] + self.bins[1:]) / 2
        data["upper bound"] =  self.bins[1:]
        if self.preexisting_slice is not None:
            with h5(out_path) as out_h5:
                preexisting_data    = np.array(out_h5[out_address])
                data["lower bound"] = preexisting_data["lower bound"]
                data["center"]      = preexisting_data["center"]
                data["upper bound"] = preexisting_data["upper bound"]
                data["count"]      += preexisting_data["count"]
        else:
            data["lower bound"] =  self.bins[:-1]
            data["center"]      = (self.bins[:-1] + self.bins[1:]) / 2
            data["upper bound"] =  self.bins[1:]

        self.datasets = OrderedDict({self.out_address:
                          dict(data = data, attrs = attrs)})

    def receive_block(self, **kwargs):
        """
        Accumulates received Blocks
        """
        dataset = self.datasets[self.out_address]["data"]
        while True:
            block = (yield)
            dataset["count"] += block.datasets[self.out_address]["count"]
            self.receive_slice(block.datasets[self.out_address]["slc"])
            print(self.incoming_slice, self.received_slices)

    def close(self, *args, **kwargs):
        """
        Calculates free energy and potential of mean force
        """
        import types

        self.func.close(*args, **kwargs)
        dataset = self.datasets[self.out_address]["data"]
        attrs   = self.datasets[self.out_address]["attrs"]
        if self.incoming_slice is None:
            self.datasets = {}
            return
        if len(self.received_slices) != 1:
            raise Exception("A portion of the data appears to have been lost; "
                    + "Expected to receive {0}; ".format(self.incoming_slice)
                    + "But received {0}".format(self.received_slices))
        if self.preexisting_slice is None:
            attrs["slice"] = str(slice(self.received_slices[0].start,
                                   self.received_slices[0].stop, 1))
        else:
            attrs["slice"] = str(slice(self.preexisting_slice.start,
                                   self.received_slices[0].stop, 1))
            

        # Calculate free energy and PMF
        probability = np.array(dataset["count"],
                        dtype = np.float32) / np.sum(dataset["count"])
        probability[probability == 0.0] = np.nan
        free_energy = -1.0 * np.log(probability)
        pmf         = probability / (dataset["center"] ** 2.0)
        pmf        /= np.nansum(pmf)
        pmf         = (-1.0 * np.log(pmf) * 0.0019872041 * self.temperature)

        # Zero PMF at selected point or over selected range
        zero_point = self.zero_point
        if zero_point:
            if isinstance(zero_point, types.StringTypes):
                if   ":" in zero_point:
                    zero_start, zero_end = zero_point.split(":")
                elif "-" in zero_point:
                    zero_start, zero_end = zero_point.split("-")
                else:
                    zero_start, zero_end = zero_point.split()
                zero_start    = np.abs(self.bins - float(zero_start)).argmin()
                zero_end      = np.abs(self.bins - float(zero_end)).argmin()
                value_at_zero = np.mean(np.ma.MaskedArray(
                                  pmf[zero_start:zero_end],
                                  np.isnan(pmf[zero_start:zero_end])))
            else:
                value_at_zero = pmf[np.abs(center - zero_point).argmin()]
            pmf                -= value_at_zero
            attrs["zero point"] = zero_point

        # Organize and return data
        dataset["probability"]     = probability
        dataset["free energy"]     = free_energy
        dataset["pmf"]             = pmf
        attrs["lower bound units"] = "A"
        attrs["center units"]      = "A"
        attrs["upper bound units"] = "A" 
        attrs["free energy units"] = "kBT"
        attrs["pmf units"]         = "kcal mol-1"
        attrs["temperature"]       = self.temperature


