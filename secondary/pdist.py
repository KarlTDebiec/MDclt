# -*- coding: utf-8 -*-
#   MDclt.secondary.pdist.py
#
#   Copyright (C) 2012-2015 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
"""
Classes for calculation of probability density function

.. todo:
    - Support bin expressions in the same variety of forms that WESTPA
      does
    - Support additional dimensions (change -coord from accepting 2
      arguments to 2 * N arguments, same for -bins)
    - Support ignoring portion of dataset
    - Calculate and print minimum, POI, desolvation barrier, etc.
      (most likely in a separate program)
    - Select KDE bandwidth automatically
"""
################################### MODULES ####################################
import os, sys
import numpy as np
from MDclt import pool_director, secondary
from MDclt import Block, Block_Accumulator, Block_Acceptor
################################## FUNCTIONS ###################################
def add_parser(tool_subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*tool_subparsers*: <argparse._SubParsersAction> to which to add
        :*\*\*kwargs*:      Passed to *subparsers*.add_parser(...)
    """
    from MDclt import h5_default_path, overridable_defaults

    tool_subparser = tool_subparsers.add_parser(
      name     = "pdist",
      help     = "Calculates probability density function along a coordinate")
    mode_subparsers = tool_subparser.add_subparsers(
      dest        = "mode",
      description = "")

    hist_subparser = secondary.add_parser(mode_subparsers,
      name = "hist",
      help = "Calculates probability density function along a coordinate " +
             "using a histogram")
    kde_subparser  = secondary.add_parser(mode_subparsers,
      name = "kde",
      help = "Calculates probability density function along a coordinate " +
             "using a kernal density estimate")

    arg_groups = {
        hist_subparser:
          {ag.title: ag for ag in hist_subparser._action_groups},
        kde_subparser:
          {ag.title: ag for ag in kde_subparser._action_groups}}

    # Input
    for mode_subparser in [hist_subparser, kde_subparser]:
        arg_groups[mode_subparser]["input"].add_argument(
          "-coord",
          type     = str,
          required = True,
          nargs    = "+",
          action   = h5_default_path(),
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and address from which to load coordinate")

    # Action
    arg_groups[hist_subparser]["action"].add_argument(
      "-frames_per_block",
      type     = int,
      default  = 128,
      help     = "Number of frames included in each block of analysis; " +
                 "may influence results of block averaging error analysis,  " +
                 "frames_per_block is minimum potential length of block " +
                 "(default: %(default)s)")
    arg_groups[kde_subparser]["action"].add_argument(
      "-frames_per_block",
      type     = int,
      default  = 128,
      help     = "Number of frames included in each block of analysis; " + 
                 "may influence results of block averaging error analysis, " +
                 "frames_per_block is minimum potential length of block " +
                 "(default: %(default)s)")
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
    for mode_subparser in [hist_subparser, kde_subparser]:
        arg_groups[mode_subparser]["action"].add_argument(
          "-zero_point",
          type     = str,
          required = False,
          help     = "Point at which to shift PMF to 0; " + 
                     "alternatively range of points (e.g. 10-12) " + 
                     "over which to adjust average to 0 (optional)")
        arg_groups[mode_subparser]["action"].add_argument(
          "-temperature",
          type     = str,
          default  = 298.0,
          help     = "System temperature (default: %(default)s)")

    # Output
    for mode_subparser in [hist_subparser, kde_subparser]:
        arg_groups[mode_subparser]["output"].add_argument(
          "-output",
          type     = str,
          required = True,
          nargs    = "+",
          action   = overridable_defaults(nargs = 2, defaults = {1: "/pdist"}),
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and optionally address in which to output data " +
                     "(default ADDRESS: /pdist)")

    hist_subparser.set_defaults(
      analysis = command_line(Hist_Block_Generator, Hist_Block_Accumulator))
    kde_subparser.set_defaults(
      analysis = command_line(KDE_Block_Generator,  KDE_Block_Accumulator))

def command_line(block_generator_class, block_accumulator_class, **kwargs):
    """
    Generates function for command line action

    **Arguments:**
        :*block_generator_class*:   Class to be used for block
                                    generation
        :*block_accumulator_class*: Class to be used for block
                                    accumulation
    """

    def func(n_cores = 1, **kwargs):
        """
        Function for command line action

        **Arguments:**
            :*n_cores*: Number of cores to use
        """
        from multiprocessing import Pool

        block_generator   = block_generator_class(**kwargs)
        block_accumulator = block_accumulator_class(
          preexisting_slice = block_generator.preexisting_slice,
          incoming_slice    = block_generator.incoming_slice,
          outputs           = block_generator.outputs,
          **kwargs)

        if n_cores == 1:                # Serial
            for block in block_generator:
                block()
                block_accumulator.send(block)
        else:                           # Parallel (processes)
            pool = Pool(n_cores)
            for block in pool.imap_unordered(pool_director, block_generator):
                pass
                block_accumulator.send(block)
            pool.close()
            pool.join()

        block_accumulator.close()

        block_acceptor = Block_Acceptor(outputs = block_accumulator.outputs,
                           **kwargs)
        block_acceptor.send(block_accumulator)
        block_acceptor.close()
    return func

################################### CLASSES ####################################
class PDist_Block_Generator(secondary.Secondary_Block_Generator):
    """
    Generator class that prepares blocks of analysis
    """
    def __init__(self, coord, frames_per_block, output, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: Tuple including path to h5 file and address
                       within h5 file
            :*force*:  Run analysis even if no new data is present
        """
        from MDclt import clean_path

        # Input
        self.coord_path, self.coord_address = coord
        self.inputs = [(self.coord_path, self.coord_address)]

        # Action
        self.frames_per_block = frames_per_block

        # Output
        output[1]    = clean_path(output[1], strip = ["pdist", "blocks"])
        self.outputs = [(output[0], os.path.normpath(output[1] + "/pdist")),
                        (output[0], os.path.normpath(output[1] + "/blocks"))]

        super(PDist_Block_Generator, self).__init__(**kwargs)

        if self.incoming_slice is not None:
            self.current_start = self.incoming_slice.start
            self.current_stop  = self.incoming_slice.start + \
                                 self.frames_per_block
            self.final_stop    = self.final_slice.stop

    def get_incoming_slice(self, **kwargs):
        """
        """
        if self.preexisting_slice is None:
            self.incoming_slice = self.final_slice
        elif self.final_slice.stop > self.preexisting_slice.stop:
            if (self.final_slice.stop - self.preexisting_slice.stop <
                self.frames_per_block):
                self.incoming_slice = None
            else:
                self.incoming_slice = slice(self.preexisting_slice.stop,
                  self.final_slice.stop, 1)
        elif self.final_slice.stop < self.preexisting_slice.stop:
            warning = "Preexisting outfile(s) appear to correspond to " + \
              "a larger slice of frames " + \
              "({0}) ".format(self.preexisting_slice) + \
              "than available in infile(s) " + \
              "({0})".format(self.final_slice)
            self.incoming_slice = None
        else:
            self.incoming_slice = None

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
            block_slice = slice(self.current_start, self.current_stop, 1)

            # Load primary data from these indexes
            #   NOTE: It is necessary to round to the scaleoffset in
            #   order to ensure that the same results are obtained for
            #   fresh and extended datasets
            with h5(self.coord_path) as coord_h5:
                scaleoffset = coord_h5[self.coord_address].scaleoffset
                block_coord = np.array(coord_h5[self.coord_address]
                                [block_slice])
                if scaleoffset is not None:
                    scaleoffset = int(str(scaleoffset)[0])
                    block_coord = np.round(block_coord, scaleoffset)

            # Iterate
            self.current_start += self.frames_per_block
            self.current_stop   = min(self.current_start+self.frames_per_block,
                                  self.final_stop)

            # Return new block
            return self.block_class(
              coord   = block_coord,
              outputs = self.outputs,
              slc     = block_slice,
              **self.block_kwargs)

    def get_preexisting_slice(self, force = False, **kwargs):
        """
        Determines slice to which preexisting data in outputs
        corresponds

        **Arguments**
            :*force*: Even if preexisting data is present, disregard
        """
        from warnings import warn
        from h5py import File as h5

        out_path, out_address = self.outputs[1]
        with h5(out_path) as out_h5:
            if out_address in out_h5:
                if force:
                    del out_h5[out_address]
                    self.preexisting_slice = None
                else:
                    blocks = np.array(out_h5[out_address])
                    attrs  = dict(out_h5[out_address].attrs)
                    if (blocks["stop"][-1] - blocks["start"][-1] !=
                      attrs["block_size"]):
                        self.preexisting_slice = slice(blocks["start"][0],
                          blocks["stop"][-2], 1)
                    else:
                        self.preexisting_slice = slice(blocks["start"][0],
                          blocks["stop"][-1], 1)
            else:
                self.preexisting_slice = None

class Hist_Block_Generator(PDist_Block_Generator):
    """
    Generator class; generates blocks of analysis for probability
    density function calculation using a histogram
    """
    def __init__(self, bins, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*bins*: Bins in which to calculate pdf
        """
        super(Hist_Block_Generator, self).__init__(**kwargs)

        # Action
        bins              = np.squeeze(np.array(eval(bins)))
        self.block_class  = Hist_Block
        self.block_kwargs = dict(bins = bins)

class KDE_Block_Generator(PDist_Block_Generator):
    """
    Generator class; yields blocks of analysis for potential of mean
    force calculation, using a kernel density estimate for the
    probability density function
    """
    def __init__(self, grid, bandwidth, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*grid*:      Grid on which to calculate pdf
            :*bandwidth*: Kernel bandwidth
        """
        super(KDE_Block_Generator, self).__init__(**kwargs)

        # Action
        grid              = np.squeeze(np.array(eval(grid)))
        self.block_class  = KDE_Block
        self.block_kwargs = dict(grid = grid, bandwidth = bandwidth)

class PDist_Block(Block):
    """
    Independent block of analysis for probability density function
    """
    def __init__(self, coord, outputs, slc, attrs = {}, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:   Coordinates
            :*outputs*: For each dataset, path to h5 file and address
                        within h5 file
            :*slc*:     Indexes of simulation frames to which this
                        block corresponds
            :*attrs*:   Attributes to add to dataset
        """
        # Input
        self.coord = coord

        # Output
        self.outputs  = outputs
        self.datasets = {self.outputs[0]: dict(slc = slc, attrs = attrs),
                         self.outputs[1]: dict(slc = slc)}

        super(PDist_Block, self).__init__(**kwargs)

class Hist_Block(PDist_Block):
    """
    Independent block of analysis for probability density function
    using a histogram
    """
    def __init__(self, bins, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*bins*: Bins in which to calculate pdf
        """
        # Action
        self.bins = bins

        super(Hist_Block, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs block of analysis
        """
        hist, _ = np.histogram(self.coord, self.bins)

        self.datasets[self.outputs[0]]["count"] = hist

class KDE_Block(PDist_Block):
    """
    Independent block of analysis for probability density function
    using a kernel density estimate
    """
    def __init__(self, grid, bandwidth, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*grid*:      Grid on which to calculate pdf
            :*bandwidth*: Kernel bandwidth
        """
        # Action
        self.grid      = grid
        self.bandwidth = bandwidth

        super(KDE_Block, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs block of analysis
        """
        from sklearn.neighbors import KernelDensity

        kde     = KernelDensity(bandwidth = self.bandwidth, **kwargs)
        kde.fit(self.coord.flatten()[:, np.newaxis])
        log_pdf = kde.score_samples(self.grid[:, np.newaxis])
        pdf     = np.exp(log_pdf)

        self.datasets[self.outputs[0]]["kde"] = pdf

class PDist_Block_Accumulator(Block_Accumulator):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a Block_Acceptor
    """
    def __init__(self, temperature, zero_point, **kwargs):
        """
        Initializes accumulator

        **Arguments:**
            :*temperature*: Simulation temperature used to calculate pmf
            :*zero_point*:  Point to shift pmf to zero, or range of
                            points over which to shift average to zero
        """
        # Action
        self.temperature = temperature
        self.zero_point  = zero_point

        super(PDist_Block_Accumulator, self).__init__(**kwargs)

    def close(self, **kwargs):
        """
        Calculates free energy and potential of mean force
        """
        import types

        self.func.close(**kwargs)
        if self.incoming_slice is None:
            self.datasets = {}
            return

        # Link to datasets for clarity
        pdist     = self.datasets[self.outputs[0]]["data"]
        blocks    = self.datasets[self.outputs[1]]["data"]
        pdist_at  = self.datasets[self.outputs[0]]["attrs"]
        blocks_at = self.datasets[self.outputs[1]]["attrs"]

        # Process slice information
        if len(self.received_slices) != 1:
            raise Exception("A portion of the data is missing; "
                    + "Expected to receive {0}; ".format(self.incoming_slice)
                    + "But received {0}".format(self.received_slices))
        if self.preexisting_slice is None:
            pdist_at["slice"] = str(slice(self.received_slices[0].start,
              self.received_slices[0].stop, 1))
        else:
            pdist_at["slice"] = str(slice(self.preexisting_slice.start,
              self.received_slices[0].stop, 1))
        blocks_at["slice"] = pdist_at["slice"]
        

        # Calculate free energy and PMF
        pdist["probability"] /= np.nansum(pdist["probability"])
        pdist["free energy"]  = -1.0 * np.log(pdist["probability"])
        pdist["pmf"]          = pdist["probability"] / (pdist["center"] ** 2.0)
        pdist["pmf"]         /= np.nansum(pdist["pmf"])
        pdist["pmf"]          = (-1.0 * np.log(pdist["pmf"]) * 0.0019872041
                                * self.temperature)

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
                zero_start    = np.abs(pdist["center"]
                                  - float(zero_start)).argmin()
                zero_end      = np.abs(pdist["center"]
                                  - float(zero_end)).argmin()
                value_at_zero = np.mean(np.ma.MaskedArray(
                  pdist["pmf"][zero_start:zero_end],
                  np.isnan(pdist["pmf"][zero_start:zero_end])))
            else:
                value_at_zero = pdist["pmf"][np.abs(x - zero_point).argmin()]
            pdist["pmf"]          -= value_at_zero
            pdist_at["zero point"] = zero_point

        # Organize and return data
        pdist_at["free energy units"] = "kBT"
        pdist_at["pmf units"]         = "kcal mol-1"
        pdist_at["temperature"]       = self.temperature

class Hist_Block_Accumulator(PDist_Block_Accumulator):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a
    Block_Acceptor
    """
    def __init__(self, preexisting_slice, incoming_slice, bins,
        frames_per_block, outputs, attrs = {}, **kwargs):
        """
        Initializes accumulator

        **Arguments:**
            :*preexisting_slice*: Slice containing frame indices whose
                                  results were included in *outputs*
                                  before this invocation of program
            :*incoming_slice*:    Slice containting frame indices whose
                                  results are to be added to *outputs*
                                  during this invocation of program
            :*bins*:              Bins on which to calculate pdf
            :*frames_per_block*:  Number of frames present in each
                                  incoming block
            :*outputs*:           Path to h5 file and address within h5
                                  file for each output dataset; list of
                                  tuples
            :*attrs*:             Attributes to add to dataset
        """
        from h5py import File as h5

        # Input
        self.preexisting_slice = preexisting_slice
        self.incoming_slice    = incoming_slice
        self.received_slices   = []

        # Action
        self.bins             = np.squeeze(np.array(eval(bins)))
        self.frames_per_block = frames_per_block

        # Output
        self.outputs = outputs

        # Prepare dataset
        pdist = np.zeros(self.bins.size - 1,
          dtype = [("lower bound", "f4"), ("center",      "f4"),
                   ("upper bound", "f4"), ("count",       "i4"),
                   ("probability", "f8"), ("free energy", "f8"),
                   ("pmf",         "f8")])
        pdist["lower bound"]       =  self.bins[:-1]
        pdist["center"]            = (self.bins[:-1] + self.bins[1:]) / 2
        pdist["upper bound"]       =  self.bins[1:]
        attrs["lower bound units"] = "A"
        attrs["center units"]      = "A"
        attrs["upper bound units"] = "A"

        out_path, out_address = self.outputs[1]
        with h5(out_path) as out_h5:
            if out_address in out_h5:
                blocks = np.array(out_h5[out_address])
                blocks = list(blocks[blocks["stop"]
                           <= self.preexisting_slice.stop])
                for block in blocks:
                    pdist["count"] += block["count"]
            else:
                blocks = []
        self.datasets = {
          self.outputs[0]: dict(data = pdist,  attrs = attrs),
          self.outputs[1]: dict(data = blocks, attrs = dict(
            block_size = self.frames_per_block))}

        super(Hist_Block_Accumulator, self).__init__(**kwargs)

    def receive_block(self, **kwargs):
        """
        Accumulates recieved Blocks
        """
        pdist  = self.datasets[self.outputs[0]]["data"]
        blocks = self.datasets[self.outputs[1]]["data"]

        while True:
            block       = (yield)
            block_count = block.datasets[self.outputs[0]]["count"]
            block_slice = block.datasets[self.outputs[0]]["slc"]
            pdist["count"]  += block_count
            blocks          += [(block_slice.start, block_slice.stop,
                                   block_count)]
            self.receive_slice(block_slice)

            print("Total incoming frames: [{0}:{1}]  ".format(
              self.incoming_slice.start, self.incoming_slice.stop) +
              "Received frames: {0}".format("".join(
              ["[{0}:{1}]".format(rs.start, rs.stop) for rs in 
              self.received_slices])))

    def close(self, **kwargs):
        """
        Estimates probability density function from histogram and
        passes on to superclass
        """
        # Link to dataset for clarity
        pdist  = self.datasets[self.outputs[0]]["data"]
        blocks = self.datasets[self.outputs[1]]["data"]

        # Calculate final probability and PMF
        if self.incoming_slice is None:
            self.datasets = {}
        else:
            pdist["probability"] = np.array(pdist["count"],
              dtype = np.float64) / np.sum(pdist["count"])
            pdist["probability"][pdist["probability"] == 0.0] = np.nan

            super(Hist_Block_Accumulator, self).close(**kwargs)

            blocks = np.array(blocks, dtype = 
              [("start",  np.int),
               ("stop",   np.int),
               ("count", (np.int, blocks[0][-1].shape))])

            self.datasets[self.outputs[1]]["data"] = blocks[
              np.argsort(blocks["start"])]

class KDE_Block_Accumulator(PDist_Block_Accumulator):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a Block_Acceptor
    """
    def __init__(self, preexisting_slice, incoming_slice, grid, bandwidth,
        frames_per_block, outputs, attrs = {}, **kwargs):
        """
        Initializes accumulator

        **Arguments:**
            :*preexisting_slice*: Slice containing frame indices whose
                                  results were included in *outputs*
                                  before this invocation of program
            :*incoming_slice*:    Slice containting frame indices whose
                                  results are to be added to *outputs*
                                  during this invocation of program
            :*grid*:              Grid on which to calculate pdf
            :*bandwidth*:         Kernel bandwidth
            :*frames_per_block*:  Number of frames present in each
                                  incoming block
            :*outputs*:           Path to h5 file and address within h5
                                  file for each output dataset; list of
                                  tuples
            :*attrs*:             Attributes to add to dataset
        """
        from h5py import File as h5

        # Input
        self.preexisting_slice = preexisting_slice
        self.incoming_slice    = incoming_slice
        self.received_slices   = []
        
        # Action
        self.grid = np.squeeze(np.array(eval(grid)))
        self.frames_per_block = frames_per_block

        # Output
        self.outputs = outputs

        # Prepare dataset
        pdist = np.zeros(self.grid.size,
          dtype = [("center",      "f4"), ("kde",         "f8"),
                   ("probability", "f8"), ("free energy", "f4"),
                   ("pmf",         "f4")])
        pdist["center"]       = self.grid
        attrs["center units"] = "A"
        attrs["bandwidth"]    = bandwidth

        out_path, out_address = self.outputs[1]
        with h5(out_path) as out_h5:
            if out_address in out_h5:
                blocks = np.array(out_h5[out_address])
                blocks = list(blocks[blocks["stop"]
                           <= self.preexisting_slice.stop])
                for block in blocks:
                    pdist["kde"] += block["kde"]
            else:
                blocks = []
        self.datasets = {
          self.outputs[0]: dict(data = pdist,  attrs = attrs),
          self.outputs[1]: dict(data = blocks, attrs = dict(
            block_size = self.frames_per_block))}

        super(KDE_Block_Accumulator, self).__init__(**kwargs)

    def receive_block(self, **kwargs):
        """
        Accumulates recieved Blocks
        """
        pdist  = self.datasets[self.outputs[0]]["data"]
        blocks = self.datasets[self.outputs[1]]["data"]

        while True:
            block       = (yield)
            block_kde   = block.datasets[self.outputs[0]]["kde"]
            block_slice = block.datasets[self.outputs[0]]["slc"]
            pdist["kde"]  += block_kde
            blocks        += [(block_slice.start, block_slice.stop,
                               block_kde)]
            self.receive_slice(block_slice)

            print("Total incoming frames: [{0}:{1}]  ".format(
              self.incoming_slice.start, self.incoming_slice.stop) +
              "Received frames: {0}".format("".join(
              ["[{0}:{1}]".format(rs.start, rs.stop) for rs in 
              self.received_slices])))

    def close(self, **kwargs):
        """
        Estimates probability density function from kernel density
        estimate and passes on to superclass
        """
        # Link to dataset for clarity
        pdist  = self.datasets[self.outputs[0]]["data"]
        blocks = self.datasets[self.outputs[1]]["data"]

        # Calculate final probability and PMF
        if self.incoming_slice is None:
            self.datasets = {}
        else:
            pdist["probability"] = pdist["kde"] / np.sum(pdist["kde"])
            pdist["probability"][pdist["probability"] == 0.0] = np.nan

            super(KDE_Block_Accumulator, self).close(**kwargs)

            blocks = np.array(blocks, dtype =
              [("start",  np.int),
               ("stop",   np.int),
               ("kde",   (np.float64, blocks[0][-1].shape))])
            self.datasets[self.outputs[1]]["data"] = blocks[
              np.argsort(blocks["start"])]


