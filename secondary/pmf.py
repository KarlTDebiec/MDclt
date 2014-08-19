#!/usr/bin/python
#   MDclt.secondary.Pmf.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-08-12
"""
Classes for calculation of potential of mean force

.. todo:
    - Support bin expressions in the same variety of forms that WESTPA does
    - Support additional dimensions (change -coord from accepting 2 arguments
      to 2 * N arguments, same for -bins)
    - Support ignoring portion of dataset
    - Calculate and print minimum, POI, desolvation barrier, etc.
    - Clear out *args; never likely to use
    - If h5 infile contains single address, just use that automatically
    - Select KDE bandwidth
    - Set frames_per_block manually
    - Add serial flag to all secondary analyses
"""
################################### MODULES ####################################
import os, sys
import numpy as np
from MDclt import secondary
from MDclt.secondary import Secondary_Block_Generator
from MDclt import Block, Block_Accumulator, Block_Acceptor
from MDclt import pool_director
################################## FUNCTIONS ###################################
def add_parser(tool_subparsers, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: <argparse._SubParsersAction> to which to add
        :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
    """
    from MDclt import overridable_defaults

    tool_subparser = tool_subparsers.add_parser(
      name     = "pmf",
      help     = "Calculates potential of mean force along a coordinate")
    pdf_subparsers = tool_subparser.add_subparsers(
      dest        = "pdf",
      description = "")

    hist_subparser = secondary.add_parser(pdf_subparsers,
      name = "hist",
      help = "Calculates potential of mean force along a coordinate using a " +
             "histogram estimate for the probability density function")
    
    kde_subparser  = secondary.add_parser(pdf_subparsers,
      name = "kde",
      help = "Calculates potential of mean force along a coordinate using a " +
             "kernal density estimate for the probability density function")

    arg_groups = {
        hist_subparser: {ag.title: ag for ag in hist_subparser._action_groups},
        kde_subparser:  {ag.title: ag for ag in kde_subparser._action_groups}}

    # Input
    for pdf_subparser in [hist_subparser, kde_subparser]:
        arg_groups[pdf_subparser]["input"].add_argument(
          "-coord",
          type     = str,
          required = True,
          nargs    = 2,
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and address from which to load coordinate")

    # Action
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
          help     = "Point at which to shift PMF to 0; " + \
                     "alternatively range of points (e.g. 10-12) " + \
                     "over which to adjust average to 0 (optional)")
        arg_groups[pdf_subparser]["action"].add_argument(
          "-temperature",
          type     = str,
          default  = 298.0,
          help     = "System temperature (default: %(default)s)")

    # Output
    for pdf_subparser in [hist_subparser, kde_subparser]:
        arg_groups[pdf_subparser]["output"].add_argument(
          "-output",
          type     = str,
          required = True,
          nargs    = "+",
          action   = overridable_defaults(nargs = 2, defaults = {1: "pmf"}),
          metavar  = ("H5_FILE", "ADDRESS"),
          help     = "H5 file and optionally address in which to output data " +
                     "(default ADDRESS: /)")

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
            :*n_cores*:                 Number of cores to use
        """
        from multiprocessing import Pool

        block_generator   = block_generator_class(**kwargs)
        block_accumulator = block_accumulator_class(
          preexisting_slice = block_generator.preexisting_slice,
          incoming_slice    = block_generator.incoming_slice,
                              **kwargs)
        block_accumulator.next()

        if n_cores == 1:                # Serial
            for block in block_generator:
                block()
                block_accumulator.send(block)
        else:                           # Parallel (processes)
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
    return func

################################### CLASSES ####################################
class PMF_Block(Block):
    """
    Independent block of analysis for potential of mean force
    """
    def __init__(self, coord, out_address, slc, attrs = {}, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:       Coordinates
            :*bins*:        Bins in which to classify coordinates
            :*out_address*: Address of dataset within h5 file
            :*slc*:         Indexes of simulation frames to which this block
                            corresponds
            :*attrs*:       Attributes to add to dataset
        """
        from collections import OrderedDict

        # Input
        self.coord = coord

        # Output
        self.out_address = out_address
        self.datasets    = OrderedDict(
          {out_address: dict(slc = slc, attrs = attrs)})

        super(PMF_Block, self).__init__(**kwargs)

class Hist_Block(PMF_Block):
    """
    Independent block of analysis for potential of mean force
    calculation, using a histogram estimate for the probability
    density function
    """
    def __init__(self, bins, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:       Coordinates
            :*bins*:        Bins in which to classify coordinates
            :*out_address*: Address of dataset within h5 file
            :*slc*:         Indexes of simulation frames to which this block
                            corresponds
            :*attrs*:       Attributes to add to dataset
        """
        # Action
        self.bins = bins

        super(Hist_Block, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs block of analysis
        """
        hist, _ = np.histogram(self.coord, self.bins)

        self.datasets[self.out_address]["count"] = hist

class KDE_Block(PMF_Block):
    """
    Independent block of analysis for potential of mean force
    calculation, using a kernel density estimate for the probability
    density function
    """
    def __init__(self, grid, bandwidth, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*coord*:       Coordinates
            :*grid*:        Grid on which to calculate pdf
            :*bandwidth*:   Kernel bandwidth
            :*out_address*: Address of dataset within h5 file
            :*slc*:         Indexes of simulation frames to which this block
                            corresponds
            :*attrs*:       Attributes to add to dataset
        """
        # Action
        self.grid      = grid
        self.bandwidth = bandwidth

        super(KDE_Block, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        Runs block of analysis
        """
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(
          self.coord.flatten(),
          bw_method = self.bandwidth / self.coord.flatten().std(ddof = 1))
        pdf = kde.evaluate(self.grid)
        pdf /= np.nansum(pdf)

        self.datasets[self.out_address]["pdf"] = pdf

class PMF_Block_Generator(Secondary_Block_Generator):
    """
    Generator class; yields blocks of analysis for potential of mean
    force calculation
    """
    def __init__(self, log, coord, output, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: List including path to h5 file and address
                       within h5 file
            :*force*:  Run analysis even if no new data is present
        """
        # Input
        self.log_path,   self.log_address   = log
        self.coord_path, self.coord_address = coord

        # Output
        self.out_path    = output[0]
        self.out_address = os.path.normpath(output[1] + "/pmf")

        super(PMF_Block_Generator, self).__init__(inputs = [log, coord],
          output = [self.out_path, self.out_address], **kwargs)

    def next(self):
        """
        Prepares and yields next Block of analysis
        """
        from h5py import File as h5

        if self.incoming_slice is None or self.start_index >= self.final_index:
            raise StopIteration()
        else:
            # Determine slice indexes
            block_slice = slice(self.start_index,
              min(self.start_index + self.frames_per_block, self.final_index),
              1)

            # Load primary data from these indexes
            with h5(self.coord_path) as coord_h5:
                block_coord = np.array(
                  coord_h5[self.coord_address][block_slice])

            # Iterate
            self.start_index += self.frames_per_block

            # Return new block
            return self.block_class(
              coord       = block_coord,
              out_address = self.out_address,
              slc         = block_slice,
              **self.block_kwargs)

class Hist_Block_Generator(PMF_Block_Generator):
    """
    Generator class; yields blocks of analysis for potential of mean
    force calculation, using a histogram estimate for the probability
    density function
    """
    def __init__(self, bins, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: List including path to h5 file and address
                       within h5 file
            :*bins*:   Bins in which to classify coordinates
            :*force*:  Run analysis even if no new data is present
        """
        # Action
        self.block_class      = Hist_Block
        self.block_kwargs     = dict(bins = np.squeeze(np.array(eval(bins))))
        self.frames_per_block = 10000

        super(Hist_Block_Generator, self).__init__(**kwargs)

class KDE_Block_Generator(PMF_Block_Generator):
    """
    Generator class; yields blocks of analysis for potential of mean
    force calculation, using a kernel density estimate for the
    probability density function
    """
    def __init__(self, grid, bandwidth, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:       Simulation log
            :*coord*:     Coordinates used to generate pmf
            :*grid*:      Grid on which to calculate pdf
            :*bandwidth*: Kernel bandwidth
            :*output*:    List including path to h5 file and address
                          within h5 file
            :*force*:     Run analysis even if no new data is present
        """
        # Action
        self.block_class      = KDE_Block
        self.block_kwargs     = dict(
          grid      = np.squeeze(np.array(eval(grid))),
          bandwidth = bandwidth)
        self.frames_per_block = 1000

        super(KDE_Block_Generator, self).__init__(**kwargs)

class PMF_Block_Accumulator(Block_Accumulator):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a Block_Acceptor
    """
    def __init__(self, preexisting_slice, incoming_slice, temperature,
                 zero_point, output, **kwargs):
        """
        Initializes accumulator

        **Arguments:**
            :*preexisting_slice*: slice containing frame indices whose results
                                  were included in *output* h5 file before
                                  this invocation of program
            :*incoming_slice*:    slice containting frame indices whose results
                                  are expected to be added to *output* h5 file
                                  after this invocation of program
            :*temperature*:       Simulation temperature used to calculate pmf
            :*zero_point*:        Point to shift pmf to zero, or range of
                                  points over which to shift average to zero
            :*output*:            List including path to h5 file and address
                                  within h5 file
        """
        from h5py import File as h5
        from collections import OrderedDict

        # Input
        self.preexisting_slice = preexisting_slice
        self.incoming_slice    = incoming_slice
        self.received_slices   = []

        # Action
        self.temperature = temperature
        self.zero_point  = zero_point

        # Output
        self.out_path    = output[0]
        self.out_address = os.path.normpath(output[1] + "/pmf")

        super(PMF_Block_Accumulator, self).__init__(**kwargs)

    def close(self, x, **kwargs):
        """
        Calculates free energy and potential of mean force
        """
        import types

        self.func.close(**kwargs)
        if self.incoming_slice is None:
            self.datasets = {}
            return

        # Link to datasets for clarity
        ds = self.datasets[self.out_address]["data"]
        at = self.datasets[self.out_address]["attrs"]

        # Process slice information
        if len(self.received_slices) != 1:
            raise Exception("A portion of the data is missing; "
                    + "Expected to receive {0}; ".format(self.incoming_slice)
                    + "But received {0}".format(self.received_slices))
        if self.preexisting_slice is None:
            at["slice"] = str(slice(self.received_slices[0].start,
              self.received_slices[0].stop, 1))
        else:
            at["slice"] = str(slice(self.preexisting_slice.start,
              self.received_slices[0].stop, 1))

        # Calculate free energy and PMF
        ds["probability"] = ds["probability"] / np.nansum(ds["probability"])
        ds["free energy"] = -1.0 * np.log(ds["probability"])
        ds["pmf"]         = ds["probability"] / (x ** 2.0)
        ds["pmf"]        /= np.nansum(ds["pmf"])
        ds["pmf"]         = (-1.0 * np.log(ds["pmf"]) * 0.0019872041
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
                zero_start    = np.abs(x - float(zero_start)).argmin()
                zero_end      = np.abs(x - float(zero_end)).argmin()
                value_at_zero = np.mean(np.ma.MaskedArray(
                  ds["pmf"][zero_start:zero_end],
                  np.isnan(ds["pmf"][zero_start:zero_end])))
            else:
                value_at_zero = ds["pmf"][np.abs(x - zero_point).argmin()]
            ds["pmf"]     -= value_at_zero
            at["zero point"] = zero_point

        # Organize and return data
        at["free energy units"] = "kBT"
        at["pmf units"]         = "kcal mol-1"
        at["temperature"]       = self.temperature

class Hist_Block_Accumulator(PMF_Block_Accumulator):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a Block_Acceptor
    """
    def __init__(self, bins, attrs = {}, **kwargs):
        """
        Initializes accumulator

        **Arguments:**
            :*preexisting_slice*: slice containing frame indices whose results
                                  were included in *output* h5 file before
                                  this invocation of program
            :*incoming_slice*:    slice containting frame indices whose results
                                  are expected to be added to *output* h5 file
                                  after this invocation of program
            :*bins*:              Bins used to calculate histogram
            :*temperature*:       Simulation temperature used to calculate pmf
            :*zero_point*:        Point to shift pmf to zero, or range of
                                  points over which to shift average to zero
            :*output*:            List including path to h5 file and address
                                  within h5 file
            :*attrs*:             Attributes to add to dataset
        """
        from h5py import File as h5

        # Action
        self.bins = np.squeeze(np.array(eval(bins)))

        super(Hist_Block_Accumulator, self).__init__(**kwargs)

        # Prepare dataset
        ds = np.zeros(self.bins.size - 1,
          dtype = [("lower bound", "f4"), ("center",      "f4"),
                   ("upper bound", "f4"), ("count",       "i4"),
                   ("probability", "f8"), ("free energy", "f8"),
                   ("pmf",         "f8")])
        if self.preexisting_slice is not None:
            with h5(self.out_path) as out_h5:
                preexisting_ds    = np.array(out_h5[self.out_address])
                ds["lower bound"] = preexisting_ds["lower bound"]
                ds["center"]      = preexisting_ds["center"]
                ds["upper bound"] = preexisting_ds["upper bound"]
                ds["count"]      += preexisting_ds["count"]
        else:
            ds["lower bound"] =  self.bins[:-1]
            ds["center"]      = (self.bins[:-1] + self.bins[1:]) / 2
            ds["upper bound"] =  self.bins[1:]
        attrs["lower bound units"] = "A"
        attrs["center units"]      = "A"
        attrs["upper bound units"] = "A" 

        self.datasets = {self.out_address: dict(data = ds, attrs = attrs)}

    def receive_block(self, **kwargs):
        """
        Accumulates bin counts from recieved Blocks
        """
        ds = self.datasets[self.out_address]["data"]
        while True:
            block       = (yield)
            block_count = block.datasets[self.out_address]["count"]
            block_slice = block.datasets[self.out_address]["slc"]

            ds["count"] += block_count
            self.receive_slice(block_slice)

            print("Total incoming frames: [{0}:{1}]  ".format(
              self.incoming_slice.start, self.incoming_slice.stop) +
              "Received frames: {0}".format("".join(
              ["[{0}:{1}]".format(rs.start, rs.stop) for rs in 
              self.received_slices])))

    def close(self, **kwargs):
        """
        Estimates probability density function from histogram and
        passes on to superclass, which calculates free energy and
        potential of mean force
        """
        # Link to dataset for clarity
        ds = self.datasets[self.out_address]["data"]

        # Calculate final probability and PMF
        if self.incoming_slice is None:
            self.datasets = {}
        else:
            ds["probability"] = np.array(ds["count"],
              dtype = np.float32) / np.sum(ds["count"])
            ds["probability"][ds["probability"] == 0.0] = np.nan
            super(Hist_Block_Accumulator, self).close(x = ds["center"],
              **kwargs)

class KDE_Block_Accumulator(PMF_Block_Accumulator):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a Block_Acceptor
    """
    def __init__(self, grid, bandwidth, attrs = {}, **kwargs):
        """
        Initializes accumulator

        **Arguments:**
            :*preexisting_slice*: slice containing frame indices whose results
                                  were included in *output* h5 file before
                                  this invocation of program
            :*incoming_slice*:    slice containting frame indices whose results
                                  are expected to be added to *output* h5 file
                                  after this invocation of program
            :*grid*:              Grid on which to calculate pdf
            :*bandwidth*:         Kernel bandwidth
            :*temperature*:       Simulation temperature used to calculate pmf
            :*zero_point*:        Point to shift pmf to zero, or range of
                                  points over which to shift average to zero
            :*output*:            List including path to h5 file and address
                                  within h5 file
            :*attrs*:             Attributes to add to dataset
        """
        from h5py import File as h5
        
        # Action
        self.grid = np.squeeze(np.array(eval(grid)))

        super(KDE_Block_Accumulator, self).__init__(**kwargs)

        # Prepare dataset
        ds = np.zeros(self.grid.size,
          dtype = [("x",           "f4"), ("probability", "f8"),
                   ("free energy", "f4"), ("pmf",         "f4")])
        if self.preexisting_slice is not None:
            with h5(self.out_path) as out_h5:
                preexisting_ds     = np.array(out_h5[self.out_address])
                ds["x"]            = preexisting_ds["x"]
                ds["probability"]  = preexisting_ds["probability"]
                self.n_frames_received = (self.preexisting_slice.stop -
                  self.preexisting_slice.start)
        else:
            ds["x"] = self.grid
            self.n_frames_received = 0
        attrs["bandwidth"] = bandwidth

        self.datasets = {self.out_address: dict(data = ds, attrs = attrs)}

    def receive_block(self, **kwargs):
        """
        Accumulates probability density functions from recieved Blocks
        """
        ds = self.datasets[self.out_address]["data"]
        while True:
            block       = (yield)
            block_pdf   = block.datasets[self.out_address]["pdf"]
            block_slice = block.datasets[self.out_address]["slc"]

            # Weigh current and incoming pdfs appropriately; probably needs
            #   a long-term fix to ensure low error, but for now this is fine
            ds["probability"] = (
              (ds["probability"] * self.n_frames_received)
              + (block_pdf * (block_slice.stop - block_slice.start))
              ) / (self.n_frames_received 
              + (block_slice.stop - block_slice.start))

            self.n_frames_received += (block_slice.stop - block_slice.start)
            self.receive_slice(block_slice)

            print("Total incoming frames: [{0}:{1}]  ".format(
              self.incoming_slice.start, self.incoming_slice.stop) +
              "Received frames: {0}".format("".join(
              ["[{0}:{1}]".format(rs.start, rs.stop) for rs in 
              self.received_slices])))

    def close(self, **kwargs):
        """
        Passes probability density function on to superclass, which
        calculates free energy and potential of mean force
        """
        # Link to dataset for clarity
        ds = self.datasets[self.out_address]["data"]

        # Calculate final probability and PMF
        if self.incoming_slice is None:
            self.datasets = {}
        else:
            super(KDE_Block_Accumulator, self).close(x = ds["x"], **kwargs)

