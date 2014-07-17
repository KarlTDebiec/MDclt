#!/usr/bin/python
#   MDclt.primary.amber.py
#   Written by Karl Debiec on 12-12-01, last updated by Karl Debiec on 14-07-10
"""
Classes for transfer of AMBER simulation logs to h5

.. todo:
    - Better support appending data rather than reloading complete dataset
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
from MDclt import primary
from MDclt import Block, Block_Acceptor
################################## FUNCTIONS ###################################
def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: argparse subparsers object to add subparser
        :*args*:       Passed to subparsers.add_parser(...)
        :*kwargs*:     Passed to subparsers.add_parser(...)

    .. todo:
        - Implement nested subparser (should be 'amber log', not just 'log')
    """
    subparser  = primary.add_parser(subparsers,
      name     = "log",
      help     = "Load AMBER logs")
    arg_groups = {ag.title:ag for ag in subparser._action_groups}

    arg_groups["input"].add_argument(
      "-frames_per_file",
      type     = int,
      required = False,
      help     = "Number of frames in each file; used to check if new data " +
                 "is present (optional)")
    arg_groups["input"].add_argument(
      "-start_time",
      type     = float,
      required = False,
      help     = "Time of first frame (ns) (optional)")

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


################################### CLASSES ####################################
class Block(Block):
    """
    Independent block of analysis
    """
    def __init__(self, infiles, raw_keys, new_keys, dtype, address, slc,
                 time_offset = 0, attrs = {}, *args, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*infiles*:     List of infiles
            :*raw_keys*:    Original names of fields in Amber mdout
            :*new_keys*:    Desired names of fields in nascent dataset
            :*dtype*:       Data type of nascent dataset
            :*address*:     Address of dataset within h5 file
            :*slc*:         Slice within dataset at which this block
                            will be stored
            :*time_offset*: Offset by which to adjust simulation time
            :*attrs*:       Attributes to add to dataset
        """
        from collections import OrderedDict

        super(Block, self).__init__(*args, **kwargs)
        self.infiles     = infiles
        self.raw_keys    = raw_keys
        self.new_keys    = new_keys
        self.time_offset = time_offset
        self.address     = address
        self.datasets    = OrderedDict({address:
                             dict(slc = slc, attrs = attrs,
                               data = np.empty(slc.stop - slc.start, dtype))})

    def __call__(self, **kwargs):
        """
        Runs this block of analysis
        """

        # Load raw data from each infile
        raw_data = {raw_key: [] for raw_key in self.raw_keys}
        for infile in self.infiles:
            with open(infile, "r") as infile:
                raw_text = [line.strip() for line in infile.readlines()]
            i = 0
            while i < len(raw_text):
                if raw_text[i].startswith("A V E R A G E S"): break
                if raw_text[i].startswith("NSTEP"):
                    while True:
                        if raw_text[i].startswith("----------"): break
                        line = raw_text[i].split("=")
                        for j, field in enumerate(line[:-1]):
                            if j == 0:
                                raw_key = field.strip()
                            else:
                                raw_key = " ".join(field.split()[1:])
                            value = line[j+1].split()[0]
                            if raw_key in self.raw_keys:
                                raw_data[raw_key] += [value]
                        i += 1
                i += 1

        # Copy from raw_data to new_data
        self.datasets[self.address]["data"]["time"] = (np.array(
          raw_data["TIME(PS)"], np.float) / 1000) + self.time_offset
        for raw_key, new_key in zip(self.raw_keys[1:], self.new_keys[1:]):
            self.datasets[self.address]["data"][new_key] = np.array(
              raw_data[raw_key])

class Block_Generator(primary.Block_Generator):
    """
    Generator class that yields blocks of analysis
    """

    fields = [("TIME(PS)",  "time",                      "ns"),
              ("Etot",      "total energy",              "kcal mol-1"),
              ("EPtot",     "potential energy",          "kcal mol-1"),
              ("EKtot",     "kinetic energy",            "kcal mol-1"),
              ("BOND",      "bond energy",               "kcal mol-1"),
              ("ANGLE",     "angle energy",              "kcal mol-1"),
              ("DIHED",     "dihedral energy",           "kcal mol-1"),
              ("EELEC",     "coulomb energy",            "kcal mol-1"),
              ("1-4 EEL",   "coulomb 1-4 energy",        "kcal mol-1"),
              ("VDWAALS",   "van der Waals energy",      "kcal mol-1"),
              ("1-4 NB",    "van der Waals 1-4 energy",  "kcal mol-1"),
              ("EHBOND",    "hydrogen bond energy",      "kcal mol-1"),
              ("RESTRAINT", "position restraint energy", "kcal mol-1"),
              ("EKCMT",     "center of mass motion kinetic energy",
                                                         "kcal mol-1"),
              ("VIRIAL",    "virial energy",             "kcal mol-1"),
              ("EPOLZ",     "polarization energy",       "kcal mol-1"),
              ("TEMP(K)",   "temperature",               "K"),
              ("PRESS",     "pressure",                  "bar"),
              ("VOLUME",    "volume",                    "A3"),
              ("Density",   "density",                   "g/cm3"),
              ("Dipole convergence: rms",
                            "dipole convergence rms",    None),
              ("iters",     "dipole convergence iterations",
                                                         None)]

    def __init__(self, output, infiles, frames_per_file = None, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*output*:          List including path to h5 file and
                                address within h5 file
            :*infiles*:         List of infiles
            :*frames_per_file*: Number of frames in each infile
                                (optional)

        .. todo:
            - Intelligently break lists of infiles into blocks larger
              than 1
        """

        # Store necessary information in instance variables
        out_path, out_address = output
        self.address          = out_address
        self.infiles          = infiles
        self.frames_per_file  = frames_per_file
        self.expected_shape   = [len(self.infiles) * self.frames_per_file]
        self.current_index    = 0

        # Adjust start time, if applicable
        self.get_time_offset(**kwargs)

        # Determine structure of input data
        self.get_dataset_format(output = output, **kwargs)

        # Disregard last infile, if applicable
        self.cut_incomplete_infiles(**kwargs)

        # Complete initialization 
        dataset_kwargs = dict(chunks = True, compression = "gzip",
          maxshape = [None])
        super(Block_Generator, self).__init__(output = output,
          dataset_kwargs = dataset_kwargs, **kwargs)

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
            return Block(infiles     = [self.infiles.pop(0)],
                         raw_keys    = self.raw_keys,
                         new_keys    = self.new_keys,
                         dtype       = self.dtype,
                         address     = self.address,
                         slc         = slice(slice_start, slice_end, 1),
                         time_offset = self.time_offset)

    def get_time_offset(self, start_time = None, **kwargs):
        """
        Calculates time offset based on desired and actual time of first frame

        **Arguments:**
            :*start_time*: Desired time of first frame (ns); typically 0.001
        """
        import subprocess

        if start_time is None:
            self.time_offset = 0
        else:
            with open(os.devnull, "w") as fnull:
                command = "cat {0} | ".format(self.infiles[0]) + \
                          "grep -m 1 'TIME(PS)' | " + \
                          "awk '{{print $6}}'"
                process = subprocess.Popen(command,
                            stdout = subprocess.PIPE,
                            stderr = fnull,
                            shell  = True)
                result  = process.stdout.read()
            self.time_offset = float(result) / -1000 + start_time

    def get_dataset_format(self, output, **kwargs):
        """
        Determines format of dataset
        """
        from h5py import File as h5

        out_path, out_address = output

        with h5(out_path) as out_h5:
            if out_address in out_h5:
                # If dataset already exists, extract current dtype
                self.dtype    = out_h5[out_address].dtype
                self.new_keys = list(self.dtype.names)
                self.raw_keys = []
                for key in self.new_keys:
                    self.raw_keys += [r for r, n, _ in self.fields if n == key]
                self.attrs    = dict(out_h5[out_address].attrs)
            else:
                # Otherwise, determine fields present in infile
                raw_keys = []
                breaking = False
                with open(self.infiles[0], "r") as infile:
                    raw_text = [line.strip() for line in infile.readlines()]
                for i in xrange(len(raw_text)):
                    if breaking:  break
                    if raw_text[i].startswith("NSTEP"):
                        while True:
                            if raw_text[i].startswith("----------"):
                                breaking = True
                                break
                            for j, field in enumerate(
                              raw_text[i].split("=")[:-1]):
                                if j == 0:
                                    raw_keys += [field.strip()]
                                else:
                                    raw_keys += [" ".join(field.split()[1:])]
                            i += 1

                # Determine appropriate dtype of new data
                self.raw_keys = ["TIME(PS)"]
                self.new_keys = ["time"]
                self.dtype    = [("time", "f4")]
                self.attrs    = {"time units": "ns"}
                for raw_key, new_key, units in self.fields[1:]:
                    if raw_key in raw_keys:
                        self.raw_keys  += [raw_key]
                        self.new_keys  += [new_key]
                        self.dtype     += [(new_key, "f4")]
                        if units is not None:
                            self.attrs[new_key + " units"] = units

    def cut_incomplete_infiles(self, **kwargs):
        """
        Checks if log of last infile is incomplete; if so removes from
        list of infiles
        """
        import subprocess

        with open(os.devnull, "w") as fnull:
            command = "tail -n 1 {0}".format(self.infiles[-1])
            process = subprocess.Popen(command,
                        stdout = subprocess.PIPE,
                        stderr = fnull,
                        shell  = True)
            result  = process.stdout.read()
        if not (result.startswith("|  Total wall time:")          # pmemd.cuda
           or   result.startswith("|  Master Total wall time:")): # pmemd
            self.infiles.pop(-1)
            self.expected_shape[0] -= self.frames_per_file


