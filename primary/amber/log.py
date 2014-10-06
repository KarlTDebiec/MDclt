#!/usr/bin/python
#   MDclt.primary.amber.py
#   Written by Karl Debiec on 12-12-01, last updated by Karl Debiec on 14-09-29
"""
Classes for transfer of AMBER simulation logs to h5
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
        :*tool_subparsers*: Argparse subparsers object to add subparser
        :*args*:            Passed to tool_subparsers.add_parser(...)
        :*\*\*kwargs*:      Passed to tool_subparsers.add_parser(...)

    .. todo:
        - Implement nested subparser (should be 'amber log', not just 'log')
    """
    from MDclt import overridable_defaults

    subparser  = primary.add_parser(tool_subparsers,
      name     = "log",
      help     = "Load AMBER logs")
    arg_groups = {ag.title:ag for ag in subparser._action_groups}

    arg_groups["input"].add_argument(
      "-frames_per_file",
      type     = int,
      required = False,
      help     = "Number of frames in each file; used to check if new data " +
                 "is present")
    arg_groups["input"].add_argument(
      "-start_time",
      type     = float,
      required = False,
      help     = "Time of first frame (ns) (optional)")

    arg_groups["output"].add_argument(
      "-output",
      type     = str,
      required = True,
      nargs    = "+",
      action   = overridable_defaults(nargs = 2, defaults = {1: "/log"}),
      help     = "H5 file and optionally address in which to output data " +
                 "(default address: /log)")

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

    block_generator = AmberLog_Block_Generator(**kwargs)
    block_acceptor  = Block_Acceptor(outputs = block_generator.outputs,
                        **kwargs)
    if n_cores == 1:                # Serial
        for block in block_generator:
            block()
            block_acceptor.send(block)
    else:                           # Parallel (processes)
        pool = Pool(n_cores)
        for block in pool.imap_unordered(pool_director, block_generator):
            pass
            block_acceptor.send(block)
        pool.close()
        pool.join()

    block_acceptor.close()

################################### CLASSES ####################################
class AmberLog_Block_Generator(primary.Primary_Block_Generator):
    """
    Generator class that prepares blocks of analysis
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

    def __init__(self, infiles, output, frames_per_file = None, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*output*:          List including path to h5 file and
                                address within h5 file
            :*infiles*:         List of infiles
            :*frames_per_file*: Number of frames in each infile

        .. todo:
            - Intelligently break lists of infiles into blocks larger
              than 1
        """

        # Input
        self.infiles           = infiles
        self.frames_per_file   = frames_per_file
        self.infiles_per_block = 5

        # Output
        self.outputs = [(output[0], os.path.normpath(output[1]))]

        # Adjust start time, if applicable
        self.get_time_offset(**kwargs)

        # Determine dtype of input data
        self.get_dataset_format(**kwargs)

        # Disregard last infile, if applicable
        self.cut_incomplete_infiles(**kwargs)

        super(AmberLog_Block_Generator, self).__init__(**kwargs)

        # Output
        self.outputs = [(output[0], os.path.normpath(output[1]),
          (self.final_slice.stop - self.final_slice.start,))]

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
            return AmberLog_Block(infiles     = block_infiles,
                                  raw_keys    = self.raw_keys,
                                  new_keys    = self.new_keys,
                                  output      = self.outputs[0],
                                  slc         = block_slice,
                                  time_offset = self.time_offset,
                                  dtype       = self.dtype)

    def get_time_offset(self, start_time = None, **kwargs):
        """
        Calculates time offset based on desired and actual time of first frame

        **Arguments:**
            :*start_time*: Desired time of first frame (ns); typically 0.001
        """
        from subprocess import Popen, PIPE

        if start_time is None:
            self.time_offset = 0
        else:
            with open(os.devnull, "w") as fnull:
                command = "cat {0} | ".format(self.infiles[0]) + \
                          "grep -m 1 'TIME(PS)' | " + \
                          "awk '{{print $6}}'"
                process = Popen(command,
                            stdout = PIPE,
                            stderr = fnull,
                            shell  = True)
                result  = process.stdout.read()
            self.time_offset = float(result) / -1000 + start_time

    def get_dataset_format(self, **kwargs):
        """
        Determines format of dataset
        """
        from h5py import File as h5

        out_path, out_address = self.outputs[0]

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
        from subprocess import Popen, PIPE

        with open(os.devnull, "w") as fnull:
            command = "tail -n 1 {0}".format(self.infiles[-1])
            process = Popen(command,
                        stdout = PIPE,
                        stderr = fnull,
                        shell  = True)
            result  = process.stdout.read()
        if not (result.startswith("|  Total wall time:")          # pmemd.cuda
           or   result.startswith("|  Master Total wall time:")): # pmemd
            self.infiles.pop(-1)
            self.final_shape[0] -= self.frames_per_file

class AmberLog_Block(Block):
    """
    Independent block of analysis
    """
    def __init__(self, infiles, raw_keys, new_keys, output, dtype, slc,
          time_offset = 0, attrs = {}, **kwargs):
        """
        Initializes block of analysis

        **Arguments:**
            :*infiles*:     List of infiles
            :*raw_keys*:    Original names of fields in Amber mdout
            :*new_keys*:    Desired names of fields in nascent dataset
            :*output*:      Path to h5 file and address within h5 file
            :*dtype*:       Data type of nascent dataset
            :*slc*:         Slice within dataset at which this block
                            will be stored
            :*time_offset*: Offset by which to adjust simulation time
            :*attrs*:       Attributes to add to dataset
        """
        super(AmberLog_Block, self).__init__(**kwargs)

        self.infiles     = infiles
        self.raw_keys    = raw_keys
        self.new_keys    = new_keys
        self.time_offset = time_offset
        self.output      = output
        self.datasets    = {self.output: dict(slc = slc, attrs = attrs,
                             data = np.empty(slc.stop - slc.start, dtype))}

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
        self.datasets[self.output]["data"]["time"] = (np.array(
          raw_data["TIME(PS)"], np.float) / 1000) + self.time_offset
        for raw_key, new_key in zip(self.raw_keys[1:], self.new_keys[1:]):
            self.datasets[self.output]["data"][new_key] = np.array(
              raw_data[raw_key])


