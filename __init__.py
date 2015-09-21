# -*- coding: utf-8 -*-
#   MDclt.__init__.py
#
#   Copyright (C) 2012-2015 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
"""
Command line tools for analysis of molecular dynamics simulations

.. todo:
    - Documentation
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
################################## FUNCTIONS ###################################
def pool_director(block):
    """
    Allows multiprocessing.Pool(...) to run blocks of analysis obtained from an
    iterator

    .. todo:
        - multiprocessing.Pool(...) only supports module-level classes;
          implement alternative method using lower-level components of
          multiprocessing to allow cleaner design
    """
    block()
    return block

def clean_path(path, strip = None, **kwargs):
    """
    Cleans a path

    **Arguments:**
        :*path*:  Initial path string
        :*strip*: Potential endings to remove from path

    **Returns:
        :*cleaned*: Cleaned path
    """
    cleaned = path

    if   strip is None:           strip  = []
    elif isinstance(strip, str):  strip  = [strip]

    for s in strip:
        if cleaned.endswith(s):
            cleaned = cleaned[:cleaned.index(s)]

    cleaned = ("/" + cleaned + "/").replace("//", "/")
    return cleaned

def overridable_defaults(nargs, defaults):
    """
    Prepares a custom argparse action to provide optionally overridable
    defaults for arguments with multiple values

    **Arguments:**
        :*defaults*: dictionary of defaults; keys are the integer index
                     within the final list of values
    """
    import argparse, types

    class Argparse_Action(argparse.Action):
        """
        Custom argparse action to provide optionally overridable
        defaults for arguments with multiple values
        """
        def __call__(self, parser, args, values, option_string = None):
            """
            Processes argument values, applying defaults where appropriate

            .. todo:
                Check that argument numbers are sequential
            """
            if not isinstance(values, types.ListType):
                values = [values]
            values = dict(defaults.items() +
                          {k: v for k, v in enumerate(values)}.items())
            values = [values[k] for k in sorted(values)]
            if len(values) != nargs:
                raise argparse.ArgumentTypeError(
                  "Expected {0} arguments for '{1}', recieved {2}".format(
                  nargs, self.dest, len(values)))
            setattr(args, self.dest, values)

    return Argparse_Action

def h5_default_path():
    """
    Prepares a custom argparse action to automatically select the path
    to a dataset or group within an h5 file, if there is only one path
    or group present in that h5 file
    """
    import argparse, types

    class Argparse_Action(argparse.Action):
        """
        Custom argparse action to automatically select the path
        to a dataset or group within an h5 file, if there is only one
        path or group present in that h5 file
        """
        def __call__(self, parser, args, values, option_string = None):
            """
            Processes argument values, automatically selected address
            if applicable
            """
            from h5py import File as h5
            if len(values) == 1:
                in_path = values[0]
                with h5(in_path) as in_h5:
                    if len(in_h5.keys()) == 1:
                        values = [in_path, in_h5.keys()[0]]
                    else:
                        raise argparse.ArgumentTypeError(
                          "Unable to determine target address for " +
                          "'{0}', ".format(self.dest) +
                          "in file '{0}', ".format(in_path) +
                          "specify manually with second argument")
            elif len(values) == 2:
                in_path, in_address = values
                with h5(in_path) as in_h5:
                    if not in_address in in_h5:
                        raise argparse.ArgumentTypeError(
                          "Target address '{0}' ".format(in_address) +
                          "not found in file '{0}'. ".format(in_path))
            else:
                raise argparse.ArgumentTypeError(
                  "Expected {0} ".format(nargs) +
                  "arguments for '{0}', ".format(self.dest) +
                  "recieved {0}".format(len(values)))
            setattr(args, self.dest, values)
    return Argparse_Action

def parse_states(in_states, **kwargs):
    from collections import OrderedDict

    states = OrderedDict()
    for in_state in in_states:
        state_name = in_state.split(":")[0].strip()
        states[state_name] = {}
        for i, coord in enumerate(in_state.split(":")[1].split(";")):
            coord_name = i
            min_cutoff = 0.0
            max_cutoff = np.inf
            for field in coord.split():
                if   field.startswith("<"):
                    max_cutoff = float(field[1:])
                elif field.startswith(">"):
                    min_cutoff = float(field[1:])
                else:
                    coord_name = field
            states[state_name][coord_name] = [min_cutoff, max_cutoff]
    return states

################################### CLASSES ####################################
class Block(object):
    """
    Independent block of analysis
    """
    def __init__(self, **kwargs):
        """
        Initializes block of analysis
        """
        pass

    def __call__(self, **kwargs):
        """
        Runs this block of analysis
        """
        raise NotImplementedError("'Block' class is not implemented")

class Block_Generator(object):
    """
    Generator class that prepares blocks of analysis
    """
    def __init__(self, **kwargs):
        """
        Initialize generator
        """
        pass

    def __iter__(self, **kwargs):
        """
        Allow class to act as a generator
        """
        return self

    def next(self, **kwargs):
        """
        Prepare and return next Block of analysis
        """
        raise NotImplementedError("")

    def get_final_slice(self, **kwargs):
        """
        """
        raise NotImplementedError("")

    def get_preexisting_slice(self, force = False, **kwargs):
        """
        Determines slice to which preexisting data in outputs
        corresponds

        NOTE: Should not initialize empty dataset here, as
           the dtype may be unknown until the analysis is
           started or completed

        **Arguments**
            :*force*: Even if preexisting data is present, disregard
        """
        from warnings import warn
        from h5py import File as h5

        out_starts = []
        out_stops  = []
        for output in self.outputs:
            if len(output) == 2:
                out_path, out_address = output
            elif len(output) == 3:
                out_path, out_address, out_shape = output
            with h5(out_path) as out_h5:
                if out_address in out_h5:
                    if force:
                        del out_h5[out_address]
                        out_starts += [np.nan]
                        out_stops  += [np.nan]
                    else:
                        attrs = dict(out_h5[out_address].attrs)
                        if "slice" in attrs:
                            out_starts += [eval(attrs["slice"]).start]
                            out_stops  += [eval(attrs["slice"]).stop]
                        else:
#                            warn("'slice' not found in dataset " +
#                              "'{0}:{1}' attributes; ".format(out_path,
#                              out_address) +
#                              "assuming first dimension is time")
                            out_starts += [0]
                            out_stops  += [out_h5[out_address].shape[0]]
                else:
                    out_starts += [np.nan]
                    out_stops  += [np.nan]
        if force:
            self.preexisting_slice = None
        elif len(set(out_starts)) != 1 or len(set(out_stops)) != 1:
            raise Exception("Preexising output datasets correspond to " +
              "different slices, use '--force' to overwrite all")
        elif np.isnan(out_starts[0]) or np.isnan(out_stops[0]):
            self.preexisting_slice = None
        else:
            self.preexisting_slice = slice(out_starts[0], out_stops[0], 1)

class Block_Accumulator(object):
    """
    Coroutine class; accumulates Blocks of data and performs analysis
    once complete dataset is present; may then be sent to a Block_Acceptor
    """
    def __init__(self, **kwargs):
        """
        Initializes wrapped function
        """
        self.func = self.receive_block()
        self.next()

    def next(self, **kwargs):
        """
        Moves wrapped function to first yield statement
        """
        self.func.next()

    def send(self, *args, **kwargs):
        """
        Transfers arguments to wrapped function
        """
        self.func.send(*args, **kwargs)

    def receive_block(self, **kwargs):
        """
        Accumulates received Blocks
        """
        raise NotImplementedError(
          "'Block_Accumulator' class is not implemented")

    def receive_slice(self, slc, **kwargs):
        """
        Acknowleges a portion (slice) of the dataset to have been
        received; maintains sorted array of received slices, and
        merges adjacent slices as they are received.

        **Arguments:**
            :*slc*: Received slice
        """
        rs    = sorted(self.received_slices + [slc], key = lambda s: s.start)
        max_i = len(rs) - 1
        i     = 0
        while i < max_i:
            if rs[i].stop == rs[i+1].start:
                rs    = rs[:i] + [slice(rs[i].start,rs[i+1].stop, 1)] + rs[i+2:]
                max_i = len(rs) - 1
                i    -= 1
            i += 1
        self.received_slices = rs

    def close(self, *args, **kwargs):
        """
        Performs analysis and terminates wrapped function
        """
        self.func.close(*args, **kwargs)

class Block_Acceptor(object):
    """
    Coroutine class used to store Blocks of data in an h5 file

    .. todo:
        - Reconsider how receive_block and receive_slice should work;
          some datasets have multidimensional shapes and the output
          for which receive_slice is intended may be too confusing
          to be useful; may be preferable to move receive_block's
          output to another function, allowing subclasses to easily
          format the messages appropriately; consider base class and
          subclass; base class accepts blocks in a time series and
          gives appropriate output, and subclass accepts blocks
          corresponding to dimensions greater than 0
    """

    def __init__(self, outputs, **kwargs):
        """
        Initializes wrapped function

        **Arguments:**
            :*outputs*: List of tuples of output datasets; first item
                        is path to h5 file, second item is address
                        within h5 file, third item is the final shape
                        of the dataset once all blocks have been
                        received (optional? probably required for
                        datasets that arrive in multiple blocks; but
                        some datasets' (e.g. lists of binding events)
                        sizes are not known until the analysis is done
        """
        self.outputs = outputs

        self.func = self.receive_block()
        self.next()

    def next(self, **kwargs):
        """
        Moves wrapped function to first yield statement
        """
        self.func.next()

    def send(self, *args, **kwargs):
        """
        Transfers arguments to wrapped function
        """
        self.func.send(*args, **kwargs)

    def close(self, *args, **kwargs):
        """
        Performs analysis and terminates wrapped function
        """
        self.func.close(*args, **kwargs)

    def receive_block(self, **kwargs):
        """
        Stores a block of data in an h5 file
        """
        from h5py import File as h5

        try:
            out_h5s = {out_path:h5(out_path) for out_path in
                        set([output[0] for output in self.outputs])}

            while(True):

                block = yield

                if not hasattr(block, "datasets"):
                    continue

                for output, dataset in block.datasets.items():
                    if len(output) == 2:
                        out_path, out_address = output
                    elif len(output) == 3:
                        out_path, out_address, out_shape = output
                    out_h5 = out_h5s[out_path]

                    if "slc" in dataset:
                        out_slc = dataset["slc"]
                        if out_address in out_h5:
                            if out_h5[out_address].shape != out_shape:
                                out_h5[out_address].resize(size = out_shape)
                            
                        else:
                            out_dtype  = dataset["data"].dtype
                            out_kwargs = dict(
                              chunks      = True,
                              compression = "gzip")
                            out_kwargs.update(
                              dataset.get("kwargs", {}))
                            out_kwargs["maxshape"] = (None,) + out_shape[1:]
                            out_h5.create_dataset(out_address,
                              data = np.empty(out_shape, out_dtype),
                              **out_kwargs)
                        out_h5[out_address][out_slc] = dataset["data"]
                        if isinstance(out_slc, slice):
                            print("Dataset stored at {0}[{1}][{2}:{3}]".format(
                              out_path, out_address, out_slc.start, out_slc.stop))
                        elif isinstance(out_slc, tuple):
                            message = "Dataset stored at {0}[{1}][".format(
                              out_path, out_address)
                            for out_slc_dim in out_slc:
                                if isinstance(out_slc_dim, slice):
                                    message += "{0}:{1},".format(
                                      out_slc_dim.start, out_slc_dim.stop)
                                elif isinstance(out_slc_dim, int):
                                    message += "{0},".format(out_slc_dim)
                                else:
                                    raise Exception(
                                      "output slice not understood")
                            message = message[:-1] + "]"
                            print(message)
                        else:
                            raise Exception("output slice not understood")
                    else:
                        if out_address in out_h5:
                            del out_h5[out_address]
                        out_h5[out_address] = dataset["data"]
                        print("Dataset stored at {0}[{1}]".format(
                          out_path, out_address))

                    if "attrs" in dataset:
                        for key, value in dataset["attrs"].items():
                            out_h5[out_address].attrs[key] = value
        except GeneratorExit:
            pass
        finally:
            for out_h5 in out_h5s.values():
                out_h5.flush()
                out_h5.close()

    def acknowledge_receipt(self, **kwargs):
        pass

    def receive_slice(self, slc, **kwargs):
        """
        Acknowleges a portion (slice) of the dataset to have been
        received; maintains sorted array of received slices, and
        merges adjacent slices as they are received.

        **Arguments:**
            :*slc*: Received slice
        """
        rs    = sorted(self.received_slices + [slc], key = lambda s: s.start)
        max_i = len(rs) - 1
        i     = 0
        while i < max_i:
            if rs[i].stop == rs[i+1].start:
                rs    = rs[:i] + [slice(rs[i].start,rs[i+1].stop, 1)] + rs[i+2:]
                max_i = len(rs) - 1
                i    -= 1
            i += 1
        self.received_slices = rs
