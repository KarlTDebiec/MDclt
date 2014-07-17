#!/usr/bin/python
#   MDclt.__init__.py
#   Written by Karl Debiec on 12-02-12, last updated by Karl Debiec on 14-07-16
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

    def analyze(self, **kwargs):
        """
        Runs this block of analysis
        """
        raise NotImplementedError("'Block' class is not implemented")

class Block_Generator(object):
    """
    Base generator class that yields blocks of analysis
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
        Prepares and yields next Block of analysis
        """
        raise NotImplementedError("'Block_Generator' class is not implemented")

class Block_Accumulator(object):
    """
    Coroutine class used to accumulate Blocks of data and perform analysis once
    the complete dataset is present; may also act as a Block itself and be
    added by a Block_Acceptor
    """
    def __init__(self, **kwargs):
        """
        Initializes wrapped function
        """
        pass

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

class Block_Acceptor(object):
    """
    Coroutine class used to store Blocks of data in an h5 file
    """

    def __init__(self, **kwargs):
        """
        Initializes wrapped function
        """
        self.func     = self.store()
        self.out_path = kwargs.get("out_path", kwargs.pop("output")[0])

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
        Terminates wrapped function
        """
        self.func.close(*args, **kwargs)

    def store(self, **kwargs):
        """
        Stores data in an h5 file

        .. todo:
            - Support dataset_kwargs
            - Support more complex addresses within the h5 file
        """
        from h5py import File as h5

        with h5(self.out_path) as out_h5:
            try:
                while(True):
                    block = yield
                    for address, dataset in block.datasets.items():
                        data = dataset["data"]
                        if "slc" in dataset:
                            slc = dataset["slc"]
                            out_h5[address][slc] = data
                            print("Dataset stored at {0}[{1}][{2}:{3}]".format(
                              out_h5.filename, address, slc.start, slc.stop))
                        else:
                            if address in out_h5:
                                del out_h5[address]
                            out_h5[address] = data
                            print("Dataset stored at {0}[{1}]".format(
                              out_h5.filename, address))
                        if "attrs" in dataset:
                            attrs = dataset["attrs"]
                            for key, value in attrs.items():
                                out_h5[address].attrs[key] = value
            except GeneratorExit:
                # Appears necessary to properly __exit__ h5 file
                pass


