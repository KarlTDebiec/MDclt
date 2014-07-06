#!/usr/bin/python
#   MDclt.primary.__init__.py
#   Written by Karl Debiec on 14-06-30, last updated by Karl Debiec on 14-07-05
"""
Classes and functions for primary analysis of molecular dynamics simulations
"""
####################################################### MODULES ########################################################
from __future__ import division, print_function
import os, sys
import numpy as np
####################################################### CLASSES ########################################################
class Primary_Analysis(object):
    """
    Base class for direct analyses of molecular dynamics trajectory, logs, or other output, including output from other
    analysis programs

    .. todo:
        - Add multiprocessing support to Primary_Analysis.command_line(...) using Pool (-np argument)
        - Support alternatives to -infiles, such as specifying a path and function to list segments
    """

    @classmethod
    def add_parser(cls, subparsers, *args, **kwargs):
        """
        Adds subparser for this analysis to a nascent argument parser

        **Arguments:**
            :*subparsers*: <argparse._SubParsersAction> to which to add
            :*\*args*:     Passed to *subparsers*.add_parser(...)
            :*\*\*kwargs*: Passed to *subparsers*.add_parser(...)
        """
        subparser = subparsers.add_parser(*args, **kwargs)
        arg_groups = {"input":  subparser.add_argument_group("input"),
                      "action": subparser.add_argument_group("action"),
                      "output": subparser.add_argument_group("output")}

        arg_groups["input"].add_argument("-infiles", type = str, required = True, action = "append", nargs = "*",
          help = "Input filenames")
        arg_groups["action"].add_argument("-n_cores", type = int, required = False, default = 1,
          help = "Number of cores on which to carry out analysis")
        arg_groups["output"].add_argument("-h5_file", type = str, required = True,
          help = "H5 file in which to output data")
        arg_groups["output"].add_argument("-address", type = str, required = True,
          help = "Location of dataset within h5 file")
        arg_groups["output"].add_argument("-attrs", type = str, required = False, action = "append", nargs = "*",
          metavar = "KEY VALUE", 
          help = "Attributes to add to dataset")
        arg_groups["output"].add_argument("--force", action = "store_true",
          help = "Overwrite data if already present")

        return subparser

    @classmethod
    def command_line(cls, h5_file, n_cores, **kwargs):
        """
        Provides basic command line functionality

        **Arguments:**
            :*cls*:     Class whose block_generator(...) and block_storer(...) will be used
            :*h5_file*: Filename of h5 file in which data will be stored
        """
        from multiprocessing import Pool

        block_generator = cls.block_generator(h5_file = h5_file, **kwargs)
        block_storer    = cls.block_storer(h5_file, **kwargs)
        block_storer.next()
        pool            = Pool(n_cores)

        for block in pool.imap_unordered(pool_director, block_generator):
            block_storer.send(block)

        pool.close()
        pool.join()
        block_storer.close()

    class block_generator(object):
        """
        Base generator class that yields blocks of analysis

        This is a class rather than a function, because it needs to perform initialization before the first call to
        next().
        """

        def __init__(self, **kwargs):
            """
            Initialize generator; typically, this function will look at input data and output h5 file and perform any
            necessary preparation.
            """
            raise NotImplementedError("'block_generator' class is not implemented")

        def __iter__(self):
            """
            Allow class to act as a generator
            """
            return self

        def _initialize(self, h5_file, address, dataset_kwargs = dict(chunks = True, compression = "gzip"), attrs = {},
            force = False, **kwargs):
            """
            Components of __init__ shared by subclasses

            **Arguments:**
                :*h5_file*:         Filename of h5 file in which data will be stored
                :*address*:         Address of dataset within h5 file
                :*dataset_kwargs*:  Keyword arguments to be passed to create_dataset(...)
                :*attrs*:           Attributes to be added to dataset
                :*force*:           Run analysis even if all data is already present

            .. todo:
                - Make more general
                - Support multiple datasets with multiple addresses, probably using syntax similar to block_storer
                - Allow specification of where attrs should be stored
            """
            import h5py

            with h5py.File(h5_file) as h5_file:
                if address in h5_file:
                    if force:
                        del h5_file[address]
                        h5_file.create_dataset(address, data = np.empty(self.expected_shape, self.dtype),
                          **dataset_kwargs)
                    else:
                        dataset       = h5_file[address]
                        current_shape = dataset.shape
                        if self.expected_shape != current_shape:
                            dataset.resize(size = self.expected_shape)
                            self.infiles        = self.infiles[int(current_shape[0] / self.frames_per_file):]
                            self.current_index  = current_shape[0]
                        else:
                            self.infiles = []
                else:
                    h5_file.create_dataset(address, data = np.empty(self.expected_shape, self.dtype),
                      **dataset_kwargs)
                for key, value in attrs.items():
                    h5_file[self.address].attrs[key] = value

        def next(self):
            """
            Prepares and yields next block of analysis
            """
            raise NotImplementedError("'block_generator' class is not implemented")

    @classmethod
    def block_storer(cls, h5_file, **kwargs):
        """
        Coroutine that accepts analysis blocks and stores their associated datasets in h5

        **Arguments:**
            :*h5_file*: Filename of h5 file in which data will be stored

        .. todo:
            - Decorate to hide call to block_storer.next()?
            - Support different levels of verbosity
              (i.e. 'Dataset ... was extended from ... to ...')
        """
        import h5py

        with h5py.File(h5_file) as h5_file:
            try:
                while(True):
                    block = yield
                    for dataset in block.datasets:
                        if "slc" in dataset:
                            h5_file[dataset["address"]][dataset["slc"]] = dataset["data"]
                            print("Dataset stored at {0}[{1}][{2}:{3}]".format(h5_file.filename,
                              dataset["address"], dataset["slc"].start, dataset["slc"].stop))
                        else:
                            h5_file[dataset["address"]]                 = dataset["data"]
                            print("Dataset stored at {0}[{1}]".format(h5_file.filename,
                              dataset["address"]))
                        if "attrs" in dataset:
                            for key, value in dataset["attrs"].items():
                                h5_file[dataset["address"]].attrs[key] = value
            except GeneratorExit:
                # Appears necessary to properly __exit__ h5 file
                pass

    def __init__(self, *args, **kwargs):
        """
        Initializes a block of analysis
        """
        # Appears necessary to allow subclasses of subclasses of this class to pass *args, **kwargs to the __init__
        #   methods of the intermediate superclasses.
        pass

    def analyze(self, **kwargs):
        """
        Runs this block of analysis; stores resulting data in an instance variable
        """
        raise NotImplementedError("'analyze' function of '{0}' is not implemented".format(self.__class__.__name__))

###################################################### FUNCTIONS #######################################################
def pool_director(block):
    """
    Allows multiprocessing.Pool(...) to run a block of analysis

    .. todo:
        - multiprocessing.Pool(...) only supports module-level classes; implement alternative method using lower-level
          components of multiprocessing to allow cleaner design
    """
    block.analyze()
    return block


