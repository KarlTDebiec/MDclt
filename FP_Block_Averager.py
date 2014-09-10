#!/usr/bin/python
# -*- coding: utf-8 -*-
#   FP_Block_Averager.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-09-10
"""
Command-line tool to estimate standard error using the block-averaging
method of Flyvbjerg and Petersen

Flyvbjerg, H., and Petersen, H. G. Error estimates on averages of
correlated data. Journal of Chemical Physics. 1989. 91 (1). 461-466.
"""
################################### MODULES ####################################
from __future__ import division, print_function
import os, sys
import numpy as np
import matplotlib.pyplot as plt
################################### CLASSES ####################################
class FP_Block_Averager(object):
    """
    Class to manage estimation of standard error using the
    block-averaging method of Flyvbjerg and Petersen
    """
    def __init__(self, dataset, name = None, func = None, debug = False,
        **kwargs):
        """
        Initializes FP_Block_Averager

        **Arguments:**
            :*dataset*: Dataset to which to apply block transform
    
            :*func*:    Function to use for block transform
            :*debug*:   Output debug information
        """
        from time import strftime

        self.dataset = dataset
        if name is None:
            self.name = strftime("%Y-%m-%d %H:%M:%S")
        else:
            self.name = name

        if func is None:
            def block_mean(*args, **kwargs):
                """
                Averages along second axis
                """
                return np.mean(*args, axis = 1, **kwargs)
            self.func = block_mean
        else:
            self.func = func

        self.debug = debug

        if self.debug:
            print("DATASET NAME: {0}".format(self.name))

    def _select_sizes_legacy(self, min_size = 1, **kwargs):
        self.full_size = self.dataset.size
        sizes = [s for s in list(set([self.full_size / s
                   for s in range(1,  self.full_size)]))
                   if  s >= min_size]
        self.sizes  = np.array(sorted(sizes), np.int)[:-1]

    def _calculate_blocks_legacy(self, **kwargs):
        sds         = np.zeros(self.sizes.size)
        n_blocks    = self.full_size // self.sizes
        for i, size in enumerate(self.sizes):
            resized = np.resize(self.dataset, (self.full_size // size, size))
            values  = self.func(resized)
            sds[i]  = np.std(values)
        ses         = sds / np.sqrt(n_blocks - 1)
        # se_sds = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
        se_sds      = (1.0 / np.sqrt(2.0 * (n_blocks - 1))) * ses
        if ses[-1] == 0.0 or se_sds[-1] == 0.0: # This happens occasionally and
            sizes   = sizes[:-1]                #  disrupts curve_fit; it is not
            ses     = ses[:-1]                  #  clear why
            se_sds  = se_sds[:-1]
        self.ses    = ses 
        self.se_sds = se_sds

    def _fit_curvei_legacy(self, cut = 0, **kwargs):
        import warnings
        from scipy.optimize import curve_fit

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            def func(x, a, b, c):
                return a + b * np.exp(c * x)
            a, b, c = curve_fit(func, self.sizes[cut:], self.ses[cut:], 
                sigma = self.se_sds[cut:], **kwargs)[0]
            self.fit = func(self.sizes, a, b, c)
            self.fit_parameters = (a, b, c)

    def select_sizes(self, factor = 1, min_size = 2, **kwargs):
        """
        Selects block sizes to be used for block transformations

        **Arguments:**
            :*factor*:   Factor by which all block sizes must be
                         divisible
            :*min_size*: Minimum block size

        .. todo:
            - Implement factor
            - Consider more carefully how important it is for all
              blocks to be exactly the same size
        """
        self.full_size     = self.dataset.size
        self.n_transforms  = np.array([i for i
          in range(int(np.ceil(np.log2(self.full_size))))
          if self.full_size / (2 ** i) >= min_size])
        self.n_blocks      = 2 ** self.n_transforms
        self.block_lengths = np.array(self.full_size / (2 ** self.n_transforms),
          np.int)

        if self.debug:
            self.debug_size  = np.column_stack(
              (self.n_transforms, self.n_blocks, self.block_lengths,
               self. n_blocks * self.block_lengths))

            print("N_TRANSFORMS     N_BLOCKS BLOCK_LENGTH   TOTAL_SIZE")
            for i in range(self.n_transforms.size):
                print("{0:>12d} {1:>12d} {2:>12d} {3:>12d}".format(
                  *self.debug_size[i]))

    def calculate_blocks(self, **kwargs):
        """
        Calculates standard error for each block transform
        """
        self.means           = np.zeros(self.n_transforms.size)
        self.stderrs         = np.zeros(self.n_transforms.size)
        self.stderrs_stddevs = np.zeros(self.n_transforms.size)

        for transform_i, block_length, n_blocks in zip(
          range(self.n_transforms.size), self.block_lengths, self.n_blocks):
            dataset_reshaped    = np.reshape(
              self.dataset[:block_length*n_blocks], (block_length, n_blocks))
            dataset_transformed = self.func(dataset_reshaped)

            mean  = np.mean(dataset_transformed)
            c_0   = (1 / block_length) * np.sum((dataset_transformed - mean)**2)
            stderr        = np.sqrt(c_0 / (block_length - 1))
            stderr_stddev = stderr / np.sqrt(2 * (block_length - 1))

            self.means[transform_i]           = mean
            self.stderrs[transform_i]         = stderr
            self.stderrs_stddevs[transform_i] = stderr_stddev

        if self.debug:
            self.debug_block = np.column_stack(
              (self.means, self.stderrs, self.stderrs_stddevs))

            print("N_TRANSFORMS     N_BLOCKS BLOCK_LENGTH   TOTAL_SIZE"
                  "        MEAN       STDERR    SE_STDDEV")
            for i in range(self.n_transforms.size):
                print("{0:>12d} {1:>12d} {2:>12d} {3:>12d}".format(
                  *self.debug_size[i]), end = "")
                print("{0:>12.5f} {1:>12.5f} {2:>12.5f}".format(
                  *self.debug_block[i]))

    def fit_curve(self, p0 = (0, 1, 4, 0.1), use_sigma = False, **kwargs):
        """
        Fits sigmoid curve to block-transformed data

        **Arguments:**
            :*use_sigma*: Include self.se_sds as standard error (not
                          recommended; fits to tightly to small number
                          of transforms
            :*kwargs*:    Passed to scipy.optimize.curve_fit
        """
        import warnings
        from scipy.optimize import curve_fit

        def sigmoid(x, a, b, c, d):
            """
                     a - b
            y = --------------- + b
                1 + (x / c) ^ d

            **Arguments:**
                :*x*: x
                :*a*: Initial y value; y(-∞) = a
                :*b*: Final y value; y(+∞) = b
                :*c*: Center of sigmoid; y(c) = (a + b) / 2
                :*d*: power

            **Returns:**
                :*y*: y(x)
            """
            return b + ((a - b) / (1 + (x / c) ** d))

        if use_sigma:
            kwargs["sigma"] = kwargs.pop("sigma", self.stderrs_stddevs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, b, c, d = curve_fit(sigmoid, self.n_transforms,
              self.stderrs, p0 = p0, **kwargs)[0]
            self.fit = sigmoid(self.n_transforms, a, b, c, d)
            self.fit_parameters = (a, b, c, d)

        if self.debug:
            print("N_TRANSFORMS     N_BLOCKS BLOCK_LENGTH   TOTAL_SIZE"
                  "        MEAN       STDERR    SE_STDDEV"
                  "         FIT")

            for i in range(self.n_transforms.size):
                print("{0:>12d} {1:>12d} {2:>12d} {3:>12d}".format(
                  *self.debug_size[i]), end = "")
                print("{0:>12.5f} {1:>12.5f} {2:>12.5f}".format(
                  *self. debug_block[i]), end = "")
                print("{0:>12.5f}".format(self.fit[i]))
            print("A = {0:<12.5f} B = {1:<12.5f} C = {2:<12.5f} D = {3:<12.5f}".format(*self.fit_parameters))

    def plot(self, outfile = "test.pdf", **kwargs):
        """
        Plots results using matplotlib

        .. todo:
            - automatically name output file if already present
        """
        import matplotlib
        from matplotlib.pyplot import figure as Figure
        from matplotlib.backends.backend_pdf import PdfPages
        matplotlib.rcParams["mathtext.default"] = "regular"
        matplotlib.rcParams["pdf.fonttype"] = "truetype"

        figure  = Figure(figsize = [6, 5])
        subplot = figure.add_subplot(111)
        subplot.set_title(self.name)
        subplot.set_xlabel("Number of Block Transformations")
        subplot.set_ylabel("$\sigma$")

        if hasattr(self, "n_transforms") and hasattr(self, "stderrs"):
            subplot.plot(self.n_transforms, self.stderrs, color = "blue")
            if hasattr(self, "stderrs_stddevs"):
                subplot.fill_between(self.n_transforms,
                  self.stderrs - 1.96 * self.stderrs_stddevs,
                  self.stderrs + 1.96 * self.stderrs_stddevs,
                  lw = 0, alpha = 0.5, color = "blue")
                if hasattr(self, "fit"):
                    subplot.plot(self.n_transforms, self.fit, color = "red")

        figure.tight_layout()
        with PdfPages(outfile) as pdf_outfile:
            figure.savefig(pdf_outfile, format = "pdf")
        print("FIGURE SAVED TO {0}".format(outfile))

if __name__ == "__main__":
    import argparse

    parser            = argparse.ArgumentParser(
      description     = __doc__,
      formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument(
      "-infile",
      type     = str,
      required = True,
      help     = "Input file; may be text file containing single column "
                 "of values, or a numpy file (.npy) containg a 1-dimensional "
                 "array")

    parser.add_argument(
      "-name",
      type     = str,
      help     = "Dataset name (default: current time)")

    parser.add_argument(
      "-outfile",
      type     = str,
      default  = "block_average.pdf",
      help     = "Output pdf file (default: %(default)s)")

    kwargs = vars(parser.parse_args())

    infile = kwargs.pop("infile")
    if infile.endswith(".npy"):
        dataset = np.load(infile)
    else:
        dataset = np.loadtxt(infile)
    block_averager = FP_Block_Averager(dataset = dataset, debug = True, **kwargs)
    block_averager.select_sizes()
    block_averager.calculate_blocks()
    block_averager.fit_curve()
    block_averager.plot(**kwargs)


