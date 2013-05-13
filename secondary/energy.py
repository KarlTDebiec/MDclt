#!/usr/bin/python
desc = """energy.py
    Functions for secondary analysis of energy
    Written by Karl Debiec on 13-05-06
    Last updated 13-05-12"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   scipy.stats import linregress
################################################## ANALYSIS FUNCTIONS ##################################################
def conservation(hdf5_file, n_cores = 1, **kwargs):
    """ Calculates energy drift """
    verbose     = kwargs.get("verbose",     False)          # Print output to terminal
    time        = hdf5_file.data["*/log"]["time"]
    energy      = hdf5_file.data["*/log"]["total"]
    temperature = hdf5_file.data["*/log"]["temperature"]
    E_line      = linregress(time, energy)
    T_line      = linregress(time, temperature)
    dtype       = np.dtype([("energy slope",      "f"), ("energy intercept",      "f"), ("energy R2",      "f"),
                            ("temperature slope", "f"), ("temperature intercept", "f"), ("temperature R2", "f")])
    data        = np.array((E_line[0], E_line[1], E_line[2] ** 2, T_line[0], T_line[1], T_line[2] ** 2), dtype)
    kwargs["data_kwargs"]   = {"chunks": False}
    if verbose:
        print "DURATION  {0:6d} ns".format(int(time[-1]))
        print "          ENERGY                 TEMPERATURE"
        print "SLOPE     {0:12.5f} kcal ns-1 {1:9.5f} K ns-1".format(float(data["energy slope"]),
                                                                     float(data["temperature slope"]))
        print "INTERCEPT {0:12.5f} kcal      {1:9.5f} K".format(float(data["energy intercept"]),
                                                                float(data["temperature intercept"]))
        print "R2        {0:12.5f}           {1:9.5f}".format(float(data["energy R2"]),
                                                              float(data["temperature R2"]))
    return  [("energy/conservation", data, kwargs),
             ("energy/conservation", {"energy slope":           "kcal ns-1",
                                      "energy intercept":       "kcal",
                                      "temperature slope":      "K ns -1",
                                      "temperature intercept":  "K",
                                      "time":                   time[-1]})]
def _check_conservation(hdf5_file, **kwargs):
    verbose     = kwargs.get("verbose",     False)
    force       = kwargs.get("force",       False)
    expected    = ["/energy/conservation"]
    hdf5_file.load("*/log", type = "table")
    if     (force
    or not (expected in hdf5_file)):
        return [(conservation, kwargs)]
    attrs       = hdf5_file.attrs("energy/conservation")
    if hdf5_file.data["*/log"]["time"][-1] != attrs["time"]:
        return [(conservation, kwargs)]
    elif verbose:
        data    = hdf5_file["energy/conservation"]
        print "DURATION  {0:6d} ns".format(int(attrs["time"]))
        print "          ENERGY                 TEMPERATURE"
        print "SLOPE     {0:12.5f} kcal ns-1 {1:9.5f} K ns-1".format(float(data["energy slope"]),
                                                                     float(data["temperature slope"]))
        print "INTERCEPT {0:12.5f} kcal      {1:9.5f} K".format(float(data["energy intercept"]),
                                                                float(data["temperature intercept"]))
        print "R2        {0:12.5f}           {1:9.5f}".format(float(data["energy R2"]),
                                                              float(data["temperature R2"]))
    return False
