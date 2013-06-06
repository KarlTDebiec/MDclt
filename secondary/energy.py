#!/usr/bin/python
desc = """energy.py
    Functions for secondary analysis of energy
    Written by Karl Debiec on 13-05-06
    Last updated 13-06-04"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import numpy as np
from   scipy.stats import linregress
################################################## ANALYSIS FUNCTIONS ##################################################
def conservation(hdf5_file, verbose = False, n_cores = 1, **kwargs):
    """ Calculates energy drift """
    time        = hdf5_file.data["*/log"]["time"]
    E_line      = linregress(time, hdf5_file.data["*/log"]["total"])
    T_line      = linregress(time, hdf5_file.data["*/log"]["temperature"])
    dtype       = np.dtype([("energy slope",      "f4"), ("energy intercept",      "f4"), ("energy R2",      "f4"),
                            ("temperature slope", "f4"), ("temperature intercept", "f4"), ("temperature R2", "f4")])
    data        = np.array((E_line[0], E_line[1], E_line[2] ** 2, T_line[0], T_line[1], T_line[2] ** 2), dtype)
    attrs       = {"energy slope units":      "kcal mol-1 ns-1", "energy intercept units":       "kcal mol-1",
                   "temperature slope units": "K ns -1",         "temperature intercept units":  "K",
                   "time":                    time[-1]}
    kwargs["data_kwargs"]   = {"chunks": False}
    if verbose:
        print "DURATION  {0:6d} ns".format(int(attrs["time"]))
        print "          ENERGY                       TEMPERATURE"
        print "SLOPE     {0:12.4f} kcal mol-1 ns-1 {1:8.4f} K ns-1".format(float(data["energy slope"]),
                                                                           float(data["temperature slope"]))
        print "INTERCEPT {0:12.4f} kcal mol-1      {1:8.4f} K".format(float(data["energy intercept"]),
                                                                      float(data["temperature intercept"]))
        print "R2        {0:12.4f}                 {1:8.4f}".format(float(data["energy R2"]),
                                                                    float(data["temperature R2"]))
    return  [("energy/conservation", data, kwargs),
             ("energy/conservation", attrs)]
def _check_conservation(hdf5_file, force = False, **kwargs):
    verbose = kwargs.get("verbose",     False)
    hdf5_file.load("*/log", type = "table")
    if    (force
    or not "/energy/conservation" in hdf5_file):
        return [(conservation, kwargs)]
    attrs   = hdf5_file.attrs("energy/conservation")
    if hdf5_file.data["*/log"]["time"][-1] != attrs["time"]:
        return [(conservation, kwargs)]
    elif verbose:
        data    = hdf5_file["energy/conservation"]
        print "DURATION  {0:6d} ns".format(int(attrs["time"]))
        print "          ENERGY                       TEMPERATURE"
        print "SLOPE     {0:12.4f} kcal mol-1 ns-1 {1:8.4f} K ns-1".format(float(data["energy slope"]),
                                                                           float(data["temperature slope"]))
        print "INTERCEPT {0:12.4f} kcal mol-1      {1:8.4f} K".format(float(data["energy intercept"]),
                                                                      float(data["temperature intercept"]))
        print "R2        {0:12.4f}                 {1:8.4f}".format(float(data["energy R2"]),
                                                                    float(data["temperature R2"]))
    return False
