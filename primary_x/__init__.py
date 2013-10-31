#!/usr/bin/python
desc = """__init__.py
    Functions for primary cross-segment analysis of MD trajectories
    Written by Karl Debiec on 13-10-30
    Last updated 13-10-31"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, types
import numpy as np
import mdtraj
################################################## ANALYSIS FUNCTIONS ##################################################
def com_unwrap(segments, side_length, destination, **kwargs):
    inner_cutoff    = side_length / 4
    final_sign      = None
    final_offset    = None
#    with open("test.xyz", "w") as outfile:
    for segment in segments:
        sign            = np.sign(segment.com)
        delta_sign      = np.zeros(sign.shape)
        delta_sign[1:]  = sign[1:] - sign[:-1]
        if final_sign is not None:  delta_sign[0] = sign[0] - final_sign
        offset  = np.zeros(sign.shape)
        if final_offset is None:    final_offset    = np.zeros(offset.shape[1:])
        else:                       offset[:]      += final_offset
        for i, j, k in np.column_stack(np.where(delta_sign == -2)):
            if np.abs(segment.com[i,j,k]) < inner_cutoff: continue
            offset[i:,j,k] += side_length
        for i, j, k in np.column_stack(np.where(delta_sign == 2)):
            if np.abs(segment.com[i,j,k]) < inner_cutoff: continue
            offset[i:,j,k] -= side_length
        final_sign      = sign[-1]
        final_offset    = offset[-1]
        final           = segment.com + offset
        yield (segment + "/" + destination, segment.com + offset)
#            for frame in final:
#                outfile.write("{0}\n".format(final.shape[1]))
#                outfile.write("test\n")
#                for atom in frame:
#                    outfile.write("  C {0:>16.6f}{1:>16.6f}{2:>16.6f}\n".format(*atom))
def _check_com_unwrap(hdf5_file, source = "com", force = False, **kwargs):
    segments    = kwargs.get("segments", [])
    kwargs["destination"] = destination = kwargs.get("destination", "com_unwrap")
    expected    = [s + "/" + destination for s in segments if  s.topology   and os.path.isfile(s.topology)
                                                           and s.trajectory and os.path.isfile(s.trajectory)]
    if (force
    or  not expected in hdf5_file):
        for segment in segments:
            segment.com = hdf5_file["{0}/{1}".format(segment, source)]
        return [(com_unwrap, kwargs)]
    else:
        return False


