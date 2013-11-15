#!/usr/bin/python
desc = """MD_toolkit.primary_x.__init__.py
    Functions for primary cross-segment analysis of MD trajectories
    Written by Karl Debiec on 13-10-30
    Last updated by Karl Debiec on 13-11-02"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, types
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def com_unwrap(hdf5_file, segments, side_length, source, destination, debug_xyz = False, **kwargs):
    inner_cutoff    = side_length / 4
    final_sign      = None
    final_offset    = None
    if debug_xyz   != False:
        outfile     = open(debug_xyz, "w")
    for segment in segments:
        com = hdf5_file["{0}/{1}".format(segment, source)]
        sign                = np.sign(com)
        delta_sign          = np.zeros(sign.shape)
        delta_sign[1:]      = sign[1:] - sign[:-1]
        if final_sign is not None:
            delta_sign[0]   = sign[0] - final_sign
        offset              = np.zeros(sign.shape)
        if final_offset is None:
            final_offset    = np.zeros(offset.shape[1:])
        else:
             offset[:]     += final_offset
        for i, j, k in np.column_stack(np.where(delta_sign == -2)):
            if np.abs(com[i,j,k]) < inner_cutoff:
                continue
            offset[i:,j,k] += side_length
        for i, j, k in np.column_stack(np.where(delta_sign == 2)):
            if np.abs(com[i,j,k]) < inner_cutoff:
                continue
            offset[i:,j,k] -= side_length
        final_sign          = sign[-1]
        final_offset        = offset[-1]
        final               = com + offset
        yield (segment + "/" + destination, final)
        if debug_xyz:
            for frame in final:
                outfile.write("{0}\n".format(final.shape[1]))
                outfile.write("test\n")
                for atom in frame:
                    outfile.write("  C {0:>16.6f}{1:>16.6f}{2:>16.6f}\n".format(*atom))
        del com, sign, offset, delta_sign, final
    if debug_xyz:
        outfile.close()
def _check_com_unwrap(hdf5_file, force = False, **kwargs):
    kwargs["hdf5_file"]   = hdf5_file
    kwargs["source"]      = source      = kwargs.get("source",      "com")
    kwargs["destination"] = destination = kwargs.get("destination", "com_unwrap")
    kwargs["segments"]    = segments    = [s for s in kwargs.get("segments", [])
                                             if  s.topology   and os.path.isfile(s.topology)
                                             and s.trajectory and os.path.isfile(s.trajectory)
                                             and "{0}/{1}".format(s, source) in hdf5_file]
    kwargs["side_length"] = kwargs.get("side_length", hdf5_file["{0}/log".format(segments[0])]["volume"][0] ** (1.0/3.0))
    expected              = [s + "/" + destination for s in segments]
    if (force
    or  not expected in hdf5_file):
        return [(com_unwrap, kwargs)]
    else:
        return False


