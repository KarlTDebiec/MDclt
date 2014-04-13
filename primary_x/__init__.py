#!/usr/bin/python
#MD_toolkit.primary_x.__init__.py
#Written by Karl Debiec on 13-10-30
#Last updated by Karl Debiec on 14-03-25
"""
Functions for primary cross-segment analysis of MD trajectories
"""
################################################# MODULES AND SETTINGS #################################################
import os, sys, types
import numpy as np
################################################## ANALYSIS FUNCTIONS ##################################################
def com_unwrap(hdf5_file, segments, side_length, source, destination, shift_origin = False, debug_xyz = False, **kwargs):
    """
    Unwraps center of mass coordinates
    Currently supports only cubic box
    """

    # Initialzie variables
    inner_cutoff    = side_length / 4                       # Changes in sign within cutoff assumed to remain in box
    final_sign      = None                                  # Sign of last frame of previous segment
    final_offset    = None                                  # Offset of last frame of previous segment
    if debug_xyz   != False:                                # Optionally save coordinates to xyz
        print "Outputting unwrapped coordinates to '{0}'".format(debug_xyz)
        outfile     = open(debug_xyz, "w")

    # Loop over segments
    for segment in segments:

        # Load an organize source data
        com = hdf5_file["{0}/{1}".format(segment, source)]
        if len(com.shape) == 2:                             # Smoothly support data in form of either (frame, xyz)
            com = np.expand_dims(com, axis = 1)             # or (frame, selection, xyz)
        attrs =  hdf5_file.attrs("{0}/{1}".format(segment, source))
        if shift_origin:                                    # Some packages store coordinates over range of 0...X,
            com -= side_length / 2                          # this adjusts to -X/2...X/2

        # Identify jumps and calculate offsets
        sign                = np.sign(com)
        delta_sign          = np.zeros(sign.shape)
        delta_sign[1:]      = sign[1:] - sign[:-1]
        if final_sign is not None:                          # Adjust first frame based on last frame of previous segment
            delta_sign[0]   = sign[0] - final_sign
        offset              = np.zeros(sign.shape)
        if final_offset is None:                            # Adjust all frames based on last frame of previous segment
            final_offset    = np.zeros(offset.shape[1:])
        else:
             offset[:]     += final_offset
        for i, j, k in np.column_stack(np.where(delta_sign == -2)):             # Selection has crossed from + to -
            if np.abs(com[i,j,k]) < inner_cutoff:                               # Cross was within box
                continue
            offset[i:,j,k] += side_length                                       # Cross was across box
        for i, j, k in np.column_stack(np.where(delta_sign == 2)):              # Selection has crossed from - to +
            if np.abs(com[i,j,k]) < inner_cutoff:                               # Cross was within box
                continue
            offset[i:,j,k] -= side_length                                       # Cross was across box
        final_sign          = sign[-1]
        final_offset        = offset[-1]
        final               = com + offset

        # Yield results, cleanup, and continue to next segment
        if debug_xyz:
            for frame in final:
                outfile.write("{0}\n".format(final.shape[1]))
                outfile.write("test\n")
                for atom in frame:
                    outfile.write("  C {0:>16.6f}{1:>16.6f}{2:>16.6f}\n".format(*atom))
            test = final[1:] - final[:-1]
            print np.min(test), np.mean(test), np.max(test)
        yield (segment + "/" + destination, final)
        yield (segment + "/" + destination, attrs)
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
    kwargs["side_length"] = kwargs.get("side_length", None)
    if  kwargs["side_length"] is None:
        kwargs["side_length"] = hdf5_file["{0}/log".format(segments[0])]["volume"][0] ** (1.0/3.0)
    expected              = [s + "/" + destination for s in segments]
    if (force
    or  not expected in hdf5_file):
        return [(com_unwrap, kwargs)]
    else:
        return False


