#!/usr/bin/python

########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
from   __future__ import division
import os, sys
import numpy as np

cimport numpy as np
cimport cython
from    libc.math cimport sqrt

FLOAT64 = np.float64
LONG    = np.long
INT8    = np.int8
ctypedef  np.float64_t  FLOAT64_t
ctypedef  np.long_t     LONG_t
ctypedef  np.int8_t     INT8_t

################################################### INLINE FUNCTIONS ###################################################
cdef inline double _apply_pbc(double d, double maxcut, double mincut, double length) nogil:
    if   d > maxcut:    return d - length
    elif d < mincut:    return d + length
    else:               return d
cdef inline double _P2(double x) nogil:
    return 0.5 * (3.0*x*x - 1.0)
###################################################### FUNCTIONS #######################################################
@cython.boundscheck(False)
@cython.wraparound(False)
def _cy_distance_pbc(np.ndarray[FLOAT64_t, ndim = 2] group1_crd,
                    np.ndarray[FLOAT64_t, ndim = 2] group2_crd,
                    double                          length):
    """ Calculates distance between coordinate sets <group1_crd> and <group2_crd> within a cubic pbc box of <length> """
    cdef np.ndarray[FLOAT64_t, ndim = 2] distance   = np.zeros((group1_crd.shape[0],group2_crd.shape[0]),dtype=FLOAT64)
    cdef np.ndarray[FLOAT64_t, ndim = 1] d_xyz      = np.zeros((3), dtype = FLOAT64)
    cdef unsigned int                    A1_i, A2_i
    cdef double                          maxcut     =       length / 2.
    cdef double                          mincut     = -1. * length / 2.
    with nogil:
        for A1_i in range(group1_crd.shape[0]):
            for A2_i in range(group2_crd.shape[0]):
                d_xyz[0] = _apply_pbc(group1_crd[A1_i,0] - group2_crd[A2_i,0], maxcut, mincut, length)
                d_xyz[1] = _apply_pbc(group1_crd[A1_i,1] - group2_crd[A2_i,1], maxcut, mincut, length)
                d_xyz[2] = _apply_pbc(group1_crd[A1_i,2] - group2_crd[A2_i,2], maxcut, mincut, length)
                distance[A1_i, A2_i] = sqrt(d_xyz[0]*d_xyz[0] + d_xyz[1]*d_xyz[1] + d_xyz[2]*d_xyz[2])
    return distance


@cython.boundscheck(False)
@cython.wraparound(False)
def _cy_contact(np.ndarray[FLOAT64_t, ndim = 2] crd,
                np.ndarray[LONG_t,    ndim = 2] atomcounts,
                np.ndarray[LONG_t,    ndim = 2] indexes):
    """ Calculates inter-residue contacts from coordinates <crd>, atom indexes <atomcounts>, and 2D to 1D conversion
        matrix <indexes> """
    cdef unsigned int                       n_res       = atomcounts.shape[0]
    cdef unsigned int                       R1_i, R2_i, A1_i, A2_i
    cdef float                              distance
    cdef unsigned int                       breaking
    cdef np.ndarray[FLOAT64_t, ndim = 1]    d_xyz       = np.zeros((3), dtype = FLOAT64)
    cdef np.ndarray[INT8_t,    ndim = 1]    contact     = np.zeros(((n_res * n_res - n_res) / 2), dtype = INT8)
    with nogil:
        for R1_i in range(n_res):
            for R2_i in range(R1_i + 1, n_res):
                breaking    = 0
                for A1_i in range(atomcounts[R1_i,0], atomcounts[R1_i,1]):
                    if breaking:    break
                    for A2_i in range(atomcounts[R2_i,0], atomcounts[R2_i,1]):
                        d_xyz[0]    = crd[0,A1_i] - crd[0,A2_i]
                        d_xyz[1]    = crd[1,A1_i] - crd[1,A2_i]
                        d_xyz[2]    = crd[2,A1_i] - crd[2,A2_i]
                        distance    = d_xyz[0]*d_xyz[0] + d_xyz[1]*d_xyz[1] + d_xyz[2]*d_xyz[2]
                        if distance > 1000:
                            breaking = 1
                            break
                        if distance <= 30.25:
                            contact[indexes[R1_i, R2_i]]    = 1
                            breaking                        = 1
                            break
    return contact


@cython.boundscheck(False)
@cython.wraparound(False)
def _cy_acf(np.ndarray[FLOAT64_t, ndim = 1] vector,
            np.ndarray[FLOAT64_t, ndim = 3] rotmat,
            unsigned int                    index_finite):
    """ Calculates autocorrelation function of <vector> as it is rotated by <rotmat> out to <index_finite> """
    cdef unsigned int                       n_frames    = rotmat.shape[0]
    cdef unsigned int                       i, j
    cdef np.ndarray[FLOAT64_t, ndim = 2]    rotated     = np.zeros((n_frames, 3))
    cdef np.ndarray[FLOAT64_t, ndim = 1]    acf         = np.zeros(index_finite + 1)
    with nogil:
        for i in xrange(n_frames):
            rotated[i,0]    = rotmat[i,0,0]*vector[0] + rotmat[i,0,1]*vector[1] + rotmat[i,0,2]*vector[2]
            rotated[i,1]    = rotmat[i,1,0]*vector[0] + rotmat[i,1,1]*vector[1] + rotmat[i,1,2]*vector[2]
            rotated[i,2]    = rotmat[i,2,0]*vector[0] + rotmat[i,2,1]*vector[1] + rotmat[i,2,2]*vector[2]
        for i in xrange(index_finite + 1):
            for j in xrange(n_frames - i):
                acf[i] += _P2(rotated[j,0]*rotated[j+i,0] + rotated[j,1]*rotated[j+i,1] + rotated[j,2]*rotated[j+i,2])
            acf[i]     /= n_frames - i
    return acf


