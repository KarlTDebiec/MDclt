#!/usr/bin/python

desc = """secondary_rotation.py
    Functions for secondary analysis of rotational motion
    Written by Karl Debiec on 13-02-08
    Last updated 13-02-13"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import time as time_module
import numpy as np
from   multiprocessing import Pool
from   scipy.optimize import curve_fit, fmin
from   scipy.linalg import svd, diagsvd
from   hdf5_functions import shape_default, process_default, postprocess_default
################################################## INTERNAL FUNCTIONS ##################################################
def _random_unit_vectors(n_unit_vectors):
    thetas              = np.random.rand(n_unit_vectors) * np.pi
    phis                = np.random.rand(n_unit_vectors) * np.pi * 2
    unit_vectors        = np.zeros((n_unit_vectors, 3))
    unit_vectors[:,0]   = np.sin(thetas) * np.cos(phis)
    unit_vectors[:,1]   = np.sin(thetas) * np.sin(phis)
    unit_vectors[:,2]   = np.cos(thetas)
    return unit_vectors
def _P2(x):      return 0.5 * (3 * x ** 2 - 1)
def _dot(a, b):  return np.sum(a * b, axis = 1)
def _acf(vector, rotmat, tau_indexes):
    rotated = np.array([np.dot(frame, np.transpose(vector)) for frame in rotmat])
    length  = rotated.shape[0]
    acf     = np.zeros(tau_indexes.shape, dtype = np.float)
    for i, tau in enumerate(tau_indexes):
        acf[i]  = np.mean(_P2(_dot(rotated[:length - tau], rotated[tau:])))
    return acf
def _anisotropic(arguments):
    def model_function(tau_l, tau_finite, F_l): return (tau_l - (F_l / (1 - np.exp(-1 * tau_finite / tau_l)))) ** 2
    i, vector, rotmat, tau_indexes, tau_times, tau_finite, dt  = arguments
    acf     = _acf(vector, rotmat, tau_indexes)
    F_l     = np.sum(acf * dt)
    tau_l   = fmin(func = model_function, x0 = 1, args = (tau_finite, F_l), disp = False)
    return i, tau_l
def _A(v):              return np.column_stack((v ** 2, 2 * v[:,0] * v[:,1], 2 * v[:,1] * v[:,2], 2 * v[:,0] * v[:,2]))
def _Q_1D_to_2D(Q):     return np.array([[Q[0], Q[3], Q[5]],[Q[3], Q[1], Q[4]],[Q[5], Q[4], Q[2]]])
def _Q_1D_to_D_2D(Q):
    Qxx, Qyy, Qzz, Qxy, Qyz, Qxz    = Q
    Dxy = -2 * Qxy
    Dyz = -2 * Qyz
    Dxz = -2 * Qxz    
    Dxx = -Qxx + Qyy + Qzz
    Dyy =  Qxx - Qyy + Qyy
    Dzz =  Qxx + Qyy - Qzz
    return np.array([[Dxx,Dxy,Dxz],[Dxy,Dyy,Dyz],[Dxz,Dyz,Dzz]])
################################################ PRIMARY DATA FUNCTIONS ################################################
def shape_rotmat(shapes):           return np.array([np.sum(shapes[:, 0]), 3, 3])
def process_rotmat(new_data):       return np.reshape(new_data, (new_data.shape[0], 3, 3))
################################################## ANALYSIS FUNCTIONS ##################################################
def anisotropic(primary_data, arguments, n_cores = 1):
    """ Calculates the rotational diffusion tensor of <domain> via the autocorrelation functions of <n_vectors> random
        unit vectors as they are rotated by <rotmat>. The autocorrelation function is integrated to time cutoff(s)
        <tau_finites> and tail corrected to obtain the local correlation time. From these correlation times the 3x3
        rotational diffusion tensor is calculated using singular value decomposition. Input data may optionally be
        reduced by <index_slice>. Follows the protocol of  Wong, V., and Case, D. A. J Phys Chem B. 2008. 112.
        6013-6024. Error is estimated by repeating calculations for halves and quarters of the dataset. """
    domain      = arguments['domain']
    tau_finites = np.array(arguments['tau_finite']) if 'tau_finite'  in arguments else np.array([5])
    n_vectors   = arguments['n_vectors']            if 'n_vectors'   in arguments else 1000
    index_slice = arguments['index_slice']          if 'index_slice' in arguments else 1
    time_full   = primary_data['*/time'][::index_slice]
    rotmat_full = primary_data['*/rotmat_' + domain][::index_slice]
    vectors     = _random_unit_vectors(n_vectors)
    size        = time_full.size
    dt          = time_full[1] - time_full[0]
    if index_slice != 1:
        domain += "/slice_{0}/".format(index_slice)
    new_data    = [("/rotation_" + domain,                 {'n_vectors': n_vectors, 'dt': dt}),
                   ("/rotation_" + domain + "/tau_finite", tau_finites),
                   ("/rotation_" + domain + "/tau_finite", {'units': 'ns'})]
    splits      = [("full",      "[:]"),         ("half/1",    "[:size/2]"),       ("half/2",    "[size/2:]"),
                   ("quarter/1", "[:size/4]"),   ("quarter/2", "[size/4:size/2]"), ("quarter/3", "[size/2:3*size/4]"),
                   ("quarter/4", "[3*size/4:]")]
    for path, index in splits:
        print path, eval("time_full{0}".format(index))
        time    = eval("time_full{0}".format(index))
        rotmat  = eval("rotmat_full{0}".format(index))
        Ds      = np.zeros((tau_finites.size, 3, 3))
        for i, tau_finite in enumerate(np.array(tau_finites)):
            print "{0} ".format(tau_finite),
            tau_indexes = np.array(range(0, int(tau_finite / dt) + 1))
            tau_times   = np.arange(0, tau_finite + dt, dt)
            tau_ls      = np.zeros(vectors.shape[0])
            arguments   = [(j, vector, rotmat, tau_indexes, tau_times, tau_finite, dt)
                            for j, vector in enumerate(vectors)]
            pool        = Pool(n_cores)
            for k, result in enumerate(pool.imap_unordered(_anisotropic, arguments)):
                j, tau_l    = result
                tau_ls[j]   = tau_l
            pool.close()
            pool.join()
            """ Note: The presence of * 2 in the line below is not understood; omitting it yields results 1/2 of
                those published by Wong and Case. The issue is likely at one of the following steps:
                    d_local is half  what is should be
                    tau_l   is twice what it should be
                    F_l     is twice what it should be
                    acf     is twice what it should be """
            d_locals_t  = np.transpose(np.mat(1 / (6 * tau_ls))) * 2
            A           = _A(vectors)
            Q           = np.squeeze(np.array(np.mat(np.linalg.pinv(A)) * np.mat(d_locals_t)))
            D           =  _Q_1D_to_D_2D(Q)
            _, P        = np.linalg.eig(D)
            Ds[i]       = np.mat(np.linalg.inv(P)) * np.mat(D) * np.mat(P)
        print Ds
        Dx, Dy, Dz  = sorted([Ds[-1,0,0], Ds[-1,1,1], Ds[-1,2,2]])
        D_average   =  (Dx + Dy + Dz) / 3
        anisotropy  = (2 * Dz) / (Dx + Dy)
        rhombicity  = (1.5 * (Dy - Dx)) / (Dz - 0.5 * (Dx + Dy))
        print
        print "D_AVERAGE ", D_average
        print "ANISOTROPY", anisotropy
        print "RHOMBICITY", rhombicity
        new_data   += [("/rotation_" + domain + "/" + path + "/D", Ds),
                       ("/rotation_" + domain + "/" + path + "/D", {'units': 'ns-1',
                                                                    'time':  time.size * dt,})]
    print new_data
    return new_data
def path_anisotropic(arguments):
    return  [('*/time',                          (shape_default, process_default, postprocess_default)),
             ('*/rotmat_' + arguments['domain'], (shape_rotmat,  process_rotmat,  postprocess_default))]
def check_anisotropic(arguments):
    return [(anisotropic, arguments)]


