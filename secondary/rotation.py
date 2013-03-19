#!/usr/bin/python

desc = """secondary_rotation.py
    Functions for secondary analysis of rotational motion
    Written by Karl Debiec on 13-02-08
    Last updated 13-03-14"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys
import time as time_module
import numpy as np
from   multiprocessing   import Pool
from   scipy.optimize    import fmin
from   scipy.integrate   import trapz
from   cython_functions  import _cy_acf
################################################## INTERNAL FUNCTIONS ##################################################
def _random_unit_vectors(n_vectors = 1):
    thetas          = np.random.rand(n_vectors) * np.pi
    phis            = np.random.rand(n_vectors) * np.pi * 2
    vectors         = np.zeros((n_vectors, 3))
    vectors[:,0]    = np.sin(thetas) * np.cos(phis)
    vectors[:,1]    = np.sin(thetas) * np.sin(phis)
    vectors[:,2]    = np.cos(thetas)
    if n_vectors == 1:  return vectors[0]
    else:               return vectors
def _calc_vector_acf(arguments):
    def model_function(tau_l, tau_finite, F_l): return (tau_l - (F_l / (1 - np.exp(-1 * tau_finite / tau_l)))) ** 2
    i, vector, rotmat, tau_finite, dt  = arguments
    index_finite    = int(tau_finite / dt)
    acf             = _cy_acf(vector, rotmat, index_finite)
    F_l             = trapz(acf, dx = dt)
    tau_l           = fmin(func = model_function, x0 = 1, args = (tau_finite, F_l), disp = False)
    return i, tau_l
def _A(v):      return np.column_stack((v ** 2, 2 * v[:,0] * v[:,1], 2 * v[:,1] * v[:,2], 2 * v[:,0] * v[:,2]))
def _Q_1D_to_Q_2D(Qxx, Qyy, Qzz, Qxy, Qyz, Qxz): return np.array([[Qxx,Qxy,Qxz],[Qxy,Qyy,Qyz],[Qxz,Qyz,Qzz]])
def _Q_diag_to_D_2D(Qxx, Qyy, Qzz):
    Dxx = -Qxx + Qyy + Qzz
    Dyy =  Qxx - Qyy + Qzz
    Dzz =  Qxx + Qyy - Qzz
    return np.array([[Dxx,0,0],[0,Dyy,0],[0,0,Dzz]])
################################################ PRIMARY DATA FUNCTIONS ################################################
def _shape_rotmat(shapes):      return np.array([np.sum(shapes[:, 0]), 3, 3])
def _process_rotmat(new_data):  return np.reshape(new_data, (new_data.shape[0], 3, 3))
################################################## ANALYSIS FUNCTIONS ##################################################
def diffusion_tensor(hdf5_file, n_cores = 1, **kwargs):
    """ Calculates the rotational diffusion tensor of <domain> as it rotates by |rotmat| over a trajectory of length
        |time|. <n_vectors> random unit vectors are generated and their autocorrelation functions calculated, integrated
        to time cutoff(s) <tau_finites>, and tail corrected to obtain local diffusion coefficients. From these local
        coefficients the overall rotational diffusion tensor is calculated using singular value decomposition. Primary
        data may be downsampled by <index_slice> for faster calculation. <convergence> may optionally be estimated by
        repeating calculations for halves and quarters of the dataset. Follows the protocol of Wong, V., and Case, D. A.
        J Phys Chem B. 2008. 112. 6013-6024. """
    output_path = kwargs.get("domain",      "")
    tau_finites = kwargs.get("tau_finite",  np.array([1.0]))
    n_vectors   = kwargs.get("n_vectors",   1000)
    index_slice = kwargs.get("index_slice", 1)
    convergence = kwargs.get("convergence", False)
    time_full   = hdf5_file.data['*/time'][::index_slice]
    rotmat_full = hdf5_file.data['*/rotmat_' + output_path][::index_slice]
    vectors     = _random_unit_vectors(n_vectors)
    size        = time_full.size
    dt          = time_full[1] - time_full[0]

    expected    = {"ubiquitin": [0.041, 0.046, 0.051, 0.046, 1.180, 1.080],
                   "GB3":       [0.050, 0.060, 0.100, 0.070, 1.810, 0.330],
                   "lysozyme":  [0.025, 0.033, 0.038, 0.032, 1.310, 1.240]}[output_path]

    if index_slice != 1:
        output_path    += "/slice_{0}/".format(index_slice)
    new_data    = [("/rotation_" + output_path + "/tau_finite", tau_finites),
                   ("/rotation_" + output_path + "/tau_finite", {'units': 'ns'}),
                   ("/rotation_" + output_path,                 {'n_vectors': n_vectors, 'dt': dt})]
    if convergence:
        splits  = [("full",      "[:]"),               ("half/1",    "[:size/2]"),       ("half/2",    "[size/2:]"),
                   ("quarter/1", "[:size/4]"),         ("quarter/2", "[size/4:size/2]"),
                   ("quarter/3", "[size/2:3*size/4]"), ("quarter/4", "[3*size/4:]")]
    else:
        splits  = [("full",      "[:]")]
    for path, index in splits:
        time    = eval("time_full{0}".format(index))
        rotmat  = eval("rotmat_full{0}".format(index))
        Ds      = np.zeros((tau_finites.size, 3, 3))

        for i, tau_finite in enumerate(tau_finites):
            tau_ls      = np.zeros(n_vectors)

            arguments   = [(j, vector, rotmat, tau_finite, dt) for j, vector in enumerate(vectors)]
            pool        = Pool(n_cores)
            for k, result in enumerate(pool.imap_unordered(_calc_vector_acf, arguments)):
                j, tau_l    = result
                tau_ls[j]   = tau_l
            pool.close()
            pool.join()

            d_locals    = np.transpose(np.mat(1 / (6 * tau_ls)))
            A           = _A(vectors)
            Q           = np.squeeze(np.array(np.mat(np.linalg.pinv(A)) * np.mat(d_locals)))
            Q           = _Q_1D_to_Q_2D(*Q)
            _, P        = np.linalg.eig(Q)
            Q           = np.mat(np.linalg.inv(P)) * np.mat(Q) * np.mat(P)
            D           = _Q_diag_to_D_2D(*[Q[0,0], Q[1,1], Q[2,2]])
            Dx, Dy, Dz  = sorted([D[0,0], D[1,1], D[2,2]])
            D_average   = (Dx + Dy + Dz)    / 3
            anisotropy  = (2 * Dz)          / (Dx + Dy)
            rhombicity  = (1.5 * (Dy - Dx)) / (Dz - 0.5 * (Dx + Dy))
            print "           {0:<5} {1:<5} {2:<2}".format("Pub.", "Calc.", "%")
            print "Dx         {0:6.4f} {1:6.4f} {2:3.0f}".format(expected[0],Dx,        100 * Dx         / expected[0])
            print "Dy         {0:6.4f} {1:6.4f} {2:3.0f}".format(expected[1],Dy,        100 * Dy         / expected[1])
            print "Dz         {0:6.4f} {1:6.4f} {2:3.0f}".format(expected[2],Dz,        100 * Dz         / expected[2])
            print "D_AVERAGE  {0:6.4f} {1:6.4f} {2:3.0f}".format(expected[3],D_average, 100 * D_average  / expected[3])
            print "ANISOTROPY {0:6.4f} {1:6.4f} {2:3.0f}".format(expected[4],anisotropy,100 * anisotropy / expected[4])
            print "RHOMBICITY {0:6.4f} {1:6.4f} {2:3.0f}".format(expected[5],rhombicity,100 * rhombicity / expected[5])
            print
        new_data   += [("/rotation_" + output_path + "/" + path + "/D", Ds),
                       ("/rotation_" + output_path + "/" + path + "/D", {'units': 'ns-1',
                                                                         'time':  time.size * dt,})]
    return None
    return new_data
def _check_diffusion_tensor(hdf5_file, **kwargs):
    domain  = kwargs.get("domain",    "")
    hdf5_file.load("*/time")
    hdf5_file.load("*/rotmat_" + domain, shaper = _shape_rotmat,  processor = _process_rotmat)
    return [(diffusion_tensor, kwargs)]


