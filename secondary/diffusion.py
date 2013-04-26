#!/usr/bin/python

desc = """diffusion.py
    Functions for secondary analysis of diffusion
    Written by Karl Debiec on 13-02-08
    Last updated 13-04-25"""
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
def rotation(hdf5_file, n_cores = 1, **kwargs):
    """ Calculates the rotational diffusion tensor of <domain> as it rotates by |rotmat| over a trajectory of length
        |time|. <n_vectors> random unit vectors are generated, rotated by |rotmat|,  and their autocorrelation functions
        calculated, integrated to time cutoff(s) <tau_finites>, and tail corrected to obtain local rotational diffusion
        coefficients. From these local coefficients the overall rotational diffusion tensor is calculated using singular
        value decomposition. Primary data may optionally be downsampled by <index_slice> for faster calculation.
        <convergence> may optionally be estimated by repeating calculations for halves and quarters of the dataset.
        Follows the protocol of Wong, V., and Case, D. A. J Phys Chem B. 2008. 112. 6013-6024. """
    domain      = kwargs.get("domain",      "")                         # Domain name
    index_slice = kwargs.get("index_slice", 1)                          # Downsample trajectory (use every Nth frame)
    tau_finites = kwargs.get("tau_finite",  np.array([1.0]))            # Finite cutoff for autocorrelation integration
    n_vectors   = kwargs.get("n_vectors",   1000)                       # Number of unit vectors
    control     = kwargs.get("control",     float("nan"))               # Control value for comparison
    convergence = kwargs.get("convergence", False)                      # Calculate for halves, quarters for convergence
    verbose     = kwargs.get("verbose",     False)                      # Print output to terminal

    time_full   = hdf5_file.data['*/time'][::index_slice]
    rotmat_full = hdf5_file.data['*/rotmat_' + domain][::index_slice]
    size        = time_full.size
    dt          = time_full[1] - time_full[0]
    vectors     = _random_unit_vectors(n_vectors)

    if index_slice == 1: output_path = "/diffusion/rotation_{0}/".format(domain)
    else:                output_path = "/diffusion/rotation_{0}/slice_{1:d}/".format(domain, int(index_slice))
    if convergence:      splits      = [("full",      "[:]"),
                                        ("half/1",    "[:size/2]"),         ("half/2",    "[size/2:]"),
                                        ("quarter/1", "[:size/4]"),         ("quarter/2", "[size/4:size/2]"),
                                        ("quarter/3", "[size/2:3*size/4]"), ("quarter/4", "[3*size/4:]")]
    else:                splits      = [("full",      "[:]")]
    
    new_data    = [(output_path + "/tau_finite", tau_finites),
                   (output_path + "/tau_finite", {"units":      "ns"}),
                   (output_path,                 {"dt":         dt,
                                                  "time":       time_full[-1],
                                                  "n_vectors":  n_vectors})]
    for path, index in splits:
        time    = eval("time_full{0}".format(index))
        rotmat  = eval("rotmat_full{0}".format(index))
        Ds      = np.zeros((tau_finites.size, 3))
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
            Ds[i]       = [Dx, Dy, Dz]

            if verbose:
                print "DURATION  {0:5d} ns TAU_F {1:5.1f} ns".format(int(time.size * dt), tau_finite)
                if np.all(np.isnan(control)):
                    print "Dx         {0:6.4f}".format(Ds[i,0])
                    print "Dy         {0:6.4f}".format(Ds[i,1])
                    print "Dz         {0:6.4f}".format(Ds[i,2])
                    print "D_AVERAGE  {0:6.4f}".format(D_average)
                    print "ANISOTROPY {0:6.4f}".format(anisotropy)
                    print "RHOMBICITY {0:6.4f}".format(rhombicity)
                else:
                    print "           Pub.   Calc.  %"
                    print "Dx         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[0],Ds[i,0],    100 * Ds[i,0]    / control[0])
                    print "Dy         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[1],Ds[i,0],    100 * Ds[i,1]    / control[1])
                    print "Dz         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[2],Ds[i,0],    100 * Ds[i,2]    / control[2])
                    print "D_AVERAGE  {0:6.4f} {1:6.4f} {2:3.0f}".format(control[3],D_average,  100 * D_average  / control[3])
                    print "ANISOTROPY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[4],anisotropy, 100 * anisotropy / control[4])
                    print "RHOMBICITY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[5],rhombicity, 100 * rhombicity / control[5])
        new_data   += [(output_path + "/" + path + "/D", Ds),
                       (output_path + "/" + path + "/D", {"units": "ns-1",
                                                          "time":  time.size * dt,})]
    return new_data
def _check_rotation(hdf5_file, **kwargs):
    domain      = kwargs.get("domain",    "")
    index_slice = kwargs.get("index_slice", 1)
    tau_finites = kwargs.get("tau_finite",  np.array([1.0]))
    n_vectors   = kwargs.get("n_vectors",   1000)
    control     = kwargs.get("control",     float("nan"))
    convergence = kwargs.get("convergence", False)
    verbose     = kwargs.get("verbose",     False)
    force       = kwargs.get("force",       False)

    if index_slice == 1: output_path = "/diffusion/rotation_{0}/".format(domain)
    else:                output_path = "/diffusion/rotation_{0}/slice_{1:d}/".format(domain, int(index_slice))
    if convergence:      splits      = ["full","half/1","half/2","quarter/1","quarter/2","quarter/3","quarter/4"]
    else:                splits      = ["full"]
    expected    = [output_path + "/tau_finite"] 
    expected   += [output_path + "/" + path + "/D" for path in splits]

    hdf5_file.load("*/time")
    hdf5_file.load("*/rotmat_" + domain, shaper = _shape_rotmat, processor = _process_rotmat)

    if (force
    or not(expected in hdf5_file)):
        return [(rotation, kwargs)]

    attrs       = hdf5_file.attrs(output_path)

    if (n_vectors                                     != attrs["n_vectors"]
    or  hdf5_file.data["*/time"][::index_slice][-1]   != attrs["time"]
    or  np.all(hdf5_file[output_path + "/tau_finite"] != tau_finites)):
        return [(rotation, kwargs)]
    elif verbose:
        for path in expected[1:]:
            time    = hdf5_file.attrs(path)["time"]
            for i, tau_finite in enumerate(tau_finites):
                Ds          = hdf5_file[path]
                Dx, Dy, Dz  = Ds[i]
                D_average   = (Dx + Dy + Dz)    / 3
                anisotropy  = (2 * Dz)          / (Dx + Dy)
                rhombicity  = (1.5 * (Dy - Dx)) / (Dz - 0.5 * (Dx + Dy))
                duration    = hdf5_file.attrs(path)["time"]
                print "DURATION  {0:5d} ns TAU_F {1:5.1f} ns".format(int(duration), tau_finite)
                if np.all(np.isnan(control)):
                    print "Dx         {0:6.4f}".format(Ds[i,0])
                    print "Dy         {0:6.4f}".format(Ds[i,1])
                    print "Dz         {0:6.4f}".format(Ds[i,2])
                    print "D_AVERAGE  {0:6.4f}".format(D_average)
                    print "ANISOTROPY {0:6.4f}".format(anisotropy)
                    print "RHOMBICITY {0:6.4f}".format(rhombicity)
                else:
                    print "           Pub.   Calc.  %"
                    print "Dx         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[0],Ds[i,0],    100 * Ds[i,0]    / control[0])
                    print "Dy         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[1],Ds[i,0],    100 * Ds[i,1]    / control[1])
                    print "Dz         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[2],Ds[i,0],    100 * Ds[i,2]    / control[2])
                    print "D_AVERAGE  {0:6.4f} {1:6.4f} {2:3.0f}".format(control[3],D_average,  100 * D_average  / control[3])
                    print "ANISOTROPY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[4],anisotropy, 100 * anisotropy / control[4])
                    print "RHOMBICITY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[5],rhombicity, 100 * rhombicity / control[5])
    return False

def translation(hdf5_file, n_cores = 1, **kwargs):
    """ Calculates the translation diffusion coefficient of <domain>
        Follows to protocol of McGuffee, S. R., and Elcock, A. H. PLoS Comp Bio. 2010. e1000694. """
    domain      = kwargs.get("domain",      "")                         # Domain name
    index_slice = kwargs.get("index_slice", 1)                          # Downsample trajectory (use every Nth frame)
    delta_ts    = kwargs.get("delta_t",     np.array([0.1]))            # Delta_t values to test (ns)
    pbc_cutoff  = kwargs.get("pbc_cutoff",  float("nan"))               # Discard points that drift more than this
                                                                        #   amount; shameful alternative to properly
                                                                        #   adjusting for periodic boundary conditions
    control     = kwargs.get("control",     float("nan"))               # Control value for comparison
    convergence = kwargs.get("convergence", False)                      # Calculate for halves, quarters for convergence
    verbose     = kwargs.get("verbose",     False)                      # Print output to terminal

    time_full   = hdf5_file.data["*/time"][::index_slice]
    com_full    = hdf5_file.data["*/com_" + domain][::index_slice]
    size        = time_full.size
    dt          = time_full[1] - time_full[0]

    if index_slice == 1: output_path = "/diffusion/translation_{0}/".format(domain)
    else:                output_path = "/diffusion/translation_{0}/slice_{1:d}/".format(domain, int(index_slice))
    if convergence:      splits      = [("full",      "[:]"),
                                        ("half/1",    "[:size/2]"),         ("half/2",    "[size/2:]"),
                                        ("quarter/1", "[:size/4]"),         ("quarter/2", "[size/4:size/2]"),
                                        ("quarter/3", "[size/2:3*size/4]"), ("quarter/4", "[3*size/4:]")]
    else:                splits      = [("full",      "[:]")]

    new_data    = [(output_path + "/delta_t",    delta_ts),
                   (output_path + "/delta_t",    {"units":      "ns"}),
                   (output_path,                 {"dt":         dt,
                                                  "time":       time_full[-1],
                                                  "pbc_cutoff": pbc_cutoff})]
    for path, index in splits:
        time    = eval("time_full{0}".format(index))
        com     = eval("com_full{0}".format(index))
        Ds      = np.zeros(delta_ts.size)
        for i, delta_t in enumerate(delta_ts):
            delta_t_index   = int(delta_t / dt)
            dr_2            = np.sum((com[:-delta_t_index] - com[delta_t_index:]) ** 2, axis = 1)
            if not np.isnan(pbc_cutoff):
                dr_2        = dr_2[dr_2 < pbc_cutoff]
            Ds[i]           = np.mean(dr_2) / (6 * delta_t)
            if verbose:
                print "DURATION  {0:5d} ns DT   {1:6.3f} ns".format(int(time.size * dt), delta_t)
                if np.isnan(control):
                    print "D          {0:4.2f}".format(Ds[i])
                else:
                    print "           Control Calc   %"
                    print "D          {0:4.2f}    {1:4.2f}   {2:3.0f}".format(control, Ds[i], 100 * Ds[i] / control)
        new_data   += [(output_path + "/" + path + "/D", Ds),
                       (output_path + "/" + path + "/D", {"units": "A2 ns-1",
                                                          "time":  time.size * dt,})]
    return new_data
def _check_translation(hdf5_file, **kwargs):
    domain      = kwargs.get("domain",    "")
    index_slice = kwargs.get("index_slice", 1)
    delta_ts    = kwargs.get("delta_t",     np.array([1000.0]))
    pbc_cutoff  = kwargs.get("pbc_cutoff",  float("nan"))
    control     = kwargs.get("control",     float("nan"))
    convergence = kwargs.get("convergence", False)
    verbose     = kwargs.get("verbose",     False)
    force       = kwargs.get("force",       False)

    if index_slice == 1: output_path = "/diffusion/translation_{0}/".format(domain)
    else:                output_path = "/diffusion/translation_{0}/slice_{1:d}/".format(domain, int(index_slice))
    if convergence:      splits      = ["full","half/1","half/2","quarter/1","quarter/2","quarter/3","quarter/4"]
    else:                splits      = ["full"]
    expected    = [output_path + "/delta_t"]
    expected   += [output_path + "/" + path + "/D" for path in splits]

    hdf5_file.load("*/time")
    hdf5_file.load("*/com_" + domain)

    if (force
    or not(expected in hdf5_file)):
        return [(translation, kwargs)]

    attrs       = hdf5_file.attrs(output_path)
    if (pbc_cutoff                                  != attrs["pbc_cutoff"]
    or  hdf5_file.data["*/time"][::index_slice][-1] != attrs["time"]
    or  np.all(hdf5_file[output_path + "/delta_t"]  != delta_ts)):
        return [(translation, kwargs)]
    elif verbose:
        for path in expected[1:]:
            time    = hdf5_file.attrs(path)["time"]
            for i, delta_t in enumerate(delta_ts):
                Ds          = hdf5_file[path]
                duration    = hdf5_file.attrs(path)["time"]
                print "DURATION  {0:5d} ns DT   {1:6.3f} ns".format(int(duration), delta_t)
                if np.isnan(control):
                    print "D          {0:4.2f}".format(Ds[i])
                else:
                    print "           Control Calc   %"
                    print "D          {0:4.2f}    {1:4.2f}   {2:3.0f}".format(control, Ds[i], 100 * Ds[i] / control)
    return False
