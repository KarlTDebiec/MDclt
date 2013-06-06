#!/usr/bin/python
desc = """diffusion.py
    Functions for secondary analysis of diffusion
    Written by Karl Debiec on 13-02-08
    Last updated 13-06-04"""
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
    vectors         = np.zeros((n_vectors, 3), np.float32)
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
def _A(v):      return np.column_stack((v**2, 2*v[:,0]*v[:,1], 2*v[:,1]*v[:,2], 2*v[:,0]*v[:,2]))
def _Q_1D_to_Q_2D(Qxx, Qyy, Qzz, Qxy, Qyz, Qxz): return np.array([[Qxx, Qxy, Qxz], [Qxy, Qyy, Qyz], [Qxz, Qyz, Qzz]])
def _Q_diag_to_D_1D(Qxx, Qyy, Qzz):
    Dxx = -Qxx + Qyy + Qzz
    Dyy =  Qxx - Qyy + Qzz
    Dzz =  Qxx + Qyy - Qzz
    return np.array([Dxx, Dyy, Dzz])
################################################ PRIMARY DATA FUNCTIONS ################################################
def _load_rotmat(self, path, **kwargs):
    segments    = self._segments()
    shapes      = np.array([self.hierarchy[segment + "/" + path[2:]].shape for segment in segments])
    data        = np.zeros((np.sum(shapes[:,0]), 9), np.float32)
    i           = 0
    for j, segment in enumerate(segments):
        data[i:i+shapes[j,0]]   = self[segment + "/" + path[2:]]
        i                      += shapes[j,0]
    return np.reshape(data, (data.shape[0], 3, 3))
################################################## ANALYSIS FUNCTIONS ##################################################
def rotation(hdf5_file, verbose = False, n_cores = 1, **kwargs):
    """ Calculates the rotational diffusion tensor of <domain> as it rotates by |rotmat| over a trajectory of length
        |time|. <n_vectors> random unit vectors are generated, rotated by |rotmat|,  and their autocorrelation functions
        calculated, integrated to time cutoff(s) <tau_finites>, and tail corrected to obtain local rotational diffusion
        coefficients. From these local coefficients the overall rotational diffusion tensor is calculated using singular
        value decomposition. Primary data may optionally be downsampled by <index_slice> for faster calculation.
        <convergence> may optionally be estimated by repeating calculations for halves and quarters of the dataset.
        Follows the protocol of Wong, V., and Case, D. A. J Phys Chem B. 2008. 112. 6013-6024. """
    domain      = kwargs.get("domain",      "")             # Domain name
    index_slice = kwargs.get("index_slice", 1)              # Downsample trajectory (use every Nth frame)
    tau_finites = kwargs.get("tau_finite",  np.array([1.0]))# Finite cutoffs for autocorrelation integration
    n_vectors   = kwargs.get("n_vectors",   1000)           # Number of unit vectors
    control     = kwargs.get("control",     np.nan)         # Control value for comparison
    convergence = kwargs.get("convergence", False)          # Calculate for halves, quarters for convergence

    time_full   = hdf5_file.data["*/log"]["time"][::index_slice]
    rotmat_full = hdf5_file.data["*/rotmat_" + domain][::index_slice]
    size        = time_full.size
    dt          = time_full[1] - time_full[0]
    vectors     = _random_unit_vectors(n_vectors)

    if index_slice == 1:    output_path = "diffusion/rotation_{0}/".format(domain)
    else:                   output_path = "diffusion/rotation_{0}/slice_{1:d}/".format(domain, int(index_slice))

    if convergence:
        splits  = [("full ",      "[:]"),
                   ("half 1 ",    "[:size/2]"),         ("half 2 ",    "[size/2:]"),
                   ("quarter 1 ", "[:size/4]"),         ("quarter 2 ", "[size/4:size/2]"),
                   ("quarter 3 ", "[size/2:3*size/4]"), ("quarter 4 ", "[3*size/4:]")]
        tensor  = [tuple(line) for line in np.column_stack([np.zeros(tau_finites.size) for x in range(43)])]
        dtype   = np.dtype([("tau finite",      "f4"),
                            ("full dx",         "f4"), ("full dy",        "f4"), ("full dz",         "f4"),
                            ("full alpha",      "f4"), ("full beta",      "f4"), ("full gamma",      "f4"),
                            ("half 1 dx",       "f4"), ("half 1 dy",      "f4"), ("half 1 dz",       "f4"),
                            ("half 1 alpha",    "f4"), ("half 1 beta",    "f4"), ("half 1 gamma",    "f4"),
                            ("half 2 dx",       "f4"), ("half 2 dy",      "f4"), ("half 2 dz",       "f4"),
                            ("half 2 alpha",    "f4"), ("half 2 beta",    "f4"), ("half 2 gamma",    "f4"),
                            ("quarter 1 dx",    "f4"), ("quarter 1 dy",   "f4"), ("quarter 1 dz",    "f4"),
                            ("quarter 1 alpha", "f4"), ("quarter 1 beta", "f4"), ("quarter 1 gamma", "f4"),
                            ("quarter 2 dx",    "f4"), ("quarter 2 dy",   "f4"), ("quarter 2 dz",    "f4"),
                            ("quarter 2 alpha", "f4"), ("quarter 2 beta", "f4"), ("quarter 2 gamma", "f4"),
                            ("quarter 3 dx",    "f4"), ("quarter 3 dy",   "f4"), ("quarter 3 dz",    "f4"),
                            ("quarter 3 alpha", "f4"), ("quarter 3 beta", "f4"), ("quarter 3 gamma", "f4"),
                            ("quarter 4 dx",    "f4"), ("quarter 4 dy",   "f4"), ("quarter 4 dz",    "f4"),
                            ("quarter 4 alpha", "f4"), ("quarter 4 beta", "f4"), ("quarter 4 gamma", "f4")])
    else:
        splits  = [("", "[:]")]
        tensor  = [tuple(line) for line in np.column_stack([np.zeros(tau_finites.size) for x in range(7)])]
        dtype   = np.dtype([("tau finite", "f4"),
                            ("dx",         "f4"), ("dy",   "f4"), ("dz",    "f4"),
                            ("alpha",      "f4"), ("beta", "f4"), ("gamma", "f4")])
    tensor                  = np.array(tensor, dtype)
    tensor["tau finite"]    = tau_finites
    attrs                   = {"tau finite units": "ns",
                               "dx units":         "ns-1",        "dy units":    "ns-1",      "dz units":    "ns-1",
                               "alpha units":      "radians",     "beta units":  "radians",   "gamma units": "radians",
                               "time":             time_full[-1], "n_vectors":   n_vectors}

    for path, index in splits:
        time    = eval("time_full{0}".format(index))
        rotmat  = eval("rotmat_full{0}".format(index))
        for i, tau_finite in enumerate(tau_finites):
            tau_ls      = np.zeros(n_vectors, np.float32)
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
            D           = _Q_diag_to_D_1D(Q[0,0], Q[1,1], Q[2,2])
            ix          = np.where(D == np.max(D))[0][0]
            iy          = np.where(np.logical_and(D != np.max(D), D != np.min(D)))[0][0]
            iz          = np.where(D == np.min(D))[0][0]
            X           = P[ix]
            Y           = P[iy]
            Z           = P[iy]
            tensor[path + "dx"][i]      = D[ix]
            tensor[path + "dy"][i]      = D[iy]
            tensor[path + "dz"][i]      = D[iz]
            tensor[path + "alpha"][i]   = np.arctan2(Z[0], -1. * Z[1])
            tensor[path + "beta"][i]    = np.arccos(Z[2])
            tensor[path + "gamma"][i]   = np.arctan2(X[2],       Y[2])
            if verbose:
                Dx          = float(tensor[path + "dx"][i])
                Dy          = float(tensor[path + "dy"][i])
                Dz          = float(tensor[path + "dz"][i])
                alpha       = float(tensor[path + "alpha"][i])
                beta        = float(tensor[path + "beta"][i])
                gamma       = float(tensor[path + "gamma"][i])
                D_average   = (Dx + Dy + Dz)    / 3
                anisotropy  = (2 * Dz)          / (Dx + Dy)
                rhombicity  = (1.5 * (Dy - Dx)) / (Dz - 0.5 * (Dx + Dy))
                print "DURATION  {0:5d} ns TAU_F {1:5.1f} ns".format(int(time.size * dt), tau_finite)
                if np.all(np.isnan(control)):
                    print "Dx         {0:6.4f}".format(Dx)
                    print "Dy         {0:6.4f}".format(Dy)
                    print "Dz         {0:6.4f}".format(Dz)
                    print "ALPHA     {0:7.4f}".format(alpha)
                    print "BETA      {0:7.4f}".format(beta)
                    print "GAMMA     {0:7.4f}".format(gamma)
                    print "D_AVERAGE  {0:6.4f}".format(D_average)
                    print "ANISOTROPY {0:6.4f}".format(anisotropy)
                    print "RHOMBICITY {0:6.4f}".format(rhombicity)
                else:
                    print "           Pub.   Calc.  %"
                    print "Dx         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[0], Dx, 100 * Dx / control[0])
                    print "Dy         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[1], Dy, 100 * Dy / control[1])
                    print "Dz         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[2], Dz, 100 * Dz / control[2])
                    print P
                    print "ALPHA            {0:7.4f}".format(alpha)
                    print "BETA             {0:7.4f}".format(beta)
                    print "GAMMA            {0:7.4f}".format(gamma)
                    print "D_AVERAGE  {0:6.4f} {1:6.4f} {2:3.0f}".format(control[3], D_average,
                                                                         100 * D_average  / control[3])
                    print "ANISOTROPY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[4], anisotropy,
                                                                         100 * anisotropy / control[4])
                    print "RHOMBICITY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[5], rhombicity,
                                                                         100 * rhombicity / control[5])
    return  [(output_path + "/tensor", tensor),
             (output_path + "/tensor", attrs)]
def _check_rotation(hdf5_file, force = False, **kwargs):
    domain      = kwargs.get("domain",      "")
    index_slice = kwargs.get("index_slice", 1)
    tau_finites = kwargs.get("tau_finite",  np.array([1.0]))
    n_vectors   = kwargs.get("n_vectors",   1000)
    control     = kwargs.get("control",     np.nan)
    convergence = kwargs.get("convergence", False)
    verbose     = kwargs.get("verbose",     False)

    if index_slice == 1: expected   = "diffusion/rotation_{0}/tensor".format(domain)
    else:                expected   = "diffusion/rotation_{0}/slice_{1:d}/tensor".format(domain, int(index_slice))
    if convergence:      splits     = ["full ","half 1 ","half 2 ","quarter 1 ","quarter 2 ","quarter 3 ","quarter 4 "]
    else:                splits     = [""]
    hdf5_file.load("*/log",              type   = "table")
    hdf5_file.load("*/rotmat_" + domain, loader = _load_rotmat)
    if    (force
    or not expected in hdf5_file):
        return [(rotation, kwargs)]

    tensor  = hdf5_file[expected]
    attrs   = hdf5_file.attrs(expected)
    if (n_vectors                                          != attrs["n_vectors"]
    or  hdf5_file.data["*/log"]["time"][::index_slice][-1] != attrs["time"]
    or  np.all(tensor["tau finite"]                        != tau_finites)):
        return [(rotation, kwargs)]
    elif verbose:
        for path in splits:
            if   "full"    in path or path == "":   time    = attrs["time"]
            elif "half"    in path:                 time    = attrs["time"] / 2.0
            elif "quarter" in path:                 time    = attrs["time"] / 4.0
            for i, tau_finite in enumerate(tau_finites):
                Dx          = float(tensor[path + "dx"][i])
                Dy          = float(tensor[path + "dy"][i])
                Dz          = float(tensor[path + "dz"][i])
                alpha       = float(tensor[path + "alpha"][i])
                beta        = float(tensor[path + "beta"][i])
                gamma       = float(tensor[path + "gamma"][i])
                D_average   = (Dx + Dy + Dz)    / 3
                anisotropy  = (2 * Dz)          / (Dx + Dy)
                rhombicity  = (1.5 * (Dy - Dx)) / (Dz - 0.5 * (Dx + Dy))
                print "DURATION  {0:5d} ns TAU_F {1:5.1f} ns".format(int(time), tau_finite)
                if np.all(np.isnan(control)):
                    print "Dx         {0:6.4f}".format(Dx)
                    print "Dy         {0:6.4f}".format(Dy)
                    print "Dz         {0:6.4f}".format(Dz)
                    print "ALPHA     {0:7.4f}".format(alpha)
                    print "BETA      {0:7.4f}".format(beta)
                    print "GAMMA     {0:7.4f}".format(gamma)
                    print "D_AVERAGE  {0:6.4f}".format(D_average)
                    print "ANISOTROPY {0:6.4f}".format(anisotropy)
                    print "RHOMBICITY {0:6.4f}".format(rhombicity)
                else:
                    print "           Pub.   Calc.  %"
                    print "Dx         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[0], Dx, 100 * Dx / control[0])
                    print "Dy         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[1], Dy, 100 * Dy / control[1])
                    print "Dz         {0:6.4f} {1:6.4f} {2:3.0f}".format(control[2], Dz, 100 * Dz / control[2])
                    print "Alpha            {0:7.4f}".format(alpha)
                    print "Beta             {0:7.4f}".format(beta)
                    print "Gamma            {0:7.4f}".format(gamma)
                    print "D_AVERAGE  {0:6.4f} {1:6.4f} {2:3.0f}".format(control[3], D_average,
                                                                         100 * D_average  / control[3])
                    print "ANISOTROPY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[4], anisotropy,
                                                                         100 * anisotropy / control[4])
                    print "RHOMBICITY {0:6.4f} {1:6.4f} {2:3.0f}".format(control[5], rhombicity,
                                                                         100 * rhombicity / control[5])
    return False

def translation(hdf5_file, n_cores = 1, **kwargs):
    """ Calculates the translational diffusion coefficient of <domain>
        Follows the protocol of McGuffee, S. R., and Elcock, A. H. PLoS Comp Bio. 2010. e1000694. """
    domain      = kwargs.get("domain",      "")             # Domain name
    index_slice = kwargs.get("index_slice", 1)              # Downsample trajectory (use every Nth frame)
    delta_ts    = kwargs.get("delta_t",     np.array([0.1]))# Delta_t values to test (ns)
    pbc_cutoff  = kwargs.get("pbc_cutoff",  np.nan)         # Discard points that drift more than this amount; shameful
                                                            # alternative to properly adjusting for periodic boundary
                                                            # conditions
    control     = kwargs.get("control",     np.nan)         # Control value for comparison
    convergence = kwargs.get("convergence", False)          # Calculate for halves, quarters for convergence
    verbose     = kwargs.get("verbose",     False)          # Print output to terminal

    time_full   = hdf5_file.data["*/log"]["time"][::index_slice]
    com_full    = hdf5_file.data["*/com_" + domain][::index_slice]
    size        = time_full.size
    dt          = time_full[1] - time_full[0]

    if index_slice == 1: output_path = "diffusion/translation_{0}/".format(domain)
    else:                output_path = "diffusion/translation_{0}/slice_{1:d}/".format(domain, int(index_slice))
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
        new_data   += [(output_path + "/" + path + "/D", Ds, kwargs),
                       (output_path + "/" + path + "/D", {"units": "A2 ns-1",
                                                          "time":  time.size * dt,})]
    return new_data
def _check_translation(hdf5_file, force = False, **kwargs):
    domain      = kwargs.get("domain",    "")
    index_slice = kwargs.get("index_slice", 1)
    delta_ts    = kwargs.get("delta_t",     np.array([0.1]))
    pbc_cutoff  = kwargs.get("pbc_cutoff",  np.nan)
    control     = kwargs.get("control",     np.nan)
    convergence = kwargs.get("convergence", False)
    verbose     = kwargs.get("verbose",     False)

    if index_slice == 1: output_path = "diffusion/translation_{0}/".format(domain)
    else:                output_path = "diffusion/translation_{0}/slice_{1:d}/".format(domain, int(index_slice))
    if convergence:      splits      = ["full","half/1","half/2","quarter/1","quarter/2","quarter/3","quarter/4"]
    else:                splits      = ["full"]
    expected    = [output_path + "/delta_t"]
    expected   += [output_path + "/" + path + "/D" for path in splits]

    hdf5_file.load("*/log", type = "table")    
    hdf5_file.load("*/com_" + domain)

    if (force
    or not(expected in hdf5_file)):
        return [(translation, kwargs)]

    attrs       = hdf5_file.attrs(output_path)
    if (pbc_cutoff                                         != attrs["pbc_cutoff"]
    or  hdf5_file.data["*/log"]["time"][::index_slice][-1] != attrs["time"]
    or  np.all(hdf5_file[output_path + "/delta_t"]         != delta_ts)):
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
