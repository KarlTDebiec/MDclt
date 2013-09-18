#!/usr/bin/python
desc = """diffusion.py
    Functions for secondary analysis of diffusion
    Written by Karl Debiec on 13-02-08
    Last updated 13-07-21"""
########################################### MODULES, SETTINGS, AND DEFAULTS ############################################
import os, sys, warnings
import time as time_module
import numpy as np
from   multiprocessing   import Pool
from   scipy.optimize    import fmin, curve_fit
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
def rotation(hdf5_file, 
             destination    = "",                           # Origin of primary data and destination of secondary data
             index_slice    = 1,                            # Downsample trajectory (use every Nth frame)
             tau_finites    = np.array([1.0]),              # Finite cutoffs for autocorrelation integration
             n_vectors      = 1000,                         # Number of unit vectors
             control        = np.nan,                       # Control value for comparison
             convergence    = False,                        # Calculate for halves, quarters for convergence
             verbose        = False, n_cores = 1, **kwargs):
    """ Calculates the rotational diffusion tensor of <domain> as it rotates by |rotmat| over a trajectory of length
        |time|. <n_vectors> random unit vectors are generated, rotated by |rotmat|,  and their autocorrelation functions
        calculated, integrated to time cutoff(s) <tau_finites>, and tail corrected to obtain local rotational diffusion
        coefficients. From these local coefficients the overall rotational diffusion tensor is calculated using singular
        value decomposition. Primary data may optionally be downsampled by <index_slice> for faster calculation.
        <convergence> may optionally be estimated by repeating calculations for halves and quarters of the dataset.
        Follows the protocol of Wong, V., and Case, D. A. J Phys Chem B. 2008. 112. 6013-6024. """
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination
    time_full   = hdf5_file.data["*/log"]["time"][::index_slice]
    rotmat_full = hdf5_file.data["*/rotmat" + destination][::index_slice]
    size        = time_full.size
    dt          = time_full[1] - time_full[0]
    vectors     = _random_unit_vectors(n_vectors)

    if index_slice == 1:    output_path = "diffusion/rotation{0}/".format(destination)
    else:                   output_path = "diffusion/rotation{0}/slice_{1:d}/".format(destination, int(index_slice))

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
                _print_rotation(tensor, [path], control, attrs)
    return  [(output_path + "/tensor", tensor),
             (output_path + "/tensor", attrs)]
def _check_rotation(hdf5_file, force = False, **kwargs):
    destination = kwargs.get("destination", "")
    index_slice = kwargs.get("index_slice", 1)
    tau_finites = kwargs.get("tau_finite",  np.array([1.0]))
    n_vectors   = kwargs.get("n_vectors",   1000)
    control     = kwargs.get("control",     np.nan)
    convergence = kwargs.get("convergence", False)
    verbose     = kwargs.get("verbose",     False)
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination

    if index_slice == 1: expected   = "diffusion/rotation{0}/tensor".format(destination)
    else:                expected   = "diffusion/rotation{0}/slice_{1:d}/tensor".format(destination, int(index_slice))
    if convergence:      splits     = ["full ","half 1 ","half 2 ","quarter 1 ","quarter 2 ","quarter 3 ","quarter 4 "]
    else:                splits     = [""]
    hdf5_file.load("*/log",                  type   = "table")
    hdf5_file.load("*/rotmat" + destination, loader = _load_rotmat)
    if     (force
    or not (expected in hdf5_file)):
        return [(rotation, kwargs)]

    D_table = hdf5_file[expected]
    attrs   = hdf5_file.attrs(expected)
    if (n_vectors                                          != attrs["n_vectors"]
    or (np.any(np.array(tau_finites, np.float32)           != D_table["tau finite"]))
    or (hdf5_file.data["*/log"]["time"][::index_slice][-1] != attrs["time"])):
        return [(rotation, kwargs)]
    elif verbose:
        _print_rotation(D_table, splits, control, attrs)
    return False
def _print_rotation(data, splits, control, attrs):
    for path in splits:
        if   "full"    in path or path == "":   time    = np.round(attrs["time"],       0)
        elif "half"    in path:                 time    = np.round(attrs["time"] / 2.0, 0)
        elif "quarter" in path:                 time    = np.round(attrs["time"] / 4.0, 0)
        for i, tau_finite in enumerate(data["tau finite"]):
            Dx          = float(data[path + "dx"][i])
            Dy          = float(data[path + "dy"][i])
            Dz          = float(data[path + "dz"][i])
            alpha       = float(data[path + "alpha"][i])
            beta        = float(data[path + "beta"][i])
            gamma       = float(data[path + "gamma"][i])
            D_average   = (Dx + Dy + Dz)    / 3
            anisotropy  = (2 * Dz)          / (Dx + Dy)
            rhombicity  = (1.5 * (Dy - Dx)) / (Dz - 0.5 * (Dx + Dy))
            print "DURATION  {0:5d} ns TAU_F {1:5.1f} ns".format(int(time), float(tau_finite))
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

def _block(data, function, **kwargs):
    full_size       = data.size
    sizes           = np.array(sorted(list(set([full_size / x for x in range(1, full_size)]))), np.int)[:-1]
    sds             = np.zeros(sizes.size)
    n_blocks        = full_size // sizes
    for i, size in enumerate(sizes):
        resized     = np.resize(data, (full_size // size, size))
        values      = function(resized, **kwargs)
        sds[i]      = np.std(values)
    ses             = sds / np.sqrt(n_blocks)
    return sizes, ses

def _fit_sigmoid(x, y):
    def func(x, min_asym, max_asym, poi, k): return max_asym + (min_asym - max_asym) / (1 + (x / poi) ** k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_asym, max_asym, poi, k  = curve_fit(func, x, y, maxfev = 100000)[0]
    y_fit   = func(x, min_asym, max_asym, poi, k)
    return min_asym, max_asym, poi, k, y_fit

def translation(hdf5_file,
                destination     = "",                       # Origin of primary data and destination of secondary data
                delta_t         = np.array([0.1]),          # Delta_t values to test (ns)
                selection       = [],                       # Selection names 
                explicit_names  = False,                    # Explicitly name selections in output table
                control         = np.nan,                   # Control value for comparison (if verbose)
                verbose         = False, n_cores = 1, **kwargs):
    """ Calculates the translational diffusion coefficient of <selection>(s) over intervals of <delta_t>. Stores in a
        table at <destination> numbered either as 0, 1, 2, ... or with explicit selection strings if <explicit_names>.
        Follows the protocol of McGuffee, S. R., and Elcock, A. H. PLoS Comp Bio. 2010. e1000694. Error is estimated
        using the blocking method of Flyvbjerg, H., and Petersen, H. G. Error Estimates on Averages of Correlated Data.
        J Phys Chem. 1989. 91. 461-466."""
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination
    time            = hdf5_file.data["*/log"]["time"]
    com             = hdf5_file.data["*/com" + destination]
    size            = time.size
    dt              = time[1] - time[0]
    n_selections    = com.shape[1]
    selection_attr  = " ".join(["\"{0}\"".format(sel) for sel in selection])
    D_table         = []
    dtype_string    = "np.dtype([('delta t', 'f4')"
    for i, selection in enumerate(selection):
        if explicit_names:  dtype_string   += ",('{0} D', 'f4'), ('{0} D se', 'f4')".format(selection)
        else:               dtype_string   += ",('{0} D', 'f4'), ('{0} D se', 'f4')".format(i)
    dtype_string   += "])"
    attrs   = {"time": time[-1], "delta t units": "ns", "D units": "A2 ps-1", "selection": selection_attr}

    def calc_D_block(dr_2, delta_t): return  np.mean(dr_2, axis = 1) / (6 * delta_t)

    for delta_t in delta_t:
        D_table        += [[delta_t]]
        delta_t_index   = int(np.round(delta_t / dt))
        dr_2            = np.sum((com[delta_t_index:] - com[:-delta_t_index]) ** 2, axis = 2)
        for j in xrange(dr_2.shape[1]):
            x, y        = _block(dr_2[:,j], calc_D_block, delta_t = delta_t)
            try:
                _, max_asym, _, _, _    = _fit_sigmoid(x, y)
            except:
                max_asym                = np.nan
                attrs["note"]           = "Standard error calculation failed for one or more delta t"
            D_table[-1]    += [np.mean(dr_2[:,j], axis = 0) / (6 * delta_t) / 1000.0, max_asym / 1000.0]
    for i in xrange(len(D_table)):
        D_table[i]          = tuple(D_table[i])
    D_table                 = np.array(D_table, eval(dtype_string))

    if verbose:     _print_translation(D_table, attrs)
    return  [("diffusion/translation" + destination, D_table),
             ("diffusion/translation" + destination, attrs)]
def _check_translation(hdf5_file, force = False, **kwargs):
    destination = kwargs.get("destination", "")
    delta_ts    = kwargs.get("delta_t",     np.array([0.1]))
    verbose     = kwargs.get("verbose",     False)
    if not (destination == "" or destination.startswith("_")):
        destination = "_" + destination
    selection           = hdf5_file.attrs("0000/com" + destination)["selection"].split("\" \"")
    selection           = [sel.strip("\"") for sel in selection]
    kwargs["selection"] = selection
    hdf5_file.load("*/log", type = "table")
    hdf5_file.load("*/com" + destination)

    if (force
    or not("diffusion/translation" + destination in hdf5_file)):
        return [(translation, kwargs)]

    data    = hdf5_file["diffusion/translation" + destination]
    attrs   = hdf5_file.attrs("diffusion/translation" + destination)

    if (np.any(np.array(delta_ts, np.float32) != data["delta t"])
    or (hdf5_file.data["*/log"]["time"][-1]   != attrs["time"])):
        return [(translation, kwargs)]
    elif verbose:   _print_translation(data, attrs)
    return False
def _print_translation(data, attrs):
    print "DURATION {0:5d} ns".format(int(attrs["time"]))
    print "DELTA_T  D        D se"
    for line in data:
        print "{0:7.3f}  {1:7.5f}  {2:7.5f}".format(float(line[0]), float(line[1]), float(line[2]))


