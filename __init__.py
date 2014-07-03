#!/usr/bin/python
#   MDclt.__init__.py
#    Written by Karl Debiec on 12-02-12, last updated by Karl Debiec on 14-07-03
"""
Command Line Tools for analysis of molecular dynamics simulations

.. todo:
    - Rewrite analysis functions as classes
    - Consider using WESTPA's work manager
    - Use mdtraj
    - Documentation
"""
################################################## GENERAL FUNCTIONS ###################################################
def ignore_index(time, ignore):
    if   ignore <  0:   return np.where(time > time[-1] + ignore - (time[1] - time[0]))[0][0]
    elif ignore == 0:   return 0
    elif ignore >  0:   return np.where(time > ignore)[0][0]
def is_num(test):
    try:    float(test)
    except: return False
    return  True
def month(string):
    month = {"jan":  1, "feb":  2, "mar":  3, "apr":  4, "may":  5, "jun":  6,
             "jul":  7, "aug":  8, "sep":  9, "oct": 10, "nov": 11, "dec": 12}
    try:    return month[string.lower()]
    except: return None
def block(data, func, min_size = 3):
    full_size   = data.shape[0]
    sizes       = [s for s in list(set([full_size / s for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    sds         = np.zeros(sizes.size)
    n_blocks    = full_size // sizes
    for i, size in enumerate(sizes):
        resized = np.resize(data, (full_size // size, size, 3))
        values  = map(func, resized)
        sds[i]  = np.std(values)
    ses                 = sds / np.sqrt(n_blocks - 1.0)
    se_sds              = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
    se_sds[se_sds == 0] = se_sds[np.where(se_sds == 0)[0] + 1]
    return sizes, ses, se_sds
def block_average(data, func = np.mean, func_kwargs = {"axis": 1}, min_size = 1, **kwargs):
    full_size   = data.size
    sizes       = [s for s in list(set([full_size / s for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    sds         = np.zeros(sizes.size)
    n_blocks    = full_size // sizes
    for i, size in enumerate(sizes):
        resized = np.resize(data, (full_size // size, size))
        values  = func(resized, **func_kwargs)
        sds[i]  = np.std(values)
    ses                 = sds / np.sqrt(n_blocks - 1.0)
#    se_sds              = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
    se_sds              = (1.0 / np.sqrt(2.0 * (n_blocks - 1.0))) * ses
    if ses[-1] == 0.0 or se_sds[-1] == 0.0:                                     # This happens occasionally and
        sizes   = sizes[:-1]                                                    # disrupts curve_fit; it is not clear
        ses     = ses[:-1]                                                      # why
        se_sds  = se_sds[:-1]
    return sizes, ses, se_sds
def fit_curve(fit_func = "single_exponential", **kwargs):
    import warnings
    from   scipy.optimize import curve_fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        def single_exponential(x, y, **kwargs):
            def func(x, a, b, c):       return a + b * np.exp(c * x)
            a, b, c       = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, func(x, a, b, c)
        def double_exponential(x, y, **kwargs):
            def func(x, a, b, c, d, e): return a + b * np.exp(c * x) + d * np.exp(e * x)
            a, b, c, d, e = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, e, func(x, a, b, c, d, e)
        def sigmoid(x, y, **kwargs):
            def func(x, a, b, c, d):    return b + (a - b) / (1.0 + (x / c) ** d)
            a, b, c, d    = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, func(x, a, b, c, d)
        return locals()[fit_func](**kwargs)

