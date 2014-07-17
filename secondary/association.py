#!/usr/bin/python
#   MDclt.secondary.Association.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-07-10
"""
Classes for analysis of molecular association

.. todo:
    - Support ignoring portion of dataset
"""
################################### MODULES ####################################
import os, sys
import numpy as np
from MDclt import secondary
from MDclt import Block, Block_Generator, Block_Accumulator, Block_Acceptor
from MDclt import  pool_director
################################## FUNCTIONS ###################################
def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: argparse subparsers object to add subparser
        :*args*:       Passed to subparsers.add_parser(...)
        :*kwargs*:     Passed to subparsers.add_parser(...)

    .. todo:
        - Implement nested subparser (should be 'amber log', not just 'log')
    """
    subparser  = secondary.add_parser(subparsers,
      name     = "association",
      help     = "Analyzes molecular association")
    arg_groups = {ag.title:ag for ag in subparser._action_groups}

    arg_groups["input"].add_argument(
      "-coord",
      type     = str,
      required = True,
      nargs    = 2,
      metavar  = ("H5_FILE", "ADDRESS"),
      help     = "H5 file and address from which to load coordinate")

    arg_groups["action"].add_argument(
      "-bound",
      type     = float,
      required = True,
      help     = "Bound state cutoff along coordinate")
    arg_groups["action"].add_argument(
      "-unbound",
      type     = float,
      required = True,
      help     = "Unbound state cutoff along coordinate")

    subparser.set_defaults(analysis = command_line)

def concentration(n, volume):
    """
    """
    return float(n) / 6.0221415e23 / (volume * 1e-27)

def block_average(data, func = np.mean, func_kwargs = {"axis": 1}, min_size = 1,
                  **kwargs):
    """
    """
    full_size   = data.size
    sizes       = [s for s in list(set([full_size / s
                    for s in range(1, full_size)])) if s >= min_size]
    sizes       = np.array(sorted(sizes), np.int)[:-1]
    sds         = np.zeros(sizes.size)
    n_blocks    = full_size // sizes
    for i, size in enumerate(sizes):
        resized = np.resize(data, (full_size // size, size))
        values  = func(resized, **func_kwargs)
        sds[i]  = np.std(values)
    ses    = sds / np.sqrt(n_blocks - 1.0)
    # se_sds = np.sqrt((2.0) / (n_blocks - 1.0)) * ses
    se_sds = (1.0 / np.sqrt(2.0 * (n_blocks - 1.0))) * ses
    if ses[-1] == 0.0 or se_sds[-1] == 0.0: # This happens occasionally and
        sizes   = sizes[:-1]                #   disrupts curve_fit; it is not
        ses     = ses[:-1]                  #   clear why
        se_sds  = se_sds[:-1]
    return sizes, ses, se_sds

def fit_curve(fit_func = "single_exponential", **kwargs):
    """
    """
    import warnings
    from   scipy.optimize import curve_fit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def single_exponential(x, y, **kwargs):
            def func(x, a, b, c):
                return a + b * np.exp(c * x)
            a, b, c       = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, func(x, a, b, c)

        def double_exponential(x, y, **kwargs):
            def func(x, a, b, c, d, e):
                return a + b * np.exp(c * x) + d * np.exp(e * x)
            a, b, c, d, e = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, e, func(x, a, b, c, d, e)

        def sigmoid(x, y, **kwargs):
            def func(x, a, b, c, d):
                return b + (a - b) / (1.0 + (x / c) ** d)
            a, b, c, d    = curve_fit(func, x, y, **kwargs)[0]
            return a, b, c, d, func(x, a, b, c, d)

        return locals()[fit_func](**kwargs)

def command_line(n_cores = 1, **kwargs):
    """
    Provides command line functionality for this analysis

    **Arguments:**
        :*n_cores*: Number of cores to use

    .. todo:
        - Figure out syntax to get this into MDclt.primary
    """
    from multiprocessing import Pool

    block_generator   = Block_Generator(**kwargs)
    block_accumulator = Block_Accumulator(**kwargs)
    block_accumulator.next()

    # Serial
    # for block in block_generator:
    #     block()
    #     block_accumulator.send(block)

    # Parallel (Processes)
    pool              = Pool(n_cores)
    for block in pool.imap_unordered(pool_director, block_generator):
        block_accumulator.send(block)
    pool.close()
    pool.join()

    block_accumulator.close()

    block_acceptor = Block_Acceptor(**kwargs)
    block_acceptor.next()
    block_acceptor.send(block_accumulator)
    block_acceptor.close()

################################### CLASSES ####################################
class Block(Block):
    def __init__(self, slc, **kwargs):
        for key, value in kwargs.items():       # Is this shameful?
            setattr(self, key, value)
        self.n_frames = self.coord.shape[0]
        self.i = slc[1].stop
        self.j = slc[2].stop

        self.datasets  = [dict(address = "pair_association",
                               slc     = slc[1:],
                               attrs   = {})]
        self.datasets += [dict(address = "events",
                               attrs   = {},
                               data    = [])]

    def __call__(self, **kwargs):
        events = self.datasets[1]["data"]

        # Assign to bound and unbound states
        bound                              = np.zeros(self.n_frames, np.int8)
        unbound                            = np.zeros(self.n_frames, np.int8)
        bound[self.coord < self.bound]     = 1
        unbound[self.coord > self.unbound] = 1

        # Locate transitions
        trans_bound   =   bound[1:] -   bound[:-1]
        trans_unbound = unbound[1:] - unbound[:-1]
        enter_bound   = np.where(trans_bound   == 1)[0] + 1
        enter_unbound = np.where(trans_unbound == 1)[0] + 1

        # Set state to bound between transitions;
        #   consider alternative of using a list of slices
        if enter_bound.size >= 1:
                                   # Start at first entrance of bound state
            enter = enter_bound[0]
            while True:
                try:                                                # Look for next entrance of unbound state
                    exit = enter_unbound[enter_unbound > enter][0]  #   Next entrance found
                    bound[enter:exit] = 1                           #   Set state to bound between entrances
                except:                                             # Trajectory ends with pair in bound state
                    bound[enter::]    = 1                           #   Set state to bound until end of trajectory
                    break                                           #   Exit
                try:                                                # Look for next entrance of bound state
                    enter = enter_bound[enter_bound > exit][0]      #   Next entrance found
                except:                                             # Trajectory ends with pair in unbound state
                    break                                           #   Exit

        # Calculate pbound
        pbound = np.sum(bound, dtype = np.float64) / float(bound.size)

        # Calculate standard error of pbound using block averaging (STORE PBOUND FOR ALL BLOCKS)
        sizes, ses, se_sds = block_average(bound)
        try:    a, pbound_se, c, d, fit = fit_curve(x = sizes, y = ses, sigma = se_sds, fit_func = "sigmoid")
        except:    pbound_se            = pbound

        # Calculate mean first passage time and tabulate binding events
        trans_bound   = bound[:-1] - bound[1:]
        enter_bound   = np.where(trans_bound  == -1)[0] + 1
        enter_unbound = np.where(trans_bound  ==  1)[0] + 1
        if enter_bound.size >= 1 and enter_unbound.size >= 1:
            if   enter_bound[0] < enter_unbound[0] and enter_bound[-1] < enter_unbound[-1]:     # Started unbound,
                fpt_on   = enter_bound[1:]   - enter_unbound[:-1]                               #   ended unbound
                fpt_off  = enter_unbound     - enter_bound
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                               for bind, unbind in np.column_stack((enter_bound, enter_unbound))] 
            elif enter_bound[0] < enter_unbound[0] and enter_bound[-1] > enter_unbound[-1]:     # Started unbound,
                fpt_on   = enter_bound[1:]   - enter_unbound                                    #   ended bound
                fpt_off  = enter_unbound     - enter_bound[:-1]
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                               for bind, unbind in np.column_stack((enter_bound[:-1], enter_unbound))] 
            elif enter_bound[0] > enter_unbound[0] and enter_bound[-1] < enter_unbound[-1]:     # Started bound,
                fpt_on   = enter_bound       - enter_unbound[:-1]                               #   ended unbound
                fpt_off  = enter_unbound[1:] - enter_bound
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                               for bind, unbind in np.column_stack((enter_bound, enter_unbound[1:]))] 
            elif enter_bound[0] > enter_unbound[0] and enter_bound[-1] > enter_unbound[-1]:     # Started bound,
                fpt_on   = enter_bound       - enter_unbound                                    #   ended bound
                fpt_off  = enter_unbound[1:] - enter_bound[:-1]
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                               for bind, unbind in np.column_stack((enter_bound[:-1] * dt, enter_unbound[1:] * dt))]

            # Convert mean first passage time into rates
            if fpt_on.size != 0:
                mean_fpt_on    = np.mean(fpt_on)
                mean_fpt_on_se = np.std(fpt_on)  / np.sqrt(fpt_on.size)
                kon_sim        = 1 / mean_fpt_on
                kon_sim_se     = kon_sim  * (mean_fpt_on_se  / mean_fpt_on)
                kon            = kon_sim  / (self.conc_single * self.conc_single)
                kon_se         = kon      * (kon_sim_se      / kon_sim)
            else:
                mean_fpt_on,  mean_fpt_on_se = np.nan, np.nan
                kon,          kon_se         = np.nan, np.nan
            if fpt_off.size != 0:
                mean_fpt_off    = np.mean(fpt_off)
                mean_fpt_off_se = np.std(fpt_off) / np.sqrt(fpt_off.size)
                koff_sim        = 1 / mean_fpt_off
                koff_sim_se     = koff_sim * (mean_fpt_off_se / mean_fpt_off)
                koff            = koff_sim /  self.conc_single
                koff_se         = koff     * (koff_sim_se     / koff_sim) 
            else:
                mean_fpt_off, mean_fpt_off_se = np.nan, np.nan
                koff,         koff_se         = np.nan, np.nan

        # Pair never switches between bound and unbound states
        else:
            mean_fpt_on,  mean_fpt_on_se  = np.nan, np.nan
            mean_fpt_off, mean_fpt_off_se = np.nan, np.nan
            kon,          kon_se          = np.nan, np.nan
            koff,         koff_se         = np.nan, np.nan

        # Organize data
        self.bound = bound
        pair_association = np.zeros(1,
          dtype = [("pbound",       "f4"), ("pbound se",       "f4"),
                   ("mean fpt on",  "f4"), ("mean fpt on se",  "f4"),
                   ("mean fpt off", "f4"), ("mean fpt off se", "f4"),
                   ("kon",          "f4"), ("kon se",          "f4"),
                   ("koff",         "f4"), ("koff se",         "f4")])
        pair_association["pbound"],       pair_association["pbound se"]       = pbound,       pbound_se
        pair_association["mean fpt on"],  pair_association["mean fpt on se"]  = mean_fpt_on,  mean_fpt_on_se
        pair_association["mean fpt off"], pair_association["mean fpt off se"] = mean_fpt_off, mean_fpt_off_se
        pair_association["kon"],          pair_association["kon se"]          = kon,          kon_se
        pair_association["koff"],         pair_association["koff se"]         = koff,         koff_se

        self.datasets[0]["data"] = pair_association
        self.datasets[1]["data"] = events

class Block_Generator(Block_Generator):
    """
    Generator class that yields blocks of analysis
    """
    def __init__(self, log, coord, bound, unbound, output, force = False, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: List including path to h5 file and address within h5 file
            :*force*:  Run analysis even if no new data is present
        """
        import warnings
        from h5py import File as h5

        # Unpack arguments
        log_path,   log_address   = log
        coord_path, coord_address = coord
        out_path,   out_address   = output

        # Check input and output files
        with h5(out_path) as out_h5, h5(log_path) as log_h5, h5(coord_path) as coord_h5:
            log_shape   = log_h5[log_address].shape
            coord_shape = coord_h5[coord_address].shape
            if log_shape[0] != coord_shape[0]:
                warning  = "Length of log dataset ({0}) ".format(log_shape[0])
                warning += "and coordinate dataset ({0}) ".format(coord_shape[0])
                warning += "do not match, using smaller of the two"
                warnings.warn(warning)
            self.volume             = np.mean(log_h5[log_address]["volume"])
            self.first_index        = 0
            self.final_index        = min(log_shape[0], coord_shape[0])
            self.n_molecule_1       = coord_shape[1]
            self.n_molecule_2       = coord_shape[2]
            self.current_molecule_1 = 0
            self.current_molecule_2 = 0

        # Store necessary data in instance variables
        self.log_path      = log_path
        self.log_address   = log_address
        self.coord_path    = coord_path
        self.coord_address = coord_address
        self.out_address   = out_address
        self.bound         = bound
        self.unbound       = unbound
        self.conc_single     = concentration(1, self.volume)
        self.conc_molecule_1 = concentration(self.n_molecule_1, self.volume)
        self.conc_molecule_2 = concentration(self.n_molecule_2, self.volume)

    def next(self):
        """
        Prepares and yields next Block of analysis
        """
        from h5py import File as h5

        if self.current_molecule_1 == self.n_molecule_1:
            raise StopIteration()
        else:
            # Load primary data (consider keeping coord_h5 open from __init__ on...)
            with h5(self.coord_path) as coord_h5:
                coord = np.array(coord_h5[self.coord_address][self.first_index:self.final_index,
                                                              self.current_molecule_1,
                                                              self.current_molecule_2])
                # SHOULD ALSO LOAD RELEVANT PORTION OF BLOCK AVERAGE DATASET
            block = Block(
                     coord       = coord,
                     bound       = self.bound,
                     unbound     = self.unbound,
                     conc_single = self.conc_single,
                     out_address = self.out_address,
                     slc         = (slice(self.first_index, self.final_index, 1),
                                    slice(self.current_molecule_1),
                                    slice(self.current_molecule_2)))
            # Iterate and return
            self.current_molecule_2     += 1
            if self.current_molecule_2  == self.n_molecule_2:
                self.current_molecule_1 += 1
                self.current_molecule_2  = 0
            return block

class Block_Accumulator(Block_Accumulator):
    def __init__(self, log, coord, **kwargs):
        from h5py import File as h5

        for key, value in kwargs.items():       # Is this shameful?
            setattr(self, key, value)
        self.func        = self.accumulate()

        log_path,   log_address   = log
        coord_path, coord_address = coord
        with h5(log_path) as log_h5, h5(coord_path) as coord_h5:
            # Need to support omitting the beginning and end of trajectories here
            coord_shape       = coord_h5[coord_address].shape
            self.n_molecule_1 = coord_shape[1]
            self.n_molecule_2 = coord_shape[2]
            self.volume       = np.mean(log_h5[log_address]["volume"])
        self.conc_molecule_1  = concentration(self.n_molecule_1, self.volume)
        self.conc_molecule_2  = concentration(self.n_molecule_2, self.volume)
        self.count            = np.zeros((coord_shape[0], self.n_molecule_1), np.int8)

        # Prepare datasets
        self.datasets  = [dict(address = "pair_association",
                               attrs   = {},
                               data    = np.zeros((self.n_molecule_1, self.n_molecule_2),
                                           dtype = [("pbound",       "f4"), ("pbound se",       "f4"),
                                                    ("mean fpt on",  "f4"), ("mean fpt on se",  "f4"),
                                                    ("mean fpt off", "f4"), ("mean fpt off se", "f4"),
                                                    ("kon",          "f4"), ("kon se",          "f4"),
                                                    ("koff",         "f4"), ("koff se",         "f4")]))]
        self.datasets += [dict(address = "overall_association",
                               attrs   = {},
                               data    = np.zeros(1,
                                           dtype = [("pbound",       "f4"), ("pbound se",       "f4"),
                                                    ("mean fpt on",  "f4"), ("mean fpt on se",  "f4"),
                                                    ("mean fpt off", "f4"), ("mean fpt off se", "f4"),
                                                    ("kon",          "f4"), ("kon se",          "f4"),
                                                    ("koff",         "f4"), ("koff se",         "f4")]))]
        self.datasets += [dict(address = "events",
                               attrs   = {},
                               data    = [])]

    def accumulate(self, **kwargs):
        pair_assoc = self.datasets[0]["data"]
        events     = self.datasets[2]["data"]
        while True:
            block   = (yield)
            events += block.datasets[1]["data"]
            pair_assoc[block.i, block.j] = block.datasets[0]["data"]
            self.count[:,block.i] += block.bound

    def close(self, *args, **kwargs):
        self.func.close(*args, **kwargs)

        # Link to datasets for clarity
        pair_assoc  = self.datasets[0]["data"]
        total_assoc = self.datasets[1]["data"]
        events      = self.datasets[2]["data"]
        count       = self.count
        C_mol1_total = self.conc_molecule_1
        C_mol2_total = self.conc_molecule_2

        # Configure pair and total association datasets
        for key in ["pbound", "mean fpt on", "mean fpt off", "kon", "koff"]:
            pair_assoc_ma          = np.ma.MaskedArray(pair_assoc[key],       np.isnan(pair_assoc[key]))
            pair_assoc_ma_se       = np.ma.MaskedArray(pair_assoc[key+" se"], np.isnan(pair_assoc[key+" se"]))
            total_assoc[key]       = np.mean(pair_assoc_ma)
            total_assoc[key+" se"] = np.sqrt(np.sum(pair_assoc_ma_se ** 2.0)) / pair_assoc_ma.size
        self.datasets[0]["attrs"].update({"fpt units": "ps", "kon units": "M-2 ps-1", "koff units": "M-1 ps-1"})
        self.datasets[1]["attrs"].update({"fpt units": "ps", "kon units": "M-2 ps-1", "koff units": "M-1 ps-1"})

        # Configure binding event dataset
        events = np.array(events,
                   dtype = [("index 1", "i4"), ("index 2",  "i4"), ("start", "f4"),
                            ("end",     "f4"), ("duration", "f4")])
        self.datasets[2]["attrs"].update({"duration units": "ps"})

        # Calculate Pstate of different bound states
        n_states  = np.max(count) + 1
        Pstate    = np.zeros(n_states)
        Pstate_se = np.zeros(n_states)
        for i in range(0, n_states):                                                # i = number of mol 2 bound to mol 1
            Pstate[i]       = float(count[count == i].size) / float(count.size)     # Accounts for presence of mult. mol 1
            mol1_in_state_i = np.zeros(count.shape, np.int8)                        # Binary over trajectory
            mol1_in_state_i[count == i] = 1
            total_mol1_in_state_i       = np.sum(mol1_in_state_i, axis = 1, dtype = np.float64)

            # Calculate standard error of Pstate using block averaging
            sizes, ses, se_sds                 = block_average(total_mol1_in_state_i / self.n_molecule_1)
            try:    a, Pstate_se[i], c, d, fit = fit_curve(x = sizes, y = ses, sigma = se_sds, fit_func = "sigmoid")
            except:    Pstate_se[i]            = Pstate[i]
        if True:
            for i in range(Pstate.size): print "P state {0:02d}: {1:7.5f} ({2:7.5f})".format(i, Pstate[i], Pstate_se[i])
            print "Sum       : {0:7.5f}".format(np.sum(Pstate))

        # Calculate concentrations
        C_mol1_free    = C_mol1_total * Pstate[0]
        C_mol1_free_se = C_mol1_free  * (Pstate_se[0] / Pstate[0])
        C_mol2_free    = C_mol2_total - C_mol1_total * np.sum([i * P for i, P in enumerate(Pstate[1:], 1)])
        C_mol2_free_se = C_mol1_total * np.sqrt(np.sum([(i * P_se) ** 2 for i, P_se in enumerate(Pstate_se[1:], 1)]))
        C_bound        = C_mol1_total * Pstate[1:]
        C_bound_se     = C_bound      * (Pstate_se[1:] / Pstate[1:])
        np.set_printoptions(precision = 5, suppress = True, linewidth = 120)
        if True:
            print "[mol1 total]:      {0:7.5f}".format(C_mol1_total)
            print "[mol2 total]:      {0:7.5f}".format(C_mol2_total)
            print "[mol1 free] (se):  {0:7.5f} ({1:7.5f})".format(C_mol1_free, C_mol1_free_se)
            print "[mol2 free] (se):  {0:7.5f} ({1:7.5f})".format(C_mol2_free, C_mol2_free_se)
            for i in range(C_bound.size):
                print "[complex {0:02d}] (se): {1:7.5f} ({2:7.5f})".format(i + 1, C_bound[i],     C_bound_se[i])
            print "Sum [mol1]:        {0:7.5f}".format(C_mol1_free + np.sum(C_bound))
            print "Sum [mol2]:        {0:7.5f}".format(C_mol2_free + np.sum([i * C for i, C in enumerate(C_bound, 1)]))

        # Calculate KAs
        KA              = np.zeros(n_states - 1)
        KA_se           = np.zeros(n_states - 1)
        KA[0]           = C_bound[0] / (C_mol1_free * C_mol2_free)
        KA_se[0]        = KA[0] * np.sqrt((C_bound_se[0] / C_bound[0]) ** 2 + (C_mol1_free_se  / C_mol1_free)  ** 2
                                          + (C_mol2_free_se / C_mol2_free) ** 2)
        for i in range(1, n_states - 1):
            KA[i]     = C_bound[i] / (C_bound[i-1] * C_mol2_free)
            KA_se[i]  = KA[i] * np.sqrt((C_bound_se[i] / C_bound[i]) ** 2 + (C_bound_se[i-1] / C_bound[i-1]) ** 2
                                          + (C_mol2_free_se / C_mol2_free) ** 2)
        KA_list       = [Pstate[0],                             Pstate_se[0]]
        KA_dtype      = [("P unbound", "f4"),                   ("P unbound se", "f4")]
        for i in range(1, Pstate.size):
            KA_list  += [Pstate[i],                             Pstate_se[i]]
            KA_dtype += [("P {0} bound".format(i), "f4"),       ("P {0} bound se".format(i), "f4")]
        KA_list      += [C_mol1_free,                           C_mol1_free_se]
        KA_dtype     += [("[mol1 free]", "f4"),                 ("[mol1 free] se", "f4")]
        KA_list      += [C_mol2_free,                           C_mol2_free_se]
        KA_dtype     += [("[mol2 free]", "f4"),                 ("[mol2 free] se", "f4")]
        for i in range(C_bound.size):
            KA_list  += [C_bound[i],                            C_bound_se[i]]
            KA_dtype += [("[complex {0}]".format(i + 1), "f4"), ("[complex {0}] se".format(i + 1), "f4")]
        KA_list      += [KA[0],                                 KA_se[0]]
        KA_dtype     += [("KA 1", "f4"),                        ("KA 1 se", "f4")]
        for i in range(1, KA.size):
            KA_list  += [KA[i],                                 KA_se[i]]
            KA_dtype += [("KA {0}".format(i + 1), "f4"),        ("KA {0} se".format(i + 1), "f4")]
        KA_array      = np.array([tuple(KA_list)], KA_dtype)
        if True:
            for i in range(KA.size): print "KA {0:02d}: {1:7.5f} ({2:7.5f})".format(i + 1, KA[i], KA_se[i])
        self.datasets += [dict(
          address = "KA",
          data    = KA_array,
          attrs   = {"N molecule 1": self.n_molecule_1,   "N molecule 2":   self.n_molecule_2,
                     "[molecule 1]": C_mol1_total,        "[molecule 2]":   C_mol2_total,
                     "volume":       self.volume})]
            # Store bound and unbound cutoffs


