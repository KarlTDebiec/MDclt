#!/usr/bin/python
#   MDclt.secondary.Association.py
#   Written by Karl Debiec on 12-08-15, last updated by Karl Debiec on 14-07-17
"""
Classes for analysis of molecular association

.. todo:
    - Support ignoring portion of dataset
"""
################################### MODULES ####################################
import os, sys
import numpy as np
from MDclt import secondary
from MDclt.secondary import Block_Generator
from MDclt import Block, Block_Accumulator, Block_Acceptor
from MDclt import  pool_director
################################## FUNCTIONS ###################################
def concentration(n, volume):
    """
    """
    return float(n) / 6.0221415e23 / (volume * 1e-27)

def block_average(data, func = np.mean, func_kwargs = {"axis": 1},
                  min_size = 1, **kwargs):
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

def add_parser(subparsers, *args, **kwargs):
    """
    Adds subparser for this analysis to a nascent argument parser

    **Arguments:**
        :*subparsers*: argparse subparsers object to add subparser
        :*args*:       Passed to subparsers.add_parser(...)
        :*kwargs*:     Passed to subparsers.add_parser(...)
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

    arg_groups["output"].add_argument(
      "-output",
      type     = str,
      required = True,
      metavar  = "H5_FILE",
      help     = "H5 file in which to output data")

    subparser.set_defaults(analysis = command_line)

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
    block_accumulator = Block_Accumulator(
                          preexisting_slice = block_generator.preexisting_slice,
                          incoming_slice    = block_generator.incoming_slice,
                          **kwargs)
    block_accumulator.next()

    # Serial
#    for block in block_generator:
#        block()
#        block_accumulator.send(block)

    # Parallel (Processes)
    pool              = Pool(n_cores)
    for block in pool.imap_unordered(pool_director, block_generator):
        block_accumulator.send(block)
    pool.close()
    pool.join()

    block_accumulator.close()

#    block_acceptor = Block_Acceptor(**kwargs)
#    block_acceptor.next()
#    block_acceptor.send(block_accumulator)
#    block_acceptor.close()

################################### CLASSES ####################################
class Block(Block):
    """
    """
    def __init__(self, coord,
                 bound_cutoff, unbound_cutoff, conc_single,
                 slc, attrs = {},
                 **kwargs):
        """
        """
        from collections import OrderedDict

        # Input
        self.coord = coord
        self.n_frames = self.coord.shape[0]
        self.i = slc[1].stop
        self.j = slc[2].stop

        # Action
        self.bound_cutoff   = bound_cutoff
        self.unbound_cutoff = unbound_cutoff
        self.conc_single    = conc_single

        # Output
        self.datasets = OrderedDict(
          {"pair_association": dict(slc   = slc[1:], attrs = attrs),
           "bound":            dict(slc   = slc[1:], attrs = {}),
           "events":           dict(data  = [],      attrs = {})})

    def __call__(self, **kwargs):
        # Get references to instance variables
        coord    = self.coord
        n_frames = self.n_frames
        bound_cutoff   = self.bound_cutoff
        unbound_cutoff = self.unbound_cutoff
        events = self.datasets["events"]["data"]

        # Assign to bound and unbound states
        bound   = np.zeros(n_frames, np.int8)
        unbound = np.zeros(n_frames, np.int8)
        bound[coord   < bound_cutoff]   = 1
        unbound[coord > unbound_cutoff] = 1

        # Locate transitions
        trans_bound   =   bound[1:] -   bound[:-1]
        trans_unbound = unbound[1:] - unbound[:-1]
        enter_bound   = np.where(trans_bound   == 1)[0] + 1
        enter_unbound = np.where(trans_unbound == 1)[0] + 1

        # Set state to bound between transitions;
        #   CONSIDER ALTERNATIVE OF USING A LIST OF SLICES
        if enter_bound.size >= 1:
            # Start at first entrance of bound state
            enter = enter_bound[0]
            while True:
                try:
                    # Look for next entrance of unbound state
                    exit = enter_unbound[enter_unbound > enter][0]
                    # Next entrance found, set state to bound between entrances
                    bound[enter:exit] = 1
                except:
                    # Trajectory ends with pair in bound state, set state to
                    #   bound until end of trajectory; exit
                    bound[enter::]    = 1
                    break
                try:
                    # Look for next entrance of bound state
                    enter = enter_bound[enter_bound > exit][0]
                    #   Next entrance found; continue in loop
                except:
                    # Trajectory ends with pair in unbound state; exit
                    break

        # Calculate pbound
        pbound = np.sum(bound, dtype = np.float64) / float(bound.size)

        # Calculate standard error of pbound using block averaging
        #   Is there a reasonable way to save/resume this?
        sizes, ses, se_sds = block_average(bound)
        try:
            a, pbound_se, c, d, fit = fit_curve(
              x = sizes, y = ses, sigma = se_sds, fit_func = "sigmoid")
        except:
            pbound_se = pbound

        # Calculate mean first passage time and tabulate binding events
        # Can this be improved?
        trans_bound   = bound[:-1] - bound[1:]
        enter_bound   = np.where(trans_bound  == -1)[0] + 1
        enter_unbound = np.where(trans_bound  ==  1)[0] + 1
        if enter_bound.size >= 1 and enter_unbound.size >= 1:
            if   (enter_bound[0] < enter_unbound[0]
            and   enter_bound[-1] < enter_unbound[-1]):
                # Started unbound, ended unbound
                fpt_on   = enter_bound[1:]   - enter_unbound[:-1]
                fpt_off  = enter_unbound     - enter_bound
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                             for bind, unbind
                                in np.column_stack((enter_bound,
                                                    enter_unbound))] 
            elif (enter_bound[0] < enter_unbound[0]
            and   enter_bound[-1] > enter_unbound[-1]):
                # Started unbound, ended bound
                fpt_on   = enter_bound[1:]   - enter_unbound
                fpt_off  = enter_unbound     - enter_bound[:-1]
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                             for bind, unbind
                               in np.column_stack((enter_bound[:-1],
                                                   enter_unbound))] 
            elif (enter_bound[0] > enter_unbound[0]
            and   enter_bound[-1] < enter_unbound[-1]):
                # Started bound, ended unbound
                fpt_on   = enter_bound       - enter_unbound[:-1]
                fpt_off  = enter_unbound[1:] - enter_bound
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                             for bind, unbind
                               in np.column_stack((enter_bound,
                                                   enter_unbound[1:]))] 
            elif (enter_bound[0] > enter_unbound[0]
            and  enter_bound[-1] > enter_unbound[-1]):
                # Started bound, ended bound
                fpt_on   = enter_bound       - enter_unbound
                fpt_off  = enter_unbound[1:] - enter_bound[:-1]
                events  += [(self.i, self.j, bind, unbind, unbind - bind)
                             for bind, unbind
                               in np.column_stack((enter_bound[:-1],
                                                   enter_unbound[1:]))]

            # Convert mean first passage times into rates
            if fpt_on.size != 0:
                mean_fpt_on    = np.mean(fpt_on)
                mean_fpt_on_se = np.std(fpt_on)  / np.sqrt(fpt_on.size)
                kon_sim    = 1 / mean_fpt_on
                kon_sim_se = kon_sim  * (mean_fpt_on_se  / mean_fpt_on)
                kon        = kon_sim  / (self.conc_single * self.conc_single)
                kon_se     = kon      * (kon_sim_se      / kon_sim)
            else:
                mean_fpt_on,  mean_fpt_on_se = np.nan, np.nan
                kon,          kon_se         = np.nan, np.nan
            if fpt_off.size != 0:
                mean_fpt_off    = np.mean(fpt_off)
                mean_fpt_off_se = np.std(fpt_off) / np.sqrt(fpt_off.size)
                koff_sim    = 1 / mean_fpt_off
                koff_sim_se = koff_sim * (mean_fpt_off_se / mean_fpt_off)
                koff        = koff_sim /  self.conc_single
                koff_se     = koff     * (koff_sim_se     / koff_sim) 
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
        self.datasets["bound"]["bound"]   = bound
        self.datasets["pair_association"]["data"] = np.array(
                  [(pbound,                  pbound_se,
                    mean_fpt_on,             mean_fpt_on_se,
                    mean_fpt_off,            mean_fpt_off,
                    kon,                     kon_se,
                    koff_se,                 koff_se)],
          dtype = [("pbound",       "f4"), ("pbound se",       "f4"),
                   ("mean fpt on",  "f4"), ("mean fpt on se",  "f4"),
                   ("mean fpt off", "f4"), ("mean fpt off se", "f4"),
                   ("kon",          "f4"), ("kon se",          "f4"),
                   ("koff",         "f4"), ("koff se",         "f4")])

class Block_Generator(Block_Generator):
    """
    Generator class that yields blocks of analysis
    """
    def __init__(self,
                 log, coord,
                 bound, unbound,
                 output, force = False, *args, **kwargs):
        """
        Initializes generator

        **Arguments:**
            :*log*:    Simulation log
            :*coord*:  Coordinates used to generate pmf
            :*output*: List including path to h5 file and address
                       within h5 file
            :*force*:  Run analysis even if no new data is present
        """
        import warnings
        from h5py import File as h5

        # Input
        self.log_path,   self.log_address   = log
        self.coord_path, self.coord_address = coord

        # Action
        self.bound   = bound
        self.unbound = unbound
        with h5(self.log_path) as log_h5, h5(self.coord_path) as coord_h5:
            coord_shape      = coord_h5[self.coord_address].shape
            self.volume      = np.mean(log_h5[self.log_address]["volume"])
        self.n_molecule_1    = coord_shape[1]
        self.n_molecule_2    = coord_shape[2]
        self.i               = 0
        self.j               = 0
        self.conc_single     = concentration(1,                 self.volume)
        self.conc_molecule_1 = concentration(self.n_molecule_1, self.volume)
        self.conc_molecule_2 = concentration(self.n_molecule_2, self.volume)

        super(Block_Generator, self).__init__(inputs = [log, coord],
          output = [output, "KA"], *args, **kwargs)

    def next(self):
        """
        Prepares and yields next Block of analysis
        """
        from h5py import File as h5

        if self.i == self.n_molecule_1:
            raise StopIteration()
        else:
            # Load primary data
            with h5(self.coord_path) as coord_h5:
                coord = np.array(coord_h5
                          [self.coord_address]
                          [self.start_index:self.final_index,
                           self.i,
                           self.j])
            block = Block(
                     coord          = coord,
                     bound_cutoff   = self.bound,
                     unbound_cutoff = self.unbound,
                     conc_single    = self.conc_single,
                     slc            = (slice(self.start_index,
                                             self.final_index, 1),
                                       slice(self.i),
                                       slice(self.j)))
            # Iterate and return
            self.j     += 1
            if self.j  == self.n_molecule_2:
                self.i += 1
                self.j  = 0
            return block

class Block_Accumulator(Block_Accumulator):
    """
    """
    def __init__(self, log, coord, **kwargs):
        """
        """
        from h5py import File as h5
        from collections import OrderedDict

        super(Block_Accumulator, self).__init__(**kwargs)
        log_path,   log_address   = log
        coord_path, coord_address = coord
        with h5(log_path) as log_h5, h5(coord_path) as coord_h5:
            # Need to support omitting the beginning and end of trajectories
            coord_shape       = coord_h5[coord_address].shape
            self.n_molecule_1 = coord_shape[1]
            self.n_molecule_2 = coord_shape[2]
            self.volume       = np.mean(log_h5[log_address]["volume"])
        self.conc_molecule_1  = concentration(self.n_molecule_1, self.volume)
        self.conc_molecule_2  = concentration(self.n_molecule_2, self.volume)
        self.count            = np.zeros((coord_shape[0], self.n_molecule_1),
                                  np.int8)

        # Prepare datasets
        self.datasets  = OrderedDict(
          pair_association = dict(
            attrs   = {},
            data    = np.zeros((self.n_molecule_1, self.n_molecule_2),
              dtype = [("pbound",       "f4"), ("pbound se",       "f4"),
                       ("mean fpt on",  "f4"), ("mean fpt on se",  "f4"),
                       ("mean fpt off", "f4"), ("mean fpt off se", "f4"),
                       ("kon",          "f4"), ("kon se",          "f4"),
                       ("koff",         "f4"), ("koff se",         "f4")])),
          overall_association = dict(
            attrs   = {},
            data    = np.zeros(1,
              dtype = [("pbound",       "f4"), ("pbound se",       "f4"),
                       ("mean fpt on",  "f4"), ("mean fpt on se",  "f4"),
                       ("mean fpt off", "f4"), ("mean fpt off se", "f4"),
                       ("kon",          "f4"), ("kon se",          "f4"),
                       ("koff",         "f4"), ("koff se",         "f4")])),
         events = dict(
          attrs   = {},
          data    = []))

    def receive_block(self, **kwargs):
        pair_association = self.datasets["pair_association"]["data"]
        events           = self.datasets["events"]["data"]
        while True:
            block   = (yield)

            events += block.datasets["events"]["data"]
            pair_association[block.i, block.j] = \
              block.datasets["pair_association"]["data"]

            self.count[:, block.i] += block.datasets["bound"]["bound"]

    def close(self, *args, **kwargs):
        self.func.close(*args, **kwargs)

        # Link to datasets for clarity
        pair_assoc   = self.datasets["pair_association"]["data"]
        total_assoc  = self.datasets["overall_association"]["data"]
        events       = self.datasets["events"]["data"]
        count        = self.count
        C_mol1_total = self.conc_molecule_1
        C_mol2_total = self.conc_molecule_2

        # Configure pair and total association datasets
        for key in ["pbound", "mean fpt on", "mean fpt off", "kon", "koff"]:
            pair_assoc_ma          = np.ma.MaskedArray(pair_assoc[key],
                                       np.isnan(pair_assoc[key]))
            pair_assoc_ma_se       = np.ma.MaskedArray(pair_assoc[key+" se"],
                                       np.isnan(pair_assoc[key+" se"]))
            total_assoc[key]       = np.mean(pair_assoc_ma)
            total_assoc[key+" se"] = np.sqrt(
              np.sum(pair_assoc_ma_se ** 2.0)) / pair_assoc_ma.size
        self.datasets["pair_association"]["attrs"].update(
          {"fpt units":  "ps",
           "kon units":  "M-2 ps-1",
           "koff units": "M-1 ps-1"})
        self.datasets["overall_association"]["attrs"].update(
          {"fpt units":  "ps",
           "kon units":  "M-2 ps-1",
           "koff units": "M-1 ps-1"})

        # Configure binding event dataset
        events = np.array(events,
                   dtype = [("index 1",  "i4"), ("index 2",  "i4"),
                            ("start",    "f4"), ("end",     "f4"),
                            ("duration", "f4")])
        self.datasets["events"]["attrs"].update({"duration units": "ps"})

        # Calculate Pstate of different bound states
        n_states  = np.max(count) + 1
        Pstate    = np.zeros(n_states)
        Pstate_se = np.zeros(n_states)
        for i in range(0, n_states):
            Pstate[i]       = float(count[count == i].size) / float(count.size)
            mol1_in_state_i = np.zeros(count.shape, np.int8)
            mol1_in_state_i[count == i] = 1
            total_mol1_in_state_i       = np.sum(mol1_in_state_i,
                                            axis = 1,
                                            dtype = np.float64)

            # Calculate standard error of Pstate using block averaging
            sizes, ses, se_sds = block_average(
              total_mol1_in_state_i / self.n_molecule_1)
            try:
                a, Pstate_se[i], c, d, fit = fit_curve(
                  x = sizes, y = ses, sigma = se_sds, fit_func = "sigmoid")
            except:
                Pstate_se[i] = Pstate[i]
        if True:
            for i in range(Pstate.size):
                print "P state {0:02d}: {1:7.5f} ({2:7.5f})".format(
                  i, Pstate[i], Pstate_se[i])
            print "Sum       : {0:7.5f}".format(np.sum(Pstate))

        # Calculate concentrations
        C_mol1_free    = C_mol1_total * Pstate[0]
        C_mol1_free_se = C_mol1_free  * (Pstate_se[0] / Pstate[0])
        C_mol2_free    = C_mol2_total - C_mol1_total * np.sum(
          [i * P for i, P in enumerate(Pstate[1:], 1)])
        C_mol2_free_se = C_mol1_total * np.sqrt(np.sum(
          [(i * P_se) ** 2 for i, P_se in enumerate(Pstate_se[1:], 1)]))
        C_bound        = C_mol1_total * Pstate[1:]
        C_bound_se     = C_bound      * (Pstate_se[1:] / Pstate[1:])
        np.set_printoptions(precision = 5, suppress = True, linewidth = 120)
        if True:
            print "[mol1 total]:      {0:7.5f}".format(C_mol1_total)
            print "[mol2 total]:      {0:7.5f}".format(C_mol2_total)
            print "[mol1 free] (se):  {0:7.5f} ({1:7.5f})".format(
              C_mol1_free, C_mol1_free_se)
            print "[mol2 free] (se):  {0:7.5f} ({1:7.5f})".format(
              C_mol2_free, C_mol2_free_se)
            for i in range(C_bound.size):
                print "[complex {0:02d}] (se): {1:7.5f} ({2:7.5f})".format(
                  i + 1, C_bound[i],     C_bound_se[i])
            print "Sum [mol1]:        {0:7.5f}".format(
              C_mol1_free + np.sum(C_bound))
            print "Sum [mol2]:        {0:7.5f}".format(
              C_mol2_free + np.sum([i * C for i, C in enumerate(C_bound, 1)]))

        # Calculate KAs
        KA              = np.zeros(n_states - 1)
        KA_se           = np.zeros(n_states - 1)
        KA[0]           = C_bound[0] / (C_mol1_free * C_mol2_free)
        KA_se[0]        = KA[0] * np.sqrt((C_bound_se[0] / C_bound[0]) ** 2
          + (C_mol1_free_se  / C_mol1_free)  ** 2
          + (C_mol2_free_se / C_mol2_free) ** 2)
        for i in range(1, n_states - 1):
            KA[i]     = C_bound[i] / (C_bound[i-1] * C_mol2_free)
            KA_se[i]  = KA[i] * np.sqrt((C_bound_se[i] / C_bound[i]) ** 2
              + (C_bound_se[i-1] / C_bound[i-1]) ** 2
              + (C_mol2_free_se / C_mol2_free) ** 2)
        KA_list       = [Pstate[0],
                         Pstate_se[0]]
        KA_dtype      = [("P unbound",    "f4"),
                         ("P unbound se", "f4")]
        for i in range(1, Pstate.size):
            KA_list  += [Pstate[i],
                         Pstate_se[i]]
            KA_dtype += [("P {0} bound".format(i),    "f4"),
                         ("P {0} bound se".format(i), "f4")]
        KA_list      += [C_mol1_free,
                         C_mol1_free_se]
        KA_dtype     += [("[mol1 free]",    "f4"),
                         ("[mol1 free] se", "f4")]
        KA_list      += [C_mol2_free,
                         C_mol2_free_se]
        KA_dtype     += [("[mol2 free]",    "f4"),
                         ("[mol2 free] se", "f4")]
        for i in range(C_bound.size):
            KA_list  += [C_bound[i],
                         C_bound_se[i]]
            KA_dtype += [("[complex {0}]".format(i + 1),    "f4"),
                         ("[complex {0}] se".format(i + 1), "f4")]
        KA_list      += [KA[0],
                         KA_se[0]]
        KA_dtype     += [("KA 1",    "f4"),
                         ("KA 1 se", "f4")]
        for i in range(1, KA.size):
            KA_list  += [KA[i],
                         KA_se[i]]
            KA_dtype += [("KA {0}".format(i + 1),    "f4"),
                         ("KA {0} se".format(i + 1), "f4")]
        KA_array      = np.array([tuple(KA_list)], KA_dtype)
        if True:
            for i in range(KA.size):
                print "KA {0:02d}: {1:7.5f} ({2:7.5f})".format(
                  i + 1, KA[i], KA_se[i])
        self.datasets["KA"] = dict(
          data  = KA_array,
          attrs = {"N molecule 1": self.n_molecule_1,
                     "N molecule 2": self.n_molecule_2,
                     "[molecule 1]": C_mol1_total,
                     "[molecule 2]": C_mol2_total,
                     "volume":       self.volume})


