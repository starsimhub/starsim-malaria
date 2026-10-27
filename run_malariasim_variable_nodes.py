"""
run_malaria_variable_nodes.py — run malaria scenarios with a configurable node count.

Generates nodes with a specified total population distributed uniformly at
random across N nodes.  Each node gets a seasonal EIR and emergence profile that
peaks in the same calendar month, but with a different seasonal shape (width),
so nodes look epidemiologically distinct while remaining in a comparable
prevalence range.

Edit the PARAMS section below to change the defaults, or call
``run_variable_nodes()`` directly with keyword arguments.

Usage
-----
    python run_malaria_variable_nodes.py
"""

import numpy as np
import sciris as sc
import starsim as ss
import malariasim as ms


# -----------------------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------------------

TOTAL_POP  = 10_000   # total human population across all nodes
N_NODES    = 1        # number of spatial nodes
PEAK_MONTH = 6        # 0-based month index of transmission peak (6 = July)
GRID_SEED  = 42       # RNG seed used only for node-grid generation
START      = 2000
STOP       = 2002

# All nodes share a common target mean EIR so their steady-state prevalence
# is comparable.  A ±VARIATION fraction of random per-node scatter is added.
MEAN_EIR_TARGET       = 10.0   # annualised bites/person/year, averaged over months
MEAN_EIR_VARIATION    = 0.20   # ±fraction of random scatter across nodes

# Emergence (adult females/human/day, mean over the year).
MEAN_EMERGE_TARGET    = 0.020  # females/human/day, mean over months
MEAN_EMERGE_VARIATION = 0.20   # ±fraction of random scatter across nodes

# Indoor bite fraction drawn uniformly per node from this range.
F_INDOOR_RANGE = (0.55, 0.85)

# Seasonal width (sigma of the wrapped Gaussian, in months).
# Smaller → sharper, more concentrated transmission peak.
# Larger  → flatter, more spread-out season.
SIGMA_RANGE = (1.5, 4.0)


# -----------------------------------------------------------------------
# Demographics — identical to run_malaria.py
# -----------------------------------------------------------------------

AGE_BIN_EDGES     = ms.AGE_BIN_EDGES
AGE_BIN_FRACTIONS = ms.AGE_BIN_FRACTIONS
AGE_DATA          = ms.make_age_data()


# -----------------------------------------------------------------------
# Node-grid generation helpers
# -----------------------------------------------------------------------

def _wrapped_gaussian(peak_month, sigma):
    """
    Return a length-12 array with a wrapped-Gaussian shape.

    The curve is centred on ``peak_month`` (0 = January) with width
    ``sigma`` (months).  Wrap-around is handled by summing contributions
    from the adjacent years.  The array is normalised to mean 1 so it can
    be scaled to any desired level independently.

    Parameters
    ----------
    peak_month : int   — 0-based calendar-month index of the peak
    sigma      : float — seasonal width in months
    """
    months  = np.arange(12, dtype=float)
    profile = np.zeros(12)
    for k in (-12, 0, 12):   # handle year wrap-around
        profile += np.exp(-0.5 * ((months - peak_month + k) / sigma) ** 2)
    return profile / profile.mean()


def make_node_grid(total_pop, n_nodes, peak_month=PEAK_MONTH, seed=GRID_SEED):
    """
    Build *n_nodes* nodes: an ``ss.geo.geography``, and the parameters of each node
    (``eir``, ``emergence``, ``f_indoor``) for ``ms.Sim``.

    Population
        A symmetric Dirichlet draw (uniform over the probability simplex)
        partitions ``total_pop`` among nodes, then values are rounded to
        integers with the largest node absorbing the rounding residual.

    Seasonal profiles
        Each node gets a wrapped-Gaussian seasonal curve whose peak falls in
        ``peak_month``.  The seasonal width (sigma) is drawn independently
        per node from ``SIGMA_RANGE``, so each node has a distinct shape
        while sharing the same peak month.  The mean level of both EIR and
        emergence is drawn within ±VARIATION of the shared target, keeping
        all nodes in a comparable prevalence range.

    Parameters
    ----------
    total_pop  : int — total human population
    n_nodes    : int — number of spatial nodes
    peak_month : int — 0-based calendar-month index of the transmission peak
    seed       : int — RNG seed (affects only the node-grid layout)
    """
    rng = np.random.default_rng(seed)

    # --- populations: uniform random partition of total_pop ------------
    fracs = rng.dirichlet(np.ones(n_nodes))
    pops  = np.maximum(1, np.round(fracs * total_pop).astype(int))
    pops[np.argmax(fracs)] += total_pop - pops.sum()   # absorb rounding

    # --- per-node shape and level parameters ---------------------------
    sigmas    = rng.uniform(*SIGMA_RANGE,    size=n_nodes)
    eir_means = MEAN_EIR_TARGET    * (1 + rng.uniform(-MEAN_EIR_VARIATION,    MEAN_EIR_VARIATION,    size=n_nodes))
    em_means  = MEAN_EMERGE_TARGET * (1 + rng.uniform(-MEAN_EMERGE_VARIATION, MEAN_EMERGE_VARIATION, size=n_nodes))
    f_indoors = rng.uniform(*F_INDOOR_RANGE, size=n_nodes)

    # --- build per-node seasonal profiles ------------------------------
    eir_profiles       = []
    emergence_profiles = []
    for i in range(n_nodes):
        shape = _wrapped_gaussian(peak_month, sigmas[i])
        eir_profiles.append((shape * eir_means[i]).tolist())
        emergence_profiles.append((shape * em_means[i]).tolist())

    nodes = ss.geo.geography(names=[f'Node {i}' for i in range(n_nodes)], pop=pops.tolist()) if n_nodes > 1 else pops.tolist()
    if n_nodes == 1: # One value, or one profile, for the one node
        eir_profiles, emergence_profiles, f_indoors = eir_profiles[0], emergence_profiles[0], f_indoors[0]
    return nodes, dict(eir=eir_profiles, emergence=emergence_profiles, f_indoor=np.asarray(f_indoors).tolist())


# -----------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------

def make_scenarios(node_grid):
    """Return the four standard scenarios wired to *node_grid* (the nodes, and their parameters)."""
    nodes, node_pars = node_grid
    return [
        dict(label='SIS + EIR-only',  human='sis',  mosquito='eir_only', nodes=nodes, **node_pars),
        dict(label='SEIS + EIR-only', human='seis', mosquito='eir_only', nodes=nodes, **node_pars),
        dict(label='SIS + agents',    human='sis',  mosquito='agents',   nodes=nodes, **node_pars),
        dict(label='SEIS + agents',   human='seis', mosquito='agents',   nodes=nodes, **node_pars),
    ]


# -----------------------------------------------------------------------
# Build and run
# -----------------------------------------------------------------------

def run_variable_nodes(
    total_pop  = TOTAL_POP,
    n_nodes    = N_NODES,
    peak_month = PEAK_MONTH,
    grid_seed  = GRID_SEED,
    do_monitor = True,
    do_plot    = True,
    show_plots = False,
    do_save    = True,
):
    """Run all four scenarios on a randomly generated N-node spatial grid.

    Parameters
    ----------
    total_pop  : int  — total human population distributed across nodes
    n_nodes    : int  — number of spatial nodes
    peak_month : int  — 0-based month index of transmission peak (6 = July)
    grid_seed  : int  — RNG seed for node-grid layout
    do_monitor : bool — track wall-clock time and RSS memory (requires psutil;
                        forces serial execution)
    do_plot    : bool — whether to plot results
    show_plots : bool — whether to display plots interactively
    do_save    : bool — whether to save plots to disk
    """
    node_grid = make_node_grid(total_pop, n_nodes, peak_month, grid_seed)

    sims = []
    for scen in make_scenarios(node_grid):
        scen  = scen.copy()
        label = scen.pop('label')
        sim   = ms.Sim(
            age_data     = AGE_DATA,
            start        = START,
            stop         = STOP,
            dt           = ss.days(1),
            label        = label,
            rand_seed    = 0,
            demographics = ms.make_steady_state_demographics(),
            **scen,
        )
        sims.append(sim)

    monitor = ms.PerformanceMonitor() if do_monitor else None
    msim    = ms.MalariaMultiSim(sims)

    if do_monitor:
        for sim in msim.sims:
            print(f'Running: {sim.label}')
            monitor.start()
            sim.run()
            monitor.stop()
            monitor.record(sim)
    else:
        msim.run()

    if do_plot:
        msim.plot_all(show_plots=show_plots, do_save=do_save)
        if monitor is not None:
            monitor.plot(show_plots=show_plots, do_save=do_save)

    return msim


if __name__ == '__main__':
    T    = sc.timer()
    msim = run_variable_nodes()
    T.toc()
