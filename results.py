"""
results.py — analyzer and plotting helpers for malaria scenario comparisons.

Public API
----------
AgeBinPopulation
    ss.Analyzer that collects per-age-bin, per-node human counts and
    per-node infected counts, plus per-node mosquito counts when
    MosquitoesAdultFemaleAgentsOnly is present.

plot_all(sims, monitor)
    Generate and save every output chart and performance summary for a list
    of completed simulations.  Intended to be called as a single line after
    the run loop in run_malaria.py.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss

from malaria import MosquitoesAdultFemaleAgentsOnly


NODE_COLORS = ['steelblue', 'darkorange', 'seagreen', 'firebrick', 'purple', 'brown']


# ------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------

class AgeBinPopulation(ss.Analyzer):
    """
    Record alive-agent counts per 5-year age bin, plus 100+ and total,
    at every timestep.  Also records per-node human population, infected
    counts, and (when agent mosquitoes are present) per-node mosquito counts.

    Results are stored in:
        self.hist                — dict[label] -> np.ndarray
        self.node_hist           — dict[node_label] -> np.ndarray
        self.node_infected_hist  — dict[node_label] -> np.ndarray
        self.node_mosquito_hist  — dict[node_label] -> np.ndarray  (agents only)
    """

    # Standard 5-year bins [0,5), [5,10), …, [95,100)
    BINS = [(lo, lo + 5) for lo in range(0, 100, 5)]

    def __init__(self):
        super().__init__()
        self.labels               = [f'{lo}-{hi}' for lo, hi in self.BINS] + ['100+', 'Total']
        self.hist                 = {lbl: [] for lbl in self.labels}
        self.node_hist            = {}   # populated lazily on first step; keyed by node label
        self.node_infected_hist   = {}   # infected count per node per step
        self.node_mosquito_hist   = {}   # mosquito count per node per step (agents mode only)

    def step(self):
        people = self.sim.people
        alive  = np.asarray(people.alive.values, dtype=bool)
        age    = np.asarray(people.age.values,   dtype=float)

        for (lo, hi), lbl in zip(self.BINS, self.labels):
            mask = alive & (age >= lo) & (age < hi)
            self.hist[lbl].append(int(mask.sum()))

        self.hist['100+'].append(int((alive & (age >= 100)).sum()))
        self.hist['Total'].append(int(alive.sum()))

        # Per-node human population and infected counts
        disease  = self.sim.diseases[0]
        ns       = disease._get_node_state()
        node_id  = np.asarray(disease.node_id.values, dtype=int)
        infected = np.asarray(disease.infected.values, dtype=bool)
        if not self.node_hist:
            for lbl in ns.grid.labels:
                self.node_hist[lbl]          = []
                self.node_infected_hist[lbl] = []
        for nid, lbl in enumerate(ns.grid.labels):
            in_node = alive & (node_id == nid)
            self.node_hist[lbl].append(int(in_node.sum()))
            self.node_infected_hist[lbl].append(int((in_node & infected).sum()))

        # Per-node mosquito counts (only when agent-based mosquitoes are present)
        mos_mod = next(
            (m for m in self.sim.modules if isinstance(m, MosquitoesAdultFemaleAgentsOnly)),
            None,
        )
        if mos_mod is not None:
            if not self.node_mosquito_hist:
                for lbl in ns.grid.labels:
                    self.node_mosquito_hist[lbl] = []
            for nid, lbl in enumerate(ns.grid.labels):
                self.node_mosquito_hist[lbl].append(
                    int((mos_mod.alive & (mos_mod.node_id == nid)).sum())
                )

    def finalize_results(self):
        super().finalize_results()
        for lbl in self.labels:
            self.hist[lbl] = np.array(self.hist[lbl])
        for lbl in self.node_hist:
            self.node_hist[lbl]          = np.array(self.node_hist[lbl])
            self.node_infected_hist[lbl] = np.array(self.node_infected_hist[lbl])
        for lbl in self.node_mosquito_hist:
            self.node_mosquito_hist[lbl] = np.array(self.node_mosquito_hist[lbl])


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def _timevec(sim):
    disease_key = type(sim.diseases[0]).__name__.lower()
    return sim.results[disease_key].timevec.years


def _slug(label):
    return label.lower().replace(' ', '_').replace('+', 'plus').replace('-', '').strip('_')


def _save(fname, dpi=150):
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    plt.show()
    print(f'Plot saved to {fname}')


def plot_prevalence(sims):
    """2×2 grid: total and per-node prevalence for each sim."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('Malaria prevalence — model comparison', fontsize=13)

    for ax, sim in zip(axes.flat, sims):
        disease_key = type(sim.diseases[0]).__name__.lower()
        res         = sim.results[disease_key]
        n_alive     = sim.results.n_alive
        timevec     = res.n_infected.timevec.years

        ax.plot(timevec, res.n_infected / n_alive * 100, lw=2, color='black', label='Total')

        az = sim.analyzers[0]
        for (node_lbl, n_inf), color in zip(az.node_infected_hist.items(), NODE_COLORS):
            n_alive_node = az.node_hist[node_lbl]
            with np.errstate(invalid='ignore', divide='ignore'):
                node_prev = np.where(n_alive_node > 0, n_inf / n_alive_node * 100, 0.0)
            ax.plot(timevec, node_prev, lw=1.2, color=color, linestyle='--', label=node_lbl)

        ax.set_title(sim.label)
        ax.set_ylabel('Prevalence (%)')
        ax.set_xlabel('Year')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _save('malaria_prevalence.png')


def plot_compartments(sims):
    """2×2 grid: S (E) I fractions for each sim."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('Malaria compartment fractions', fontsize=13)

    for ax, sim in zip(axes.flat, sims):
        disease_key = type(sim.diseases[0]).__name__.lower()
        res         = sim.results[disease_key]
        n_alive     = sim.results.n_alive
        timevec     = res.timevec.years

        ax.plot(timevec, res.n_susceptible / n_alive, label='Susceptible', color='steelblue',  lw=1.5)
        if 'n_exposed' in res:
            ax.plot(timevec, res.n_exposed / n_alive, label='Exposed',     color='darkorange', lw=1.5)
        ax.plot(timevec, res.n_infected   / n_alive, label='Infectious',   color='firebrick',  lw=1.5)

        ax.set_title(sim.label)
        ax.set_xlabel('Year')
        ax.set_ylabel('Fraction of population')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _save('malaria_compartments.png')


def plot_mosquito_sei(sims):
    """2×2 grid: mosquito SEI compartments (EIR-only panels show placeholder text)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle('Mosquito SEI compartments', fontsize=13)

    for ax, sim in zip(axes.flat, sims):
        mos_mod = next(
            (m for m in sim.modules if isinstance(m, MosquitoesAdultFemaleAgentsOnly)),
            None,
        )
        ax.set_title(sim.label)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of mosquitoes')
        ax.grid(True, alpha=0.3)

        if mos_mod is None:
            ax.text(0.5, 0.5, 'EIR-only\n(no mosquito agents)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            continue

        res     = sim.results[type(mos_mod).__name__.lower()]
        timevec = res.timevec.years
        ax.plot(timevec, res.n_total,      label='Total',      color='dimgray',    lw=1.5)
        ax.plot(timevec, res.n_exposed,    label='Exposed',    color='darkorange', lw=1.5)
        ax.plot(timevec, res.n_infectious, label='Infectious', color='firebrick',  lw=1.5)
        ax.legend(fontsize=8)

    _save('malaria_mosquitoes.png')


def plot_mosquito_by_node(sims):
    """One chart per agents-mode sim: mosquito count per node."""
    agents_sims = [s for s in sims if s.analyzers[0].node_mosquito_hist]
    if not agents_sims:
        return

    fig, axes = plt.subplots(1, len(agents_sims),
                             figsize=(6 * len(agents_sims), 5),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle('Mosquito count by node', fontsize=13)

    for ax, sim in zip(axes.flat, agents_sims):
        az      = sim.analyzers[0]
        timevec = _timevec(sim)
        for (node_lbl, counts), color in zip(az.node_mosquito_hist.items(), NODE_COLORS):
            ax.plot(timevec, counts, lw=1.5, color=color, label=node_lbl)
        ax.set_title(sim.label)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of mosquitoes')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _save('malaria_mosquitoes_by_node.png')


def plot_age_bins(sims):
    """One 4×6 figure per sim: population by 5-year age bin + per-node totals."""
    NCOLS, NROWS = 6, 4   # 24 slots; 22 age-bin + 1 per-node + 1 hidden

    for sim in sims:
        az      = sim.analyzers[0]
        timevec = _timevec(sim)

        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 14), sharex=True)
        fig.suptitle(f'Population by 5-year age bin — {sim.label}', fontsize=14)

        for ax, lbl in zip(axes.flat, az.labels):
            ax.plot(timevec, az.hist[lbl], lw=1.2, color='steelblue')
            ax.set_title(lbl, fontsize=9)
            ax.set_ylabel('Count', fontsize=7)
            ax.set_xlabel('Year', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        # 23rd slot: per-node population totals (one line per node)
        if az.node_hist:
            ax_node = axes.flat[len(az.labels)]
            for (node_lbl, counts), color in zip(az.node_hist.items(), NODE_COLORS):
                ax_node.plot(timevec, counts, lw=1.2, color=color, label=node_lbl)
            ax_node.set_title('By node', fontsize=9)
            ax_node.set_ylabel('Count', fontsize=7)
            ax_node.set_xlabel('Year', fontsize=7)
            ax_node.tick_params(labelsize=7)
            ax_node.legend(fontsize=6)
            ax_node.grid(True, alpha=0.3)
            hidden_start = len(az.labels) + 1
        else:
            hidden_start = len(az.labels)

        for ax in axes.flat[hidden_start:]:
            ax.set_visible(False)

        _save(f'malaria_age_bins_{_slug(sim.label)}.png')


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def plot_all(sims, monitor):
    """Generate and save all output charts and performance summaries."""
    plot_prevalence(sims)
    plot_compartments(sims)
    plot_mosquito_sei(sims)
    plot_mosquito_by_node(sims)
    plot_age_bins(sims)
    monitor.plot()
