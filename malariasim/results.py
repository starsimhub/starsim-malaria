"""
results.py — analyzer and MultiSim subclass for malaria scenario comparisons.

Public API
----------
ResultsAnalyzer
    ss.Analyzer that collects per-age-bin, per-node human counts and
    per-node infected counts, per-node EIR, plus per-node mosquito counts when
    MosquitoesAdultFemaleAgentsOnly is present.

MalariaMultiSim
    ss.MultiSim subclass with malaria-specific plotting methods.
    Call msim.plot_all(show_plotss=False,do_save=True) after running to generate all charts.
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import starsim as ss

from .mosquito_agents import MosquitoesAdultFemaleAgentsOnly
from .mosquito_ode    import MosquitoesODE
from .human_ode       import MalariaHumanODE

_MOSQUITO_MODULES = (MosquitoesAdultFemaleAgentsOnly, MosquitoesODE)


NODE_COLORS = ['steelblue', 'darkorange', 'seagreen', 'firebrick', 'purple', 'brown']


def _plain_counts(ax):
    """Format y-axis as plain integers with comma thousands separator (no sci notation)."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))


# ------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------

class ResultsAnalyzer(ss.Analyzer):
    """
    Record alive-agent counts per 5-year age bin (0-5, 5-10, …, 95-100)
    plus a Total, at every timestep.  Also records per-node human population,
    infected counts, per-node EIR, and (when agent mosquitoes are present)
    per-node mosquito counts.  No 100+ bin is created: the hard age cap set by
    ``ss.Deaths`` at ``max_age=100`` ensures no agent legitimately exceeds 100
    years old.

    Results are stored in:
        self.hist                — dict[label] -> np.ndarray
        self.node_hist           — dict[node_label] -> np.ndarray
        self.node_infected_hist  — dict[node_label] -> np.ndarray
        self.node_eir_hist       — dict[node_label] -> np.ndarray  (bites/person/year)
        self.node_mosquito_hist  — dict[node_label] -> np.ndarray  (agents/ODE only)
    """

    # Standard 5-year bins [0,5), [5,10), …, [95,100)
    BINS = [(lo, lo + 5) for lo in range(0, 100, 5)]

    def __init__(self):
        super().__init__()
        self.labels               = [f'{lo}-{hi}' for lo, hi in self.BINS] + ['Total']
        self.hist                 = {lbl: [] for lbl in self.labels}
        self.node_hist            = {}   # populated lazily on first step; keyed by node label
        self.node_infected_hist   = {}   # infected count per node per step
        self.node_eir_hist        = {}   # EIR per node per step (bites/person/year)
        self.node_mosquito_hist   = {}   # mosquito count per node per step (agents/ODE only)
        self.ode_age_hist         = {}   # ODE age-bin totals; keyed by bin label (e.g. '0-5')
        self.births_hist          = []   # births per step
        self.deaths_hist          = []   # deaths per step

    def step(self):
        disease = self.sim.diseases[0]

        # ODE human manages its own compartments — skip the agent-based age bins.
        if isinstance(disease, MalariaHumanODE):
            n_bins  = disease._n_bins
            n_nodes = disease._Sh.raw.size // n_bins
            Sh_mat  = disease._Sh.raw.reshape(n_nodes, n_bins)
            Ih_mat  = disease._Ih.raw.reshape(n_nodes, n_bins)
            n_ode   = int(round((Sh_mat + Ih_mat).sum()))
            for lbl in self.labels[:-1]:  # 20 five-year age bins
                self.hist[lbl].append(0)
            self.hist['Total'].append(n_ode)

            # Record per-age-bin totals if ODE has age structure.
            if disease._age_bin_edges is not None:
                edges = disease._age_bin_edges
                if not self.ode_age_hist:
                    for i in range(len(edges) - 1):
                        lbl = f'{int(edges[i])}-{int(edges[i+1])}'
                        self.ode_age_hist[lbl] = []
                bin_totals = (Sh_mat + Ih_mat).sum(axis=0)  # shape (n_bins,)
                for lbl, val in zip(self.ode_age_hist, bin_totals):
                    self.ode_age_hist[lbl].append(float(val))
        else:
            people  = self.sim.people
            alive   = np.asarray(people.alive.values, dtype=bool)
            age     = np.asarray(people.age.values,   dtype=float)

            # bin_idx 0-19 maps to the twenty 5-year bins [0-5)…[95-100).
            # Any agent transiently past 100 (mid-step timing) is folded
            # into the last bin (95-100) rather than creating a spurious 100+ bin.
            bin_idx = np.minimum((age[alive] / 5.0).astype(int), 19)
            counts  = np.bincount(bin_idx, minlength=20)
            for i, lbl in enumerate(self.labels[:-1]):   # 20 five-year bins
                self.hist[lbl].append(int(counts[i]))
            self.hist['Total'].append(int(alive.sum()))

        # --- per-node human and infected counts ---
        disease = self.sim.diseases[0]
        ns      = disease._get_node_state()
        n_nodes = ns.grid.n_nodes
        if not self.node_hist:
            for lbl in ns.grid.labels:
                self.node_hist[lbl]          = []
                self.node_infected_hist[lbl] = []
        if isinstance(disease, MalariaHumanODE):
            n_bins_nd      = disease._n_bins
            n_nodes_nd     = disease._Sh.raw.size // n_bins_nd
            Sh_nd          = disease._Sh.raw.reshape(n_nodes_nd, n_bins_nd)
            Ih_nd          = disease._Ih.raw.reshape(n_nodes_nd, n_bins_nd)
            n_per_node     = (Sh_nd + Ih_nd).sum(axis=1)  # (n_nodes,)
            n_inf_per_node = Ih_nd.sum(axis=1)
        else:
            node_id  = np.asarray(disease.node_id.values, dtype=int)
            infected = np.asarray(disease.infected.values, dtype=bool)
            n_per_node     = np.bincount(node_id[alive],            minlength=n_nodes).astype(float)
            n_inf_per_node = np.bincount(node_id[alive & infected], minlength=n_nodes).astype(float)
        for nid, lbl in enumerate(ns.grid.labels):
            self.node_hist[lbl].append(float(n_per_node[nid]))
            self.node_infected_hist[lbl].append(float(n_inf_per_node[nid]))

        # --- per-node EIR (annualized) ---
        if not self.node_eir_hist:
            for lbl in ns.grid.labels:
                self.node_eir_hist[lbl] = []
        dt_years = self.sim.t.dt_year
        for nid, lbl in enumerate(ns.grid.labels):
            eir_per_step = float(ns.eir_indoor[nid] + ns.eir_outdoor[nid])
            self.node_eir_hist[lbl].append(eir_per_step / dt_years)

        # --- births and deaths per step ---
        n_births = 0
        n_deaths = 0
        for mod in self.sim.demographics.values():
            if hasattr(mod, 'n_births_this_step'):
                n_births += mod.n_births_this_step
            if hasattr(mod, 'n_deaths'):
                n_deaths += mod.n_deaths
        self.births_hist.append(n_births)
        self.deaths_hist.append(n_deaths)

        # --- per-node mosquito counts (ABM or ODE mode) ---
        mos_mod = next(
            (m for m in self.sim.modules if isinstance(m, _MOSQUITO_MODULES)),
            None,
        )
        if mos_mod is not None:
            if not self.node_mosquito_hist:
                for lbl in ns.grid.labels:
                    self.node_mosquito_hist[lbl] = []
            if isinstance(mos_mod, MosquitoesAdultFemaleAgentsOnly):
                n_mos_per_node = np.bincount(mos_mod.node_id[mos_mod.alive], minlength=n_nodes)
            else:
                # ODE: per-node total from compartment arrays
                n_mos_per_node = mos_mod._Sm + mos_mod._Em + mos_mod._Im
            for nid, lbl in enumerate(ns.grid.labels):
                self.node_mosquito_hist[lbl].append(float(n_mos_per_node[nid]))

    def finalize_results(self):
        super().finalize_results()
        for lbl in self.labels:
            self.hist[lbl] = np.array(self.hist[lbl])
        for lbl in self.node_hist:
            self.node_hist[lbl]          = np.array(self.node_hist[lbl])
            self.node_infected_hist[lbl] = np.array(self.node_infected_hist[lbl])
        for lbl in self.node_eir_hist:
            self.node_eir_hist[lbl] = np.array(self.node_eir_hist[lbl])
        for lbl in self.node_mosquito_hist:
            self.node_mosquito_hist[lbl] = np.array(self.node_mosquito_hist[lbl])
        for lbl in self.ode_age_hist:
            self.ode_age_hist[lbl] = np.array(self.ode_age_hist[lbl])
        self.births_hist = np.array(self.births_hist)
        self.deaths_hist = np.array(self.deaths_hist)


# ------------------------------------------------------------------
# MalariaMultiSim
# ------------------------------------------------------------------

class MalariaMultiSim(ss.MultiSim):
    """
    ss.MultiSim subclass with malaria-specific plotting.

    Stores an optional PerformanceMonitor and exposes plot_all() to
    generate all output charts and performance summaries.
    """

    def __init__(self, sims=None, monitor=None, **kwargs):
        super().__init__(sims=sims, **kwargs)
        self.monitor = monitor

    # -------------------------------------------------------------
    # -- helpers
    # -------------------------------------------------------------

    def _save(self, fname, show_plots=False, do_save=True, dpi=150):
        plt.tight_layout()
        if do_save:
            plt.savefig(fname, dpi=dpi)
            print(f'Plot saved to {fname}')
        if show_plots:
            plt.show()
        plt.close()

    # -------------------------------------------------------------
    # -- individual plot methods
    # -------------------------------------------------------------

    def plot_prevalence(self, show_plots=False, do_save=True, prefix=''):
        """Grid: total and per-node prevalence for each sim."""
        nrows, ncols = sc.getrowscols(len(self.sims))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        fig.suptitle('Malaria prevalence — model comparison', fontsize=13)

        for ax, sim in zip(axes.flat, self.sims):
            disease_key = type(sim.diseases[0]).__name__.lower()
            res         = sim.results[disease_key]
            timevec     = res.n_infected.timevec.years

            if isinstance(sim.diseases[0], MalariaHumanODE):
                n_total = res.n_susceptible + res.n_infected
            else:
                n_total = sim.results.n_alive
            n_total = np.where(n_total > 0, n_total, 1.0)

            ax.plot(timevec, res.n_infected / n_total * 100, lw=2, color='black', label='Total')

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

        for ax in axes.flat[len(self.sims):]:
            ax.set_visible(False)

        self._save(f'{prefix}malaria_prevalence.png', show_plots=show_plots, do_save=do_save)

    def plot_compartments(self, show_plots=False, do_save=True, prefix=''):
        """Grid: S (E) I fractions for each sim."""
        nrows, ncols = sc.getrowscols(len(self.sims))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        fig.suptitle('Malaria compartment fractions', fontsize=13)

        for ax, sim in zip(axes.flat, self.sims):
            disease_key = type(sim.diseases[0]).__name__.lower()
            res         = sim.results[disease_key]
            timevec     = res.timevec.years

            # ODE human: compartment sum is the correct denominator (fixed population).
            # ABM human: use n_alive — compartment flags on dead agents may not be cleared.
            if isinstance(sim.diseases[0], MalariaHumanODE):
                n_total = res.n_susceptible + res.n_infected
                if 'n_exposed' in res:
                    n_total = n_total + res.n_exposed
            else:
                n_total = sim.results.n_alive
            n_total = np.where(n_total > 0, n_total, 1.0)

            ax.plot(timevec, res.n_susceptible / n_total, label='Susceptible', color='steelblue',  lw=1.5)
            if 'n_exposed' in res:
                ax.plot(timevec, res.n_exposed / n_total, label='Exposed',     color='darkorange', lw=1.5)
            ax.plot(timevec, res.n_infected   / n_total, label='Infectious',   color='firebrick',  lw=1.5)

            ax.set_title(sim.label)
            ax.set_xlabel('Year')
            ax.set_ylabel('Fraction of population')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(self.sims):]:
            ax.set_visible(False)

        self._save(f'{prefix}malaria_compartments.png', show_plots=show_plots, do_save=do_save)

    def plot_mosquito_sei(self, show_plots=False, do_save=True, prefix=''):
        """Grid: mosquito SEI compartments for each sim (EIR-only panels show placeholder text)."""
        nrows, ncols = sc.getrowscols(len(self.sims))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
        axes = np.atleast_2d(axes)
        fig.suptitle('Mosquito SEI compartments', fontsize=13)

        for ax, sim in zip(axes.flat, self.sims):
            mos_mod = next(
                (m for m in sim.modules if isinstance(m, _MOSQUITO_MODULES)),
                None,
            )
            ax.set_title(sim.label)
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of mosquitoes')
            ax.grid(True, alpha=0.3)
            _plain_counts(ax)

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

        for ax in axes.flat[len(self.sims):]:
            ax.set_visible(False)

        self._save(f'{prefix}malaria_mosquitoes_compartments.png', show_plots=show_plots, do_save=do_save)

    def plot_mosquito_by_node(self, show_plots=False, do_save=True, prefix=''):
        """Grid: mosquito count per node for each sim (EIR-only panels show placeholder text)."""
        nrows, ncols = sc.getrowscols(len(self.sims))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        fig.suptitle('Mosquito count by node', fontsize=13)

        for ax, sim in zip(axes.flat, self.sims):
            az = sim.analyzers[0]
            ax.set_title(sim.label)
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of mosquitoes')
            ax.grid(True, alpha=0.3)

            if not az.node_mosquito_hist:
                ax.text(0.5, 0.5, 'EIR-only\n(no mosquito agents)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                continue

            timevec = sim.results.timevec.years
            for (node_lbl, counts), color in zip(az.node_mosquito_hist.items(), NODE_COLORS):
                ax.plot(timevec, counts, lw=1.5, color=color, label=node_lbl)
            ax.legend(fontsize=8)
            _plain_counts(ax)

        for ax in axes.flat[len(self.sims):]:
            ax.set_visible(False)

        self._save(f'{prefix}malaria_mosquitoes_by_node.png', show_plots=show_plots, do_save=do_save)

    def plot_eir_by_node(self, show_plots=False, do_save=True, prefix=''):
        """Grid: annualized EIR per node over time for each sim."""
        nrows, ncols = sc.getrowscols(len(self.sims))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
        axes = np.atleast_2d(axes)
        fig.suptitle('EIR by node', fontsize=13)

        for ax, sim in zip(axes.flat, self.sims):
            az      = sim.analyzers[0]
            timevec = sim.results.timevec.years
            ax.set_title(sim.label)
            ax.set_xlabel('Year')
            ax.set_ylabel('EIR (bites/person/year)')
            ax.grid(True, alpha=0.3)
            for (node_lbl, eir), color in zip(az.node_eir_hist.items(), NODE_COLORS):
                ax.plot(timevec, eir, lw=1.5, color=color, label=node_lbl)
            ax.legend(fontsize=8)

        for ax in axes.flat[len(self.sims):]:
            ax.set_visible(False)

        self._save(f'{prefix}malaria_eir_by_node.png', show_plots=show_plots, do_save=do_save)

    def plot_age_bins(self, show_plots=False, do_save=True, prefix=''):
        """One figure per sim: population by 5-year age bin + per-node totals."""
        for sim in self.sims:
            az      = sim.analyzers[0]
            timevec = sim.results.timevec.years
            is_ode_human = isinstance(sim.diseases[0], MalariaHumanODE)

            if is_ode_human and az.ode_age_hist:
                # ODE has age structure — plot per-bin totals.
                n_plots  = len(az.ode_age_hist) + 1   # bins + total
                NROWS, NCOLS = sc.getrowscols(n_plots)
                fig, axes = plt.subplots(NROWS, NCOLS, figsize=(4 * NCOLS, 4 * NROWS), sharex=True)
                axes = np.atleast_2d(axes)
                fig.suptitle(f'Population by age bin (ODE) — {sim.label}', fontsize=14)

                for ax, (lbl, counts) in zip(axes.flat, az.ode_age_hist.items()):
                    ax.plot(timevec, counts, lw=1.2, color='steelblue')
                    ax.set_title(lbl, fontsize=9)
                    ax.set_ylabel('Count', fontsize=7)
                    ax.set_xlabel('Year', fontsize=7)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
                    _plain_counts(ax)

                # Last slot: total population
                ax_tot = axes.flat[len(az.ode_age_hist)]
                ax_tot.plot(timevec, az.hist['Total'], lw=1.5, color='firebrick')
                ax_tot.set_title('Total', fontsize=9)
                ax_tot.set_ylabel('Count', fontsize=7)
                ax_tot.set_xlabel('Year', fontsize=7)
                ax_tot.tick_params(labelsize=7)
                ax_tot.grid(True, alpha=0.3)
                _plain_counts(ax_tot)

                for ax in axes.flat[n_plots:]:
                    ax.set_visible(False)

            elif is_ode_human:
                # ODE with no age structure — show total only.
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.suptitle(f'Population — {sim.label}', fontsize=14)
                ax.plot(timevec, az.hist['Total'], lw=1.5, color='steelblue', label='Total (ODE)')
                ax.set_title('ODE human: no age structure\n(fixed compartment population)', fontsize=10)
                ax.set_ylabel('Count')
                ax.set_xlabel('Year')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                _plain_counts(ax)
            else:
                n_plots = len(az.labels) + (1 if az.node_hist else 0) + 2
                NROWS, NCOLS = sc.getrowscols(n_plots)

                fig, axes = plt.subplots(NROWS, NCOLS, figsize=(4 * NCOLS, 4 * NROWS), sharex=True)
                axes = np.atleast_2d(axes)
                fig.suptitle(f'Population by 5-year age bin — {sim.label}', fontsize=14)

                for ax, lbl in zip(axes.flat, az.labels):
                    ax.plot(timevec, az.hist[lbl], lw=1.2, color='steelblue')
                    ax.set_title(lbl, fontsize=9)
                    ax.set_ylabel('Count', fontsize=7)
                    ax.set_xlabel('Year', fontsize=7)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
                    _plain_counts(ax)

                # Next slot: per-node population totals
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
                    _plain_counts(ax_node)
                    hidden_start = len(az.labels) + 1
                else:
                    hidden_start = len(az.labels)

                ax_births = axes.flat[hidden_start]
                ax_births.plot(timevec, az.births_hist, lw=1.2, color='steelblue')
                ax_births.set_title('Births / step', fontsize=9)
                ax_births.set_ylabel('Count', fontsize=7)
                ax_births.set_xlabel('Year', fontsize=7)
                ax_births.tick_params(labelsize=7)
                ax_births.grid(True, alpha=0.3)
                _plain_counts(ax_births)

                ax_deaths = axes.flat[hidden_start + 1]
                ax_deaths.plot(timevec, az.deaths_hist, lw=1.2, color='firebrick')
                ax_deaths.set_title('Deaths / step', fontsize=9)
                ax_deaths.set_ylabel('Count', fontsize=7)
                ax_deaths.set_xlabel('Year', fontsize=7)
                ax_deaths.tick_params(labelsize=7)
                ax_deaths.grid(True, alpha=0.3)
                _plain_counts(ax_deaths)

                hidden_start += 2

                for ax in axes.flat[hidden_start:]:
                    ax.set_visible(False)

            self._save(f'{prefix}malaria_age_bins_{sc.sanitizefilename(sim.label)}.png',
                         show_plots=show_plots,
                         do_save=do_save)

    # -------------------------------------------------------------
    # -- entry point
    # -------------------------------------------------------------

    def plot_all(self, show_plots=False, do_pop_plot=False, do_save=True, prefix='',
                 do_compartments=False, do_mosquito_compartments=False, do_mosquito_by_node=True):
        """Generate all output charts and performance summaries."""
        self.plot_prevalence(show_plots=show_plots, do_save=do_save, prefix=prefix)
        if do_compartments:
            self.plot_compartments(show_plots=show_plots, do_save=do_save, prefix=prefix)
        if do_mosquito_compartments:
            self.plot_mosquito_sei(show_plots=show_plots, do_save=do_save, prefix=prefix)
        if do_mosquito_by_node:
            self.plot_mosquito_by_node(show_plots=show_plots, do_save=do_save, prefix=prefix)
        self.plot_eir_by_node(show_plots=show_plots, do_save=do_save, prefix=prefix)
        if do_pop_plot:
            self.plot_age_bins(show_plots=show_plots, do_save=do_save, prefix=prefix)
        if self.monitor is not None:
            self.monitor.plot(out_prefix=prefix + 'perf', show_plots=show_plots, do_save=do_save)
