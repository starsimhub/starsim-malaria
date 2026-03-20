"""
run_malaria.py — run and compare malaria model configurations.

Runs four scenarios that cover the two human models (SIS, SEIS) crossed with
the two mosquito models (eir_only, agents), then plots prevalence side by side.

Usage
-----
    python run_malaria.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import time
import starsim as ss

from malaria import make_sim, NodeGrid
from malaria import MosquitoesAdultFemaleAgentsOnly



# ------------------------------------------------------------------
# Analyzer: population count per 5-year age bin
# ------------------------------------------------------------------

class AgeBinPopulation(ss.Analyzer):
    """
    Record alive-agent counts per 5-year age bin, plus 100+ and total,
    at every timestep.

    Bins stored in self.hist, keyed by label string.
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

        # Per-node population and infected counts
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
# Demographics: age pyramid + balanced birth/death rates
# ------------------------------------------------------------------

AGE_BIN_EDGES     = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,100]
AGE_BIN_FRACTIONS = [0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.005]

# Expand bin fractions to per-year weights for ss.People age_data
_ages, _vals = [], []
for lo, hi, frac in zip(AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:], AGE_BIN_FRACTIONS):
    per_year = frac / (hi - lo)
    for age in range(lo, hi):
        _ages.append(age)
        _vals.append(per_year)
AGE_DATA = pd.DataFrame({'age': _ages, 'value': _vals})

DEMOGRAPHICS = [
    ss.Births(birth_rate=ss.peryear(20)),
    ss.Deaths(death_rate=ss.peryear(20)),
]

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

N_AGENTS = 10_000
START    = 2000
STOP     = 2002

# ------------------------------------------------------------------
# Three-node spatial configuration
# Node 0 (Urban):      5 000 people — high transmission, mostly indoor biting
# Node 1 (Peri-urban): 3 000 people — moderate transmission
# Node 2 (Rural):      2 000 people — lower transmission, more outdoor biting
# ------------------------------------------------------------------

THREE_NODES = NodeGrid(
    populations = [5_000, 3_000, 2_000],
    labels      = ['Urban', 'Peri-urban', 'Rural'],

    # Per-node seasonal EIR (bites/person/year) — used by eir_only mosquito model.
    eir = [
        [ 5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5,  5],   # Urban    — high year-round
        [ 2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2],   # Peri-urban — moderate wet-season
        [ 1,  1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1],   # Rural    — lower with sharp peak
    ],

    # Per-node monthly emergence (new females/human/day) — used by agents model.
    emergence = [
        [0.008, 0.008, 0.015, 0.030, 0.045, 0.060, 0.060, 0.045, 0.030, 0.015, 0.008, 0.008],
        [0.005, 0.005, 0.010, 0.020, 0.030, 0.040, 0.040, 0.030, 0.020, 0.010, 0.005, 0.005],
        [0.002, 0.002, 0.005, 0.010, 0.018, 0.025, 0.025, 0.018, 0.010, 0.005, 0.002, 0.002],
    ],

    # Per-node indoor bite fraction.
    f_indoor = [0.80, 0.70, 0.55],
)

SCENARIOS = [
    dict(
        label    = 'SIS + EIR-only',
        human    = 'sis',
        mosquito = 'eir_only',
        nodes    = THREE_NODES,
    ),
    dict(
        label    = 'SEIS + EIR-only',
        human    = 'seis',
        mosquito = 'eir_only',
        nodes    = THREE_NODES,
    ),
    dict(
        label    = 'SIS + agents',
        human    = 'sis',
        mosquito = 'agents',
        nodes    = THREE_NODES,
    ),
    dict(
        label    = 'SEIS + agents',
        human    = 'seis',
        mosquito = 'agents',
        nodes    = THREE_NODES,
    ),
]

# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------

_proc = psutil.Process(os.getpid())

sims  = []
perfs = []
for sc in SCENARIOS:
    label        = sc.pop('label')
    sim = make_sim(
        n_agents      = N_AGENTS,
        start         = START,
        stop          = STOP,
        dt            = ss.days(1),
        label         = label,
        rand_seed     = 0,
        people        = ss.People(N_AGENTS, age_data=AGE_DATA),
        demographics  = DEMOGRAPHICS,
        analyzers     = AgeBinPopulation(),
        **sc,
    )
    print(f'Running: {label}')
    mem_before = _proc.memory_info().rss
    t0         = time.perf_counter()
    sim.run()
    elapsed    = time.perf_counter() - t0
    mem_after  = _proc.memory_info().rss
    sims.append(sim)
    perfs.append(dict(
        elapsed      = elapsed,
        mem_before   = mem_before / 1024**2,
        mem_after    = mem_after  / 1024**2,
        mem_delta    = (mem_after - mem_before) / 1024**2,
    ))

NODE_COLORS = ['steelblue', 'darkorange', 'seagreen', 'firebrick', 'purple', 'brown']

# ------------------------------------------------------------------
# Plot: prevalence comparison
# ------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle('Malaria prevalence — model comparison', fontsize=13)

for ax, sim in zip(axes.flat, sims):
    # Disease module name is lowercased class name
    disease_key = type(sim.diseases[0]).__name__.lower()
    res         = sim.results[disease_key]

    n_alive     = sim.results.n_alive
    prevalence  = res.n_infected / n_alive * 100   # percent

    timevec = res.n_infected.timevec.years
    ax.plot(timevec, prevalence, lw=2, color='black', label='Total')

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

plt.tight_layout()
plt.savefig('malaria_prevalence.png', dpi=150)
plt.show()
print('Plot saved to malaria_prevalence.png')

# ------------------------------------------------------------------
# Plot: compartment fractions (S, E when present, I) — 2×2 grid
# ------------------------------------------------------------------

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

    ax.plot(timevec, res.n_infected / n_alive, label='Infectious',     color='firebrick',  lw=1.5)

    ax.set_title(sim.label)
    ax.set_xlabel('Year')
    ax.set_ylabel('Fraction of population')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('malaria_compartments.png', dpi=150)
plt.show()
print('Plot saved to malaria_compartments.png')

# ------------------------------------------------------------------
# Plot: mosquito SEI compartments — 2×2 grid (empty for EIR-only sims)
# ------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle('Mosquito SEI compartments', fontsize=13)

for ax, sim in zip(axes.flat, sims):
    mos_mod = next(
        (mod for mod in sim.modules if isinstance(mod, MosquitoesAdultFemaleAgentsOnly)),
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

plt.tight_layout()
plt.savefig('malaria_mosquitoes.png', dpi=150)
plt.show()
print('Plot saved to malaria_mosquitoes.png')

# ------------------------------------------------------------------
# Plot: mosquito count per node — one chart per agents-mode sim
# ------------------------------------------------------------------

agents_sims = [s for s in sims if s.analyzers[0].node_mosquito_hist]
if agents_sims:
    fig, axes = plt.subplots(1, len(agents_sims), figsize=(6 * len(agents_sims), 5),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle('Mosquito count by node', fontsize=13)

    for ax, sim in zip(axes.flat, agents_sims):
        az          = sim.analyzers[0]
        disease_key = type(sim.diseases[0]).__name__.lower()
        timevec     = sim.results[disease_key].timevec.years

        for (node_lbl, counts), color in zip(az.node_mosquito_hist.items(), NODE_COLORS):
            ax.plot(timevec, counts, lw=1.5, color=color, label=node_lbl)

        ax.set_title(sim.label)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of mosquitoes')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('malaria_mosquitoes_by_node.png', dpi=150)
    plt.show()
    print('Plot saved to malaria_mosquitoes_by_node.png')

# ------------------------------------------------------------------
# Plot: population by 5-year age bin — one image per sim, 4×6 grid
# ------------------------------------------------------------------

NCOLS, NROWS = 6, 4   # 24 slots; 22 age-bin + 1 per-node + 1 hidden


for sim in sims:
    az          = sim.analyzers[0]
    disease_key = type(sim.diseases[0]).__name__.lower()
    timevec     = sim.results[disease_key].timevec.years

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

    slug  = sim.label.lower().replace(' ', '_').replace('+', 'plus').replace('-', '').strip('_')
    fname = f'malaria_age_bins_{slug}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f'Plot saved to {fname}')

# ------------------------------------------------------------------
# Performance: CPU plot + text summary — one per sim
# ------------------------------------------------------------------

for sim, perf in zip(sims, perfs):
    slug = sim.label.lower().replace(' ', '_').replace('+', 'plus').replace('-', '').strip('_')

    # CPU-time bar chart from the integration loop
    fig = sim.loop.plot_cpu()
    if fig is not None:
        cpu_fname = f'malaria_cpu_{slug}.png'
        fig.savefig(cpu_fname, dpi=150)
        plt.close(fig)
        print(f'CPU plot saved to {cpu_fname}')

    # Per-module CPU breakdown
    loop_df      = sim.loop.to_df()
    module_times = loop_df.groupby('module')['cpu_time'].sum().sort_values(ascending=False)

    disease_name   = type(sim.diseases[0]).__name__.lower()
    connector_names = [type(c).__name__.lower() for c in sim.connectors] if sim.connectors else []

    W = 52
    lines = [
        f'{"=" * W}',
        f'  {sim.label}',
        f'{"=" * W}',
        f'  Runtime        : {perf["elapsed"]:.2f}s',
        f'  Memory (RSS)   : {perf["mem_before"]:.1f} → {perf["mem_after"]:.1f} MB  (Δ {perf["mem_delta"]:+.1f} MB)',
        f'  Human  ({disease_name})',
        f'    CPU          : {module_times.get(disease_name, 0.0):.3f}s',
    ]
    for cname in connector_names:
        lines += [
            f'  Mosquito ({cname})',
            f'    CPU          : {module_times.get(cname, 0.0):.3f}s',
        ]
    lines.append(f'  Top modules by CPU:')
    for mod, t in module_times.head(8).items():
        lines.append(f'    {mod:<38}  {t:.3f}s')

    summary = '\n'.join(lines)

    perf_fname = f'malaria_perf_{slug}.txt'
    with open(perf_fname, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'Performance summary saved to {perf_fname}')
