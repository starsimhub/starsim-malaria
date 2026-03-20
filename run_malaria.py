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
import pandas as pd
import starsim as ss

from malaria import make_sim


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
        self.labels = [f'{lo}-{hi}' for lo, hi in self.BINS] + ['100+', 'Total']
        self.hist   = {lbl: [] for lbl in self.labels}

    def step(self):
        people = self.sim.people
        alive  = np.asarray(people.alive.values, dtype=bool)
        age    = np.asarray(people.age.values,   dtype=float)

        for (lo, hi), lbl in zip(self.BINS, self.labels):
            mask = alive & (age >= lo) & (age < hi)
            self.hist[lbl].append(int(mask.sum()))

        self.hist['100+'].append(int((alive & (age >= 100)).sum()))
        self.hist['Total'].append(int(alive.sum()))

    def finalize_results(self):
        super().finalize_results()
        for lbl in self.labels:
            self.hist[lbl] = np.array(self.hist[lbl])

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
STOP     = 2010

SEASONAL_EIR = [2, 2, 5, 6, 8, 10, 12, 8, 5, 3, 2, 2]   # wet-season peak

SCENARIOS = [
    dict(
        label        = 'SIS + EIR-only',
        human        = 'sis',
        mosquito     = 'eir_only',
        mosquito_pars= dict(eir=SEASONAL_EIR),
    ),
    dict(
        label        = 'SEIS + EIR-only',
        human        = 'seis',
        mosquito     = 'eir_only',
        mosquito_pars= dict(eir=SEASONAL_EIR),
    ),
    dict(
        label        = 'SIS + agents',
        human        = 'sis',
        mosquito     = 'agents',
    ),
    dict(
        label        = 'SEIS + agents',
        human        = 'seis',
        mosquito     = 'agents',
    ),
]

# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------

sims = []
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
    sim.run()
    sims.append(sim)

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

    ax.plot(res.n_infected.timevec, prevalence, lw=1.5)
    ax.set_title(sim.label)
    ax.set_ylabel('Prevalence (%)')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('malaria_prevalence.png', dpi=150)
plt.show()
print('Plot saved to malaria_prevalence.png')

# ------------------------------------------------------------------
# Plot: compartment fractions (S, E when present, I) — 2×2 grid
# ------------------------------------------------------------------

from malaria import MosquitoesAdultFemaleAgentsOnly

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
# Plot: population by 5-year age bin — one image per sim, 4×6 grid
# ------------------------------------------------------------------

NCOLS, NROWS = 6, 4   # 24 slots; 22 used, 2 hidden

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

    for ax in axes.flat[len(az.labels):]:
        ax.set_visible(False)

    slug  = sim.label.lower().replace(' ', '_').replace('+', 'plus').replace('-', '').strip('_')
    fname = f'malaria_age_bins_{slug}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f'Plot saved to {fname}')
