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
import starsim as ss

from malaria import make_sim

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

N_AGENTS = 10_000
START    = 2000
STOP     = 2004

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
