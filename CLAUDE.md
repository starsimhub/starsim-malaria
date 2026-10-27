# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Starsim-NGM contains four malaria model implementations of increasing complexity:

| Folder | Description |
| --- | --- |
| `malaria_demo/` | Direct translation of Mahmud et al. (2025) compartmental model |
| `malaria_sslaser/` | Direct translation of the LASER-NGM agent-based model |
| `malaria_hybrid/` | Ross-Macdonald model with interchangeable ODE/ABM human and mosquito modules |
| `malariasim/` | More complete agent-based model drawing on EMOD-Malaria |

Entry points: `run_malaria_demo.py`, `run_malaria_hybrid.py`, `run_malaria_sslaser.py`, `run_malariasim.py`.

## Environment

The Python virtual environment is in the `env/` directory. Always use `env/Scripts/python.exe` to run Python or pytest:

```bash
env/Scripts/python.exe -m pytest
env/Scripts/python.exe run_malariasim.py
```

## Commands

```bash
# Install (editable mode)
env/Scripts/python.exe -m pip install -e .

# Install with dev dependencies
env/Scripts/python.exe -m pip install -e ".[dev]"

# Run a specific model
env/Scripts/python.exe run_malaria_hybrid.py
env/Scripts/python.exe run_malariasim.py

# Run tests
pytest
```

## malaria_hybrid Architecture

`malaria_hybrid/malaria_hybrid.py` implements a Ross-Macdonald malaria model with four
interchangeable module combinations (ODE or ABM for humans, ODE or ABM for mosquitoes).

### Shared state bus

All modules communicate through `sim.c` (an `sc.objdict`) and two shared sim-level pars:

```text
sim.c.Sh, sim.c.Ih          — human susceptible/infectious counts
sim.c.Sm, sim.c.Em, sim.c.Im — mosquito susceptible/exposed/infectious counts
sim.pars.a                   — biting rate (bites per mosquito per day)
sim.pars.Nh                  — human population size
```

### Execution order per timestep (Gauss-Seidel Euler)

1. Human step (`HumanODE` or `HumanABM`) — reads old `Im`, writes `Sh`/`Ih` to `sim.c`
2. Mosquito step (`MosquitoODE` or `MosquitoABM`) — reads updated `Ih`, writes `Sm`/`Em`/`Im`
3. `Malaria.update_results()` — records derived quantities (`lambda_h`, EIR, R0)

### malaria_hybrid key design decisions

- **Any human module works with any mosquito module.** Both write to and read from `sim.c`, so all four combinations (ODE×ODE, ABM×ODE, ODE×ABM, ABM×ABM) share one interface.
- **Mosquito agents are separate from `ss.People`.** `MosquitoABM` manages its own pre-allocated numpy arrays (3× peak seasonal equilibrium) to avoid mixing species.
- **ODE humans use `n_agents=1`.** `Sim.__init__` sets `n_agents=Nh` only for ABM humans; ODE humans need a minimal People object to satisfy Starsim.
- **Seasonal mosquito population control.** Both ODE and ABM mosquito modules use a relaxation term (`sn_relax`) to track a seasonal target `M*(t)` — births top up shortfalls, extra deaths trim excess.

### malaria_hybrid module map

- `malaria_hybrid/malaria_hybrid.py` — all six classes: `HumanODE`, `HumanABM`, `MosquitoODE`, `MosquitoABM`, `Malaria`, `Sim`, `MultiSim`
- `run_malaria_hybrid.py` — runs all four ODE/ABM combinations

## malariasim Architecture

`malariasim/` is a malaria model on Starsim 4 with a spatial multi-node grid. The model is one declared disease, so the same model runs as agents or as an ODE (`solver='abm'` or `solver='ode'`). The port from Starsim 3 is described in `v4_port_oct26.md`.

### The model

`ms.Malaria` (`malariasim/malaria.py`) declares the states and transitions:

- People: `S`, `E` (SEIS only), `I`; infected by indoor and outdoor bites (`bite_indoor`, `bite_outdoor`, at `b × f_indoor × EIR` and `b × (1 − f_indoor) × EIR`), recovering back to `S`.
- The EIR comes from one of three mosquito models: `'ode'` (mosquitoes as counts `Sv`, `Ev`, `Iv`, states with `people=False`, which emerge per person, feed, incubate, and die), `'eir_only'` (an EIR given, by month and node), or `'agents'` (individual mosquitoes written as code in `ms.MosquitoAgents`, whose per-node EIR the people's rates read as `ss.by(patch=fn)`).
- Nodes are a patch stratification (`ss.geo.geography`, `kind='space'`); per-node values (`eir`, `emergence`, `f_indoor`) are `ss.by(patch=...)`, and monthly profiles are `ss.timeseries` repeating each year (`ms.monthly()`).
- Biting heterogeneity is a `biting` stratification with gamma-distributed levels, and the mosquitoes are infected by the biting-weighted count of infectious people (`(ss.by(biting=w)*I).n`); age-dependent biting is an `ss.by(age=...)` multiplier.

### Interventions

Products are Starsim products delivered by Starsim's deliveries: vaccines as `ss.Vaccine` (waning as a box then a decay with `ss.box(dur=, decay=)`), `ms.Bednet` (an `ss.Vaccine` against indoor bites with four effects, each with its own waning, whose blocking and killing the mosquitoes read from its coverage, `itn.coverage`), drugs as `ss.Treatment(cure=dict(I=1.0))`, and outbreaks as `ss.Delivery(move=mal.S >> mal.I)`. Targeting is `where=` (e.g. `ss.ppl.patch == 'Rural'`). Waning by `ss.box()` runs under every solver; by any other function of time, as agents only.

### malariasim module map

- `malariasim/malaria.py` — `Malaria` (the declared disease), `monthly()`, `per_node()`, and `mosquito_eir()` (the mosquito agents' EIR by node, which the rates read)
- `malariasim/mosquito_agents.py` — `MosquitoAgents` (SEI mosquito agents, written as code)
- `malariasim/interventions.py` — `Bednet`
- `malariasim/sim.py` — `Sim(ss.Sim)`, assembling the disease, nodes, people, and interventions
- `malariasim/defaults.py` — `CommonParams`, the age pyramid and node constants, `make_steady_state_demographics()` (declared `ss.birth`/`ss.death`), `make_age_data()`
- `malariasim/utils.py` — `eir_to_params()` (the emergence that gives a target EIR)
- `malariasim/results.py` — `MalariaMultiSim` with plotting
- `malariasim/performance.py` — `PerformanceMonitor` (wall-clock + RSS tracking)
- `run_malariasim.py` — entry point running 8 scenarios (SIS/SEIS as agents, SIS as an ODE × EIR-only/ODE/agent mosquitoes) with a 3-node spatial grid

## Dependencies

Core: `numpy`, `pandas`, `matplotlib`, `starsim`, `sciris`. Dev: `pytest`. Requires Python >= 3.12.

## Known Quirks

- `PerformanceMonitor` requires `psutil` but it is not listed in `pyproject.toml` dependencies.
- The canonical import form is `import malariasim as ms`.
