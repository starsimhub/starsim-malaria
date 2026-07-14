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

`malariasim/` is a more complete agent-based malaria model with a spatial multi-node grid.

### Module communication pattern

Human and mosquito modules are **fully decoupled** — neither imports the other. They communicate each timestep through a shared `NodeState` object (an `ss.Module` registered as a connector), discovered at runtime via the `_is_node_state` duck-type marker:

```text
Mosquito → Human  :  eir_indoor & eir_outdoor  (per-node, per-timestep)
Human   → Mosquito:  p_inf_indoor & p_inf_outdoor & n_humans  (per-node, per-timestep)
```

There is a one-step lag on EIR because diseases run before connectors in Starsim's execution order.

### Execution order per timestep

1. **Human disease step** (`ss.Disease.step` → `step_state` + `step`) — reads EIR from NodeState, writes `p_inf` and `n_humans` back
2. **NodeState.step()** — no-op ordering anchor
3. **Mosquito connector step** — reads `p_inf`/`n_humans`, writes EIR for next step

### malariasim key design decisions

- **Mosquitoes are connectors, not diseases.** Both `NodeState` and the mosquito module are registered as `connectors` in `ss.Sim`, not `diseases`. This is because Starsim runs diseases before connectors.
- **Mosquito agents are separate from `ss.People`.** The agent-based mosquito module manages its own numpy arrays internally (pre-allocated to 3× max equilibrium population) to avoid mixing species in the human People object.
- **EIR-based transmission bypasses Starsim networks.** `MalariaHumanBase.step()` overrides the default `ss.SIS.step()` to implement EIR-driven mosquito-to-human transmission directly, ignoring Starsim's contact network machinery.
- **`ss.SIS` immunity is neutralized.** `imm_boost=0` disables the built-in immunity system since malaria immunity is not modeled at this level.
- **Durations are in years, converted to timesteps.** Starsim stores time indices as integer steps. `set_prognoses()` divides year-denominated durations by `dt_year` to get step counts.

### Spatial model

`NodeGrid` defines static spatial configuration (populations, per-node EIR/emergence/indoor-bite-fraction). `NodeState` holds dynamic per-node fields updated each timestep. Agents are assigned to nodes at birth via `pop_fractions`. When no `NodeGrid` is provided, `Sim()` creates a single-node grid.

### Extending the human model

Inherit from `MalariaHumanBase` and override:

- `_do_disease_transitions()` — disease-state progression (called inside `step_state()` template)
- `set_prognoses(uids, sources)` — entry point for newly exposed agents
- `_update_symptom_state()` — symptom classification from observables
- `step_die(uids)` — reset extra states on agent death (call `super()`)

### malariasim module map

- `malariasim/sim.py` — `Sim(ss.Sim)` subclass wiring human + mosquito + NodeState together
- `malariasim/human_base.py` — `MalariaHumanBase` (shared biting infrastructure, observable states, EIR transmission)
- `malariasim/human_sis.py` — `MalariaHumanSIS` (thin SIS subclass)
- `malariasim/human_seis.py` — `MalariaHumanSEIS` (adds exposed compartment with ~12-day incubation)
- `malariasim/mosquito_eir_only.py` — `MosquitoesEirOnly` (fixed/seasonal EIR, no mosquito agents)
- `malariasim/mosquito_agents.py` — `MosquitoesAdultFemaleAgentsOnly` (SEI agent-based mosquitoes)
- `malariasim/node_grid.py` — `NodeGrid` (static spatial config)
- `malariasim/node_state.py` — `NodeState` (dynamic per-node communication bus)
- `malariasim/results.py` — `AgeBinPopulation` analyzer + `MalariaMultiSim` with plotting
- `malariasim/performance.py` — `PerformanceMonitor` (wall-clock + RSS tracking)
- `run_malariasim.py` — entry point running 4 scenarios (SIS/SEIS × EIR-only/agents) with 3-node spatial grid

## Dependencies

Core: `numpy`, `pandas`, `matplotlib`, `starsim`, `sciris`. Dev: `pytest`. Requires Python >= 3.12.

## Known Quirks

- `PerformanceMonitor` requires `psutil` but it is not listed in `pyproject.toml` dependencies.
- The canonical import form is `import malariasim as ms`.
