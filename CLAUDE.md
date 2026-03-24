# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Starsim-NGM is a Starsim-based malaria modeling framework. It implements agent-based malaria transmission with interchangeable human disease modules (SIS, SEIS) and mosquito modules (EIR-only, agent-based SEI), communicating through a decoupled NodeState bus.

## Commands

```bash
# Install (editable mode)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run all four scenarios (SIS/SEIS × EIR-only/agents)
python run_malaria.py

# Run tests
pytest
```

## Architecture

### Module Communication Pattern

Human and mosquito modules are **fully decoupled** — neither imports the other. They communicate each timestep through a shared `NodeState` object (an `ss.Module` registered as a connector), discovered at runtime via the `_is_node_state` duck-type marker:

```
Mosquito → Human  :  eir_indoor & eir_outdoor  (per-node, per-timestep)
Human   → Mosquito:  p_inf_indoor & p_inf_outdoor & n_humans  (per-node, per-timestep)
```

There is a one-step lag on EIR because diseases run before connectors in Starsim's execution order.

### Execution Order (per timestep)

1. **Human disease step** (`ss.Disease.step` → `step_state` + `step`) — reads EIR from NodeState, writes `p_inf` and `n_humans` back
2. **NodeState.step()** — no-op ordering anchor
3. **Mosquito connector step** — reads `p_inf`/`n_humans`, writes EIR for next step

### Key Design Decisions

- **Mosquitoes are connectors, not diseases.** Both `NodeState` and the mosquito module are registered as `connectors` in `ss.Sim`, not `diseases`. This is because Starsim runs diseases before connectors.
- **Mosquito agents are separate from `ss.People`.** The agent-based mosquito module manages its own numpy arrays internally (pre-allocated to 3× max equilibrium population) to avoid mixing species in the human People object.
- **EIR-based transmission bypasses Starsim networks.** `MalariaHumanBase.step()` overrides the default `ss.SIS.step()` to implement EIR-driven mosquito-to-human transmission directly, ignoring Starsim's contact network machinery.
- **`ss.SIS` immunity is neutralized.** `imm_boost=0` disables the built-in immunity system since malaria immunity is not modeled at this level.
- **Durations are in years, converted to timesteps.** Starsim stores time indices as integer steps. `set_prognoses()` divides year-denominated durations by `dt_year` to get step counts.

### Spatial Model

`NodeGrid` defines static spatial configuration (populations, per-node EIR/emergence/indoor-bite-fraction). `NodeState` holds dynamic per-node fields updated each timestep. Agents are assigned to nodes at birth via `pop_fractions`. When no `NodeGrid` is provided, `Sim()` creates a single-node grid.

### Extending the Human Model

Inherit from `MalariaHumanBase` and override:
- `_do_disease_transitions()` — disease-state progression (called inside `step_state()` template)
- `set_prognoses(uids, sources)` — entry point for newly exposed agents
- `_update_symptom_state()` — symptom classification from observables
- `step_die(uids)` — reset extra states on agent death (call `super()`)

### Module Map

- `malariasim/sim.py` — `Sim(ss.Sim)` subclass wiring human + mosquito + NodeState together
- `malariasim/human_base.py` — `MalariaHumanBase` (shared biting infrastructure, observable states, EIR transmission)
- `malariasim/human_sis.py` — `MalariaHumanSIS` (thin SIS subclass)
- `malariasim/human_seis.py` — `MalariaHumanSEIS` (adds exposed compartment with ~12-day incubation)
- `malariasim/mosquito_eir_only.py` — `MosqitoesEirOnly` (fixed/seasonal EIR, no mosquito agents)
- `malariasim/mosquito_agents.py` — `MosquitoesAdultFemaleAgentsOnly` (SEI agent-based mosquitoes)
- `malariasim/node_grid.py` — `NodeGrid` (static spatial config)
- `malariasim/node_state.py` — `NodeState` (dynamic per-node communication bus)
- `malariasim/results.py` — `AgeBinPopulation` analyzer + `MalariaMultiSim` with plotting
- `malariasim/performance.py` — `PerformanceMonitor` (wall-clock + RSS tracking)
- `run_malaria.py` — entry point running 4 scenarios (SIS/SEIS × EIR-only/agents) with 3-node spatial grid

## Dependencies

Core: `numpy`, `pandas`, `matplotlib`, `starsim`. Also uses `sciris` at runtime (imported in `run_malaria.py`). Dev: `pytest`. Requires Python >= 3.12.

## Known Quirks

- The class name `MosqitoesEirOnly` has a typo (missing 'u' in "Mosquitoes") — this is intentional/historical and used throughout including `__init__.py` exports.
- `PerformanceMonitor` requires `psutil` but it is not listed in `pyproject.toml` dependencies.
- The canonical import form is `import malariasim as ms`.
