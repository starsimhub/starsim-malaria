# LASER → Starsim Conversion: malaria_model → malaria_sslaser

## 1. Overview

`malaria_sslaser` is a functionally equivalent Starsim-compatible re-implementation of `malaria_model`. The conversion uses a bridge-based approach: Starsim-compatible wrappers (`SSPeople`, `SSPars`) provide the same API as LASER's `LaserFrame` and `PropertySet`, so every component's logic is preserved verbatim. Only imports and the model orchestrator change.

The conversion was performed following the LASER→Starsim 10-step process in the `laser-starsim-converter` agent.

**What was converted:**
- `malaria_model/model.py` → `malaria_sslaser/model.py`
- `malaria_model/components/*.py` → `malaria_sslaser/components/*.py`
- `malaria_model/results.py` → `malaria_sslaser/results.py`
- `run_sim.py` → `run_sim_ss.py`
- `tests/test_sim.py` → `tests/test_sim_ss.py`

**What was added:**
- `malaria_sslaser/starsim.py` — bridge classes (`SSPeople`, `SSPars`, `Results`)
- `malaria_sslaser/laser_utils.py` — re-implements `laser.core.random` API
- `tests/test_laser_starsim.py` — cross-framework validation test

---

## 2. Key Structural Changes

### Model

| LASER | Starsim |
|-------|---------|
| `MinimalMalariaModel(scenario, params: PropertySet)` | `MinimalMalariaModel(scenario, params: SSPars)` |
| `model.people: LaserFrame` | `model.people: SSPeople` |
| `model.nodes: LaserFrame` | `model.nodes: _NodesFrame` |
| `model.params: PropertySet` | `model.params: SSPars` |
| Tick loop: `for tick in range(nticks): comp(model, tick)` | Identical — no change to loop |

### People / Agents

`SSPeople` is a LaserFrame-compatible wrapper that stores properties as raw numpy arrays:
- `add_scalar_property(name, dtype, default)` → `np.full(capacity, default, dtype)`
- `.count` → total agents in use
- `.capacity` → pre-allocated array size

### Nodes

`_NodesFrame` provides the same API for node-level arrays:
- `add_scalar_property(name, dtype, default)` → 1-D numpy array of size `nnodes`
- `add_vector_property(name, length, dtype, default)` → 2-D numpy array of shape `(length, nnodes)` (tick-indexed like `nodes.deaths_natural[today, nid]`)

### Params

`SSPars` extends `ss.Pars`, providing:
- Attribute access: `params.nticks`
- `setattr` for dynamic param addition: `params.new_param = value`
- `hasattr` for param existence checks

### Components

No component logic changed. Only imports:
- `from laser.core import PropertySet` → removed (PropertySet not needed in components)
- `import laser.core.random as lrandom` → `import malaria_sslaser.laser_utils as lrandom`
- `from malaria_model.enums import ...` → `from malaria_sslaser.enums import ...`

---

## 3. API Mapping Table

| LASER API | Starsim Equivalent | Notes |
|-----------|-------------------|-------|
| `from laser.core import LaserFrame` | `from malaria_sslaser.starsim import SSPeople` | Bridge class |
| `from laser.core.propertyset import PropertySet` | `from malaria_sslaser.starsim import SSPars` | Bridge class |
| `LaserFrame(capacity=N, initial_count=N)` | `SSPeople(capacity=N, initial_count=N)` | Same constructor |
| `LaserFrame(capacity=nnodes, initial_count=nnodes)` (nodes) | `_NodesFrame(nnodes=nnodes)` | Separate class |
| `population.add_scalar_property(name, dtype, default)` | `people.add_scalar_property(name, dtype, default)` | Identical API |
| `nodes.add_vector_property(name, length, dtype, default)` | `nodes.add_vector_property(name, length, dtype, default)` | Identical API |
| `PropertySet({...})` | `SSPars({...})` | Drop-in |
| `import laser.core.random as lrandom` | `import malaria_sslaser.laser_utils as lrandom` | Same `prng()` / `seed()` API |
| `lrandom.seed(seed)` | `lrandom.seed(seed)` | Identical |
| `lrandom.prng()` | `lrandom.prng()` | Returns `np.random.Generator` |
| `ppl.count` | `ppl.count` | Identical |
| `comp(model, tick)` | `comp(model, tick)` | No change to component call signature |

---

## 4. New Files

### `starsim.py`
Copied from `laser-polio/src/laser_polio/starsim.py`. Contains:
- `SSPars` — `ss.Pars` subclass with `PropertySet`-compatible API
- `SSPeople` — LaserFrame-compatible people container
- `Results` — simple results container (array storage)

This file is the bridge layer for incremental migration; components work unchanged while the container is Starsim-compatible.

### `laser_utils.py`
Re-implements `laser.core.random` module behavior:
- `seed(seed_val)` — sets global `np.random.Generator` with given seed
- `prng()` — returns the global `np.random.Generator` (or auto-seeds from microseconds)

Both functions use `np.random.default_rng()` identically to the LASER implementation, so behavior is equivalent when seeded identically.

---

## 5. Challenges

### `_NodesFrame` vs `LaserFrame` for nodes
LASER uses `LaserFrame` for both people and nodes, but `SSPeople` is designed for agents (rows = agents). For nodes, a simple `_NodesFrame` class is used that provides the same `add_scalar_property` / `add_vector_property` interface. The `add_vector_property` on nodes creates shape `(length, nnodes)` arrays to match the `nodes.deaths_natural[today, nid]` indexing pattern used by `Demographics`.

### `hs_symptom_state` pre-initialization
`Demographics.__init__` runs before `HumanSymptoms.__init__` when components are added in order, but `Demographics.__call__` accesses `ppl.hs_symptom_state` at tick 0. The key insight is that all component `__init__` methods run during `add_component()` (before `run()`), so by the time tick 0 executes, `hs_symptom_state` has been declared by `HumanSymptoms.__init__`. The execution order in `model.add_component()` calls establishes the correct initialization order.

### Stochastic results
Both models use `np.random.default_rng(seed)` as the global PRNG. When seeded identically and run in the same process, the RNG state advances identically through both models — confirmed by `test_laser_starsim.py` where prevalence values match exactly (0.4327 in both).

---

## 6. Performance

From `test_laser_starsim.py` (730 ticks, ~10,000 agents, 3 nodes):

| Framework | Wall time |
|-----------|-----------|
| LASER     | ~1.91s    |
| Starsim   | ~1.92s    |
| Ratio     | ~1.00x    |

The conversion has zero performance cost because all component logic runs through identical numpy arrays in both cases. The only overhead from `SSPeople` and `_NodesFrame` is a thin Python wrapper — no copies, no translation layers.

---

## 7. Known Differences

### Stochastic divergence
If the two models are run in separate processes (different Python invocations), the RNG state will differ between `laser.core.random` and `malaria_sslaser.laser_utils` because they maintain separate global `_prng` instances. Results will be qualitatively equivalent but numerically different. Within the same process (as in the validation test), results are identical.

### `laser.core.random._nbseed`
The original LASER `seed()` also calls `_nbseed()` to set Numba per-thread RNG seeds. `laser_utils.seed()` does NOT call `_nbseed` since the malaria model contains no `@numba.njit`-decorated functions. If Numba JIT functions are added in the future, `laser_utils.seed` should be extended accordingly.

### `PropertySet` vs `SSPars`
`SSPars` extends `ss.Pars` (a sciris `odict` subclass) rather than replicating `PropertySet`'s exact inheritance chain. The observable behavior is identical: attribute access, `setattr`, and `hasattr` all work the same way for all patterns used in this codebase.
