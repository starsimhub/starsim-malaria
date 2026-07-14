# malariasim Intervention Design

## Why We Diverge from `ss.Intervention`

Starsim's `ss.Intervention` bundles two responsibilities: what the intervention does
biologically and how and when it is distributed. It also implies that each intervention
manages its own per-step updates.

This breaks down for malaria in two ways.

**Separation of concerns.** A vaccine delivered via routine immunization and a vaccine
delivered via a mass campaign are the same biological intervention with different
distribution strategies. Starsim's model would require two separate `Intervention`
objects with duplicated biological logic.

**Performance.** Both distributors would iterate over all their respective recipients
every step to update waning efficacy. Recipients overlap. With three distribution
strategies and 10,000 vaccinated people, that's 30,000 agent updates per step instead
of 10,000.

---

## Interventions Planned

| Intervention | Target | Mechanism |
|---|---|---|
| Vaccine (e.g. RTS,S, R21) | Person | Reduces probability of infection per bite |
| Insecticide-treated bed net (ITN) | Person | Reduces biting rate (human + mosquito side); kills mosquitoes via insecticide contact |
| Antimalarial drug | Person | Clears infection, provides prophylaxis |
| Diagnostic test (RDT, microscopy) | Person | Identifies infected individuals, triggers treatment |
| Ivermectin | Person | Makes blood meal toxic to feeding mosquitoes |
| Larvicide | Node | Reduces mosquito emergence rate |
| Indoor residual spraying (IRS) | Node | Increases indoor mosquito death rate |
| Attractive targeted sugar bait (ATSB) | Node | Kills mosquitoes at sugar sources |
| Mosquito release (SIT, Wolbachia) | Node | Modifies mosquito population |

**One active instance per intervention type per agent.** An agent can hold at most one
active dose of each intervention type at a time. A person can have both a vaccine and a
bed net simultaneously, but not two vaccines. A second dose of the same intervention
overwrites the first — the new dose time, efficacy, and waning profile replace the
previous values by simply writing the new values into the existing state arrays. This
simplifies per-person state to fixed-size arrays and avoids tracking dose histories.

---

## Five Kinds of New Objects

The `build_campaign` function works with five distinct kinds of objects, each with a
single responsibility.

**ProductComponent** (`WaningEffect`, ...) — a pure data class that defines the
behaviour of part of an intervention but is not a biological product itself.
`WaningEffect` holds initial efficacy, box duration, and decay time constant — it is
passed into a Product and has no `administer()` method. ProductComponents carry no
registration logic; they are created freely at any time and passed to a Product. The
Product is responsible for registering the required state arrays with the appropriate
Updater.

**Product** (`MalariaVaccine`, `Bednet`, ...) — a `ss.Product` subclass (which is
itself a `ss.Module` subclass) that defines what an intervention IS biologically and
implements `administer(uids, weights)` to apply that effect. `administer` reads the
current time via `self.sim.t.year` — `self.t` is a per-module clock that advances only
inside `super().step()` and is stale when `administer()` is called from a distributor's
`step()`, so `self.sim.t.year` is required. This means `administer(uids, weights)` needs
no time argument. `weights` carries the per-target coverage fraction from the eligibility
resolution step; for ABM agents weights are always 1.0, for ODE compartments they carry
the `coverage` value. One Product instance is shared across any number of distribution
events — the same vaccine given by routine immunization and two mass campaigns is one
object, not three.

**Updater** (`WaningEffectUpdater`, `MalariaVaccineUpdater`, `BednetUpdater`, ...) — runs
once per step and processes every person who has ever received the intervention, regardless
of which distributor gave it to them. This is the solution to the performance problem
above: one pass over all recipients per step, not one pass per distributor. Updaters are
created automatically by `ms.Sim` and are invisible to the user.

**Distributor** (`InterventionDistributorScheduled`, `InterventionDistributorTriggered`,
`InterventionDistributorReferenceTracking`) — decides who gets dosed and when. A
distributor holds a reference to a Product and calls `administer()` on it when it fires.
Multiple distributors can share the same Product; they write into the same per-person
state arrays, and the Updater handles all of them uniformly. Each `InterventionDistributorScheduled` fires exactly once at its `start` year.
Repetition is handled by the user adding multiple distributors with different `start`
years. Because the Starsim loop plan is fixed at `init()` time and does not support
removing interventions mid-run, `InterventionDistributorScheduled` uses a `_fired`
flag: `step()` checks `self.sim.ti != self._start_ti` and returns immediately on every
step except the one exact match; on that step it distributes and sets `_fired = True`,
after which all subsequent `step()` calls also return immediately. `_start_ti` is
resolved in `init_pre()` via `sc.findnearest(sim.t.yearvec, start_year)`, so `start`
can be an `ss.date` or a plain float year.

**Program** (`InterventionProgram`, e.g. `SeasonalMalariaChemoprevention`) — composes
one or more distributors into a named, reusable delivery strategy. A program encodes
domain knowledge about how an intervention is typically delivered — for example, SMC
combines a fixed schedule, an age target, and a multi-dose drug regimen — so the user
expresses intent ("run SMC") rather than wiring up primitives by hand. `InterventionProgram`
defines a single method `as_interventions()` that returns a list of
`InterventionDistributor` instances. `Campaign` calls `as_interventions()` on each
program and flattens the results into the distributor list passed to `ss.Sim`.

The separation means biological logic lives in one place, distribution strategy lives in
another, per-step maintenance is handled centrally, and complex delivery patterns are
named and reusable. Adding a new distribution strategy for an existing vaccine requires
no changes to the Product, its ProductComponents, or its Updater.

---

## Updater Guard and Registration Pattern

`Updater` is the base class for all per-step connectors. It carries the registration
guard that ensures Products can only register state requirements during the
`build_campaign()` window — not at import time or after the sim is initialised.

### Guard classmethods on `Updater`

```python
class Updater(ss.Module):
    _active: bool = False            # True only inside build_campaign()
    _possible_updaters: dict = {}    # keyed by _updater_key, populated by __init_subclass__

    @classmethod
    def activate(cls):   cls._active = True
    @classmethod
    def deactivate(cls): cls._active = False

    @classmethod
    def get(cls, key):
        if not cls._active:
            raise RuntimeError(
                f"Updater.get({key!r}) called outside the build_campaign() window."
            )
        return cls._possible_updaters[key]
```

`ms.Sim` calls `Updater.activate()` immediately before invoking `build_campaign()` and
`Updater.deactivate()` immediately after. Any Product `__init__` that calls
`Updater.get()` outside this window — for example at module import time or after `sim.run()`
— receives a clear `RuntimeError` rather than silently registering into a stale updater.

### How a Product registers

A Product (e.g. `MalariaVaccine`) calls `Updater.get()` in its own `__init__` to fetch
the appropriate Updater and call `register(prefix, target)` on it directly:

```python
class MalariaVaccine(ss.Product):
    PREFIX = 'vaccine'

    def __init__(self, waning=None, name=None):
        super().__init__(name=name)
        self.waning = waning if waning is not None else WaningEffect()
        Updater.get(WaningEffectUpdater._updater_key).register(self.PREFIX, Target.DISEASE)
```

`Bednet.__init__` registers all four prefixes with `WaningEffectUpdater`:

```python
for prefix in (PREFIX_REPELLING, PREFIX_BLOCKING, PREFIX_KILLING, PREFIX_USAGE):
    wu.register(prefix, Target.DISEASE)
```

`WaningEffectUpdater.register(prefix, target)` stores the prefix and target enum.
Registering the same prefix + target twice is a silent no-op (intentional sharing);
registering the same prefix with a different target raises `ValueError`:

```python
class WaningEffectUpdater(Updater):
    def register(self, prefix, target=Target.DISEASE):
        if prefix in self._entries:
            existing = self._entries[prefix].target
            if existing != target:
                raise ValueError(...)
            return   # same prefix + same target: intentional sharing, no-op
        self._entries[prefix] = _WaningEntry(prefix, target)
```

After `build_campaign()` with both a vaccine and a bednet, a fully registered
`WaningEffectUpdater` holds:

```
'vaccine'        → Target.DISEASE
'itn_repelling'  → Target.DISEASE
'itn_blocking'   → Target.DISEASE
'itn_killing'    → Target.DISEASE
'itn_usage'      → Target.DISEASE
```

### State injection in `init_pre()`

`WaningEffectUpdater.init_pre()` runs during Starsim's `init_modules_pre()` phase.
Connectors are initialised **before** diseases in this phase, which is critical: the
waning state arrays must exist on the disease module before the disease's own
`init_pre()` calls `sim.people.add_module(self)`, which links all of the module's
`ss.Arr` attributes to the People object.

```python
def init_pre(self, sim, force=False):
    super().init_pre(sim, force)
    for entry in self._entries.values():
        entry.module = self._resolve_target(entry.prefix, entry.target)
    self._inject_waning_states()
```

`_resolve_target()` duck-types the sim to find the right module: `Target.DISEASE`
searches `sim.diseases` for a module with `_supports_intervention_effects`;
`Target.NODE_STATE` searches `sim.connectors` for `_is_node_state`.

`_inject_waning_states()` calls `module.define_states()` to create four `ss.FloatArr`
per prefix on the target module:

```
{prefix}_efficacy0   — initial efficacy at dose time (anchor for decay); default 0.0
{prefix}_efficacy    — current efficacy (updated each step by the Updater); default 0.0
{prefix}_box_end     — time when box phase ends and decay begins; sentinel -1
{prefix}_decay_end   — time when efficacy falls below 1% of initial; sentinel -1
```

### Updater public API (called by Products and companion Updaters)

`WaningEffectUpdater` exposes three operations so that neither `MalariaVaccine` nor
`MalariaVaccineUpdater` ever reference the underlying array names directly:

```python
apply(prefix, uids, waning_effect, t_yrs)
    # Write arrays at dose time; return initial efficacy.
    # t_yrs is a per-uid float array (from disease.pre_administer for ODE)
    # or a scalar (sim.t.year for ABM via Bednet.administer).

current_efficacy(prefix)
    # Return current efficacy FloatArr (UID-indexed).

active_mask(prefix)
    # Return BoolArr: decay_end > 0; use .uids to get affected UIDs.
```

In `step()`, the Updater loops over prefixes and performs vectorised updates.
Time is read from `self.sim.t.year` (not `self.t.year` — each module owns its own
`ss.Timeline` whose `ti` only advances inside `super().step()`; the sim's clock is
always current).

`_apply_waning` uses FloatArr comparison operators to produce BoolArr masks, then
indexes via `.uids` — which always returns correct UIDs regardless of agent deaths:

```python
@staticmethod
def _apply_waning(efficacy0, efficacy, box_end, decay_end, t):
    affected = decay_end > 0
    decaying = affected & (box_end < t) & (decay_end > t)
    expired  = affected & (decay_end <= t)

    if len(decaying.uids):
        e0    = efficacy0[decaying.uids]
        t_box = box_end[decaying.uids]
        t_dec = decay_end[decaying.uids]
        tau   = (t_dec - t_box) / _LOG_EXPIRY_INVERSE
        efficacy[decaying.uids] = e0 * np.exp(-(t - t_box) / tau)

    if len(expired.uids):
        efficacy[expired.uids] = 0.0
```

The sentinel guard `decay_end > 0` skips undosed agents without any extra branching.
A second dose simply overwrites the four values — the Updater picks it up automatically
on the next step, enforcing the "one active instance per type" rule without extra logic.

`clear()` resets `self._entries` to an empty dictionary, ready for the next
`build_campaign()` call.

---

## Eligibility Strategy Objects

An `Eligibility` object is a callable `eligibility(sim)` → `ss.uids` that returns the
pool of eligible target IDs.  It performs filtering only — coverage is not applied here.
Passing an eligibility object to a distributor is optional — the default is all alive
agents (`sim.people.auids`).

Both classes inherit from `Eligibility` (abstract base) and implement `__call__(sim)`.

Coverage is applied separately by the distributor via `disease.apply_coverage(uids, coverage)`
after eligibility filtering — see `InterventionDistributor._resolve_targets` below.

### `HumanEligibility`

```python
HumanEligibility(
    min_age  = ss.years(0.5),   # inclusive lower age bound (optional)
    max_age  = ss.years(5),     # exclusive upper age bound (optional)
    node_ids = [2, 3],          # spatial node IDs to target (optional; default = all nodes)
)
```

`__call__(sim)` filters `sim.people.auids` directly — no disease access needed:

- **Age**: `min_age <= sim.people.age[uid] < max_age`.  For ABM agents `sim.people.age`
  is the real agent age in years.  For ODE compartments `sim.people.age[uid]` is the
  bin's minimum age (e.g. 10.0 for the [10, 20) bin) — so the standard half-open filter
  selects whole bins automatically without any special ODE logic.  A campaign targeting
  [10, 30) selects the [10, 20) bin (age=10 ≥ 10 and < 30) and the [20, 30) bin
  (age=20 ≥ 10 and < 30) but not the [30, 40) bin (age=30 is not < 30).
- **Node**: `disease.node_id[uid]` in `node_ids` (read from the human disease module found via `_supports_intervention_effects`, not from `sim.people`).

Returns `ss.uids` — the eligible pool only, no coverage applied.

Constructed with no arguments, targets all alive agents / all compartments.

Validated at construction time:
- `min_age` and `max_age` must each be `>= 0`
- `min_age` must be strictly less than `max_age` when both are supplied

### `NodeEligibility`

```python
NodeEligibility(
    node_ids = [0, 2],   # explicit subset (optional; default = all nodes)
)
```

Returns `ss.uids` of node IDs sourced from `NodeState`. When `node_ids=None` (the
default) all nodes in the simulation are eligible. When an explicit list is supplied
it is validated against the node count from `NodeState` — any ID outside
`0..n_nodes-1` raises `ValueError` naming the offending IDs.

`NodeState` must be present in `sim.connectors` in both cases; its absence raises
`RuntimeError` so that a missing connector surfaces immediately rather than silently
producing an empty eligible pool.

---

## `InterventionDistributor` and `InterventionDistributorScheduled`

### Class hierarchy

```
ss.Intervention
└── InterventionDistributor              abstract base
        └── InterventionDistributorScheduled   fires exactly once at start
```

### `InterventionDistributor` (base)

```python
InterventionDistributor(
    name,              # str — unique name required by starsim's ndict
    products,          # ss.Product or list of ss.Product
    start,             # ss.date or float — when the distributor may fire
    eligibility=None,  # Eligibility callable; defaults to all alive agents
)
```

Holds the product list and eligibility strategy. `name` is required and must be unique
within the sim — starsim uses `module.name` as the ndict key, so two distributors with
the same name in the same sim will raise a `ValueError`. Choose something descriptive
that will be meaningful in a year, e.g. `'vacc_2005_u5'`. This follows the starsim
convention shown in `ss.routine_vx(name='intv_low', ...)`.

`_resolve_targets(coverage)` applies eligibility filtering then coverage sampling in two steps:

```python
eligible       = self.eligibility(self.sim)                   # filter only → ss.uids
uids, weights  = disease.apply_coverage(eligible, coverage)   # ABM: bernoulli; ODE: fraction
```

When `eligibility=None`, `eligible` is `self.sim.people.auids` (all alive agents).
`disease` is found via duck-typing on `_supports_intervention_effects`.  Returns
`(uids, weights)` which are passed directly to `_deliver_to`.

`Campaign.add()` calls `disease.validate_eligibility(item.eligibility)` immediately
after the start-year check, before appending the distributor.  `disease` is found
via the same `_supports_intervention_effects` duck-type on `_thin_sim.diseases`
(populated by `ms.Sim` before calling `build_campaign`).  For ABM this is a no-op;
for ODE it raises `ValueError` if `min_age` or `max_age` don't coincide with a bin
edge in `disease._age_bin_edges`.

`_deliver_to(uids, weights)` calls `product.administer(uids, weights)` on every
product.  Multiple distributors can share one `Product`; state is overwritten on
re-dose per the one-active-instance rule.

`init_pre()` calls `product.init_pre(sim)` for each product. A product shared between
two distributors is initialised once; Starsim's module init guard makes subsequent
calls no-ops.

`step()` raises `NotImplementedError`, making `InterventionDistributor` abstract in
practice. Direct instantiation is possible but running the sim would fail immediately
with a clear message naming the concrete subclass needed.

### `InterventionDistributorScheduled`

```python
InterventionDistributorScheduled(
    name,                # str — unique name required by starsim's ndict
    products,            # ss.Product or list of ss.Product
    start,               # ss.date or float — timestep to fire
    eligibility=None,    # Eligibility callable; defaults to all alive agents
    coverage=None,       # fraction of eligible targets (ss.bernoulli)
    num_targeted=None,   # exact count — not yet implemented (raises NotImplementedError)
)
```

Fires exactly once when `sim.ti == start_ti`. `start_ti` is resolved in `init_pre()`
via `sc.findnearest(sim.t.yearvec, start_year)` so that `start` can be an `ss.date`
or a plain float year.

Coverage is applied via `ss.bernoulli(p=coverage).filter(eligible)`, consistent with
Starsim's probabilistic coverage model. After firing, `_fired = True` is set and all
subsequent `step()` calls return immediately at zero cost.

`num_targeted` is accepted in the constructor (so the API is stable) but raises
`NotImplementedError` immediately — use `coverage` until it is implemented.

Exactly one of `coverage` / `num_targeted` must be supplied; providing both or neither
raises `ValueError` at construction time.

---

## `Campaign` Object

`Campaign` is a container for all intervention distributors in a scenario. It is
constructed inside `build_campaign(sim)` and returned to `ms.Sim`.

### Why `build_campaign` receives `sim`

Passing `sim` into the builder lets `Campaign.add()` validate each distributor's
`start` year against `sim.t.yearvec` at the moment it is added — the `ValueError`
points directly to the offending `add()` call rather than surfacing later in
`as_interventions()` where the culprit is harder to identify.

### `Campaign(sim, items=None)`

```python
campaign = Campaign(sim, [
    InterventionDistributorScheduled(
        name='vacc_2005', products=vaccine, start=ss.date(2005), coverage=0.50),
    InterventionDistributorScheduled(
        name='vacc_2010_u5', products=vaccine, start=ss.date(2010), coverage=0.90,
        eligibility=HumanEligibility(min_age=ss.years(0.5), max_age=ss.years(5),
                                     node_ids=[0, 2])),
])
```

`items` is optional — distributors can also be added one at a time via `add()`.

### `Campaign.add(item)`

Accepts any `InterventionDistributor` subclass. Validates immediately:

- **Type**: raises `TypeError` naming the received type if `item` is not an
  `InterventionDistributor` subclass
- **Start year**: raises `ValueError` naming the start year if it falls outside
  `[sim.t.yearvec[0], sim.t.yearvec[-1]]`

> **Note**: `InterventionProgram` support (named delivery strategies such as
> `SeasonalMalariaChemoprevention`) is planned but not yet implemented.

### `Campaign.as_interventions()`

Called by `ms.Sim` after `build_campaign(sim)` returns. Returns a flat list of all
added distributors sorted ascending by start year. Sorting ensures distributors that
fire earliest are processed first and simplifies any future optimisations that skip
already-fired distributors. Returns a copy — mutating the result does not affect the
`Campaign`.

Before returning, validates that all distributor names are unique. If any two
distributors share a name, raises `ValueError` listing each conflicting distributor
with its `name`, `start` year, and product types.

---

## `WaningEffect` Parameters and `decay_end`

```python
WaningEffect(
    initial_efficacy    = 1.0,           # efficacy in [0,1] at time of dosing; default = full
    box_duration        = ss.days(inf),  # constant-efficacy phase before decay begins; default = permanent
    decay_time_constant = ss.days(inf),  # τ for exponential decay; default = permanent
)
```

A typical RTS,S-like profile:

```python
WaningEffect(
    initial_efficacy    = 0.80,
    box_duration        = ss.days(180),
    decay_time_constant = ss.days(730),
)
```

`decay_end` is not supplied by the user — it is computed at dose time as the point
where efficacy falls below 1% of its initial value:

```
E0 * exp(-t_decay / τ) = E0 * EXPIRY_THRESHOLD   (EXPIRY_THRESHOLD = 0.01)
t_decay = τ * ln(1 / EXPIRY_THRESHOLD)
        = τ * ln(100)

decay_end = box_end + τ * ln(100)
```

`decay_end` depends only on `τ` — not on `initial_efficacy`. The Updater sets
`{prefix}_efficacy = 0` and stops updating an agent once `t >= decay_end`.

Both `box_duration` and `decay_time_constant` must be simultaneously infinite (permanent
effect) or simultaneously finite (decaying effect). Mixed configurations are rejected
at construction time with `ValueError`.

---

## Intervention Effect Interface

Updaters do not write to shared state arrays on `ss.People` or `NodeState` to
communicate effects to human or mosquito modules. Instead, each Updater calls
biological-mechanism methods directly on the appropriate module. This eliminates
intermediate shared arrays and the extra reads and resets they require.

Each human module that wishes to be intervention-compatible must implement a set of
biological-mechanism methods. Methods are named by mechanism, not by intervention type.
For example, both a vaccine and an ITN blocking effect reduce the probability of
infection per bite — they both call the same method on the human module, passing
different `uids`, `efficacy`, and `target` values:

```python
# called by MalariaVaccineUpdater — protects against all bites
human_module.set_acquisition_probability(uids, efficacy, weights, target=BiteTarget.ALL)

# called by BednetUpdater — protects only against indoor bites
human_module.set_acquisition_probability(uids, efficacy, weights, target=BiteTarget.INDOOR)
```

`BiteTarget` is an enum defined in `human_base.py` and exported from `malariasim`:

```python
class BiteTarget(Enum):
    INDOOR  = 'indoor'
    OUTDOOR = 'outdoor'
    ALL     = 'all'
```

`MalariaHumanBase` and `MalariaHumanODE` both define these methods as the shared
contract. The Updater calls the method without knowing which implementation it is
talking to.

---

## `_prob_acquire_indoor` / `_prob_acquire_outdoor` Reset

`reset_acquisition_probability()` is called at the end of `_do_disease_transitions()`,
which runs inside `step_state()` each tick. This resets both `_prob_acquire_indoor` and
`_prob_acquire_outdoor` to 1.0 for all alive agents before any connectors run. The
reset happens in `_do_disease_transitions()` immediately after calling `super().step_state()`:

```python
def _do_disease_transitions(self):
    super().step_state()
    self.reset_acquisition_probability()
```

For `MalariaHumanODE`, `reset_acquisition_probability()` is called directly from
`step_state()`, which runs in the disease step slot before connectors.

Because connectors run after diseases in Starsim's execution order, the `WaningEffectUpdater`
and `BednetUpdater` always write into a clean 1.0 baseline — there is no accumulation
across steps.

---

## `BiteTarget` enum

```python
class BiteTarget(Enum):
    INDOOR  = 'indoor'
    OUTDOOR = 'outdoor'
    ALL     = 'all'
```

- **Vaccine** (`MalariaVaccineUpdater`): uses `BiteTarget.ALL` — writes both
  `_prob_acquire_indoor` and `_prob_acquire_outdoor`.
- **Bednet** (`BednetUpdater`): uses `BiteTarget.INDOOR` — writes only
  `_prob_acquire_indoor`. Outdoor bites are unaffected by bednets.

For ABM (`MalariaHumanBase`), `set_acquisition_probability` applies a multiplicative
reduction:

```python
reduction = 1.0 - efficacy * weights
if target in (BiteTarget.ALL, BiteTarget.INDOOR):
    self._prob_acquire_indoor[uids]  *= reduction
if target in (BiteTarget.ALL, BiteTarget.OUTDOOR):
    self._prob_acquire_outdoor[uids] *= reduction
```

For ODE (`MalariaHumanODE`), it is an assignment (not multiplicative), since the ODE
applies coverage via weights:

```python
val = 1.0 - weights * efficacy
if target in (BiteTarget.ALL, BiteTarget.INDOOR):
    self._prob_acquire_indoor[uids]  = val
if target in (BiteTarget.ALL, BiteTarget.OUTDOOR):
    self._prob_acquire_outdoor[uids] = val
```

The values are consumed in the transmission step:

- **ABM** (`_p_inf_bite_callable`):
  ```python
  eir_in  = eir_indoor[node_ids]  * _prob_acquire_indoor[uids]
  eir_out = eir_outdoor[node_ids] * _prob_acquire_outdoor[uids]
  p = (eir_in + eir_out) * prob_human_infected_on_bite * rel_biting_rate[uids]
  ```

- **ODE** (vectorized Euler step):
  ```python
  new_inf = b * Sh_mat * (eir_in_dt * pa_in_mat + eir_out_dt * pa_out_mat)
  ```

---

## What the User Writes

`build_campaign` is a function that takes `sim` and an `eir_only` boolean and returns a
`Campaign` object. The function itself — not the result — is passed to `ms.Sim` via
`build_campaign=`. `ms.Sim` calls it after setting up Updater infrastructure, passing
itself so that `Campaign.add()` can validate distributor start years immediately.

**`eir_only` flag** — `True` when `mosquito='eir_only'`; `False` for `'agents'` and
`'ode'`.  Both `Outbreak` and `MosquitoesEirOnly` infect human agents directly —
`Outbreak` via `set_prognoses()` and `MosquitoesEirOnly` via an inoculation rate
applied each step.  They can be used together, but typically are not: when mosquitoes
are explicit agents, `Outbreak` is the standard way to seed an initial infection load;
when using `MosquitoesEirOnly`, the EIR itself drives infections from the start and
`Outbreak` is usually unnecessary and can interfere with clean experiments.  The flag
lets a single `build_campaign` function handle all mosquito modes correctly:

```python
def build_campaign(sim, eir_only=False):
    waning  = WaningEffect(
        initial_efficacy    = 0.80,
        box_duration        = ss.days(180),
        decay_time_constant = ss.days(730),
    )
    vaccine = MalariaVaccine(waning=waning)
    bednet  = Bednet(
        repelling = WaningEffect(initial_efficacy=0.5, box_duration=ss.days(0),   decay_time_constant=ss.days(730)),
        blocking  = WaningEffect(initial_efficacy=0.9, box_duration=ss.days(365), decay_time_constant=ss.days(365)),
        killing   = WaningEffect(initial_efficacy=0.6, box_duration=ss.days(180), decay_time_constant=ss.days(365)),
        usage     = WaningEffect(initial_efficacy=0.8, box_duration=ss.days(365), decay_time_constant=ss.days(365)),
    )
    distributors = []
    if not eir_only:
        # With explicit mosquito agents, Outbreak seeds the initial infection load.
        # With MosquitoesEirOnly the EIR drives infections directly, so Outbreak is
        # usually omitted to avoid double-seeding.
        distributors.append(InterventionDistributorScheduled(
            name='seed', products=Outbreak(), start=2000, coverage=1.0,
        ))
    distributors.append(InterventionDistributorScheduled(name='vacc_2005', products=vaccine, start=ss.date(2005), coverage=0.50))
    distributors.append(InterventionDistributorScheduled(name='itn_2006',  products=bednet,  start=ss.date(2006), coverage=0.80),
    return Campaign(sim, distributors)

sim = ms.Sim(..., build_campaign=build_campaign)
```

The `eir_only` flag is set automatically by `ms.Sim` based on the `mosquito=` argument
— the user never passes it explicitly.

---

## Updater Auto-Registration

All updaters inherit from a common `Updater` base class. The base class maintains a
static dictionary `_possible_updaters` and uses Python's `__init_subclass__` hook to
register each subclass automatically. The moment a subclass is defined, it instantiates
itself and adds the instance to that dictionary under its `_updater_key`:

```python
class Updater:
    _possible_updaters: dict[str, 'Updater'] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._updater_key is not None:
            Updater._possible_updaters[cls._updater_key] = cls()
```

All Updater subclasses live in the `malariasim/updaters/` directory — one file per
ProductComponent/Product/Updater group (e.g. `vaccine.py`, `bednet.py`, `waning.py`). The
`updaters/__init__.py` uses `importlib` to automatically import every module in the
directory, so the registration fires for all of them without any list to maintain:

```python
# malariasim/updaters/__init__.py
import importlib
import pkgutil
from pathlib import Path

for _mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
    importlib.import_module(f'.{_mod.name}', package=__name__)
```

Adding a new intervention is then a single step: create a new file in `updaters/` with
the ProductComponent, Product, and paired Updater subclass. No other file needs to change.

---

## What `ms.Sim` Does With It

Before calling `build_campaign()`, `ms.Sim` reads `Updater._possible_updaters` —
populated automatically when the `updaters` package was imported — and calls `clear()`
on each updater. `clear()` wipes any state requirements accumulated by a previous
`build_campaign()` call — important when the same process constructs multiple `ms.Sim`
instances in sequence:

```python
for updater in Updater._possible_updaters.values():
    updater.clear()
```

`ms.Sim` then calls `Updater.activate()` and invokes `build_campaign()`. The
`Updater._active` flag being `False` outside this window means any Product instantiated
at the wrong time will raise a clear `RuntimeError` from `Updater.get()` rather than
silently registering into a stale updater.

As each Product is instantiated inside `build_campaign()`, its `__init__` calls
`Updater.get(key).register(prefix, target)`, telling the Updater what state arrays it
will need — prefix name and target location (disease module or NodeState).

After `build_campaign()` returns, `ms.Sim` calls `Updater.deactivate()`,
calls `campaign.as_interventions()` to get the flat distributor list, and registers the
updaters as connectors and the distributors as interventions with `ss.Sim`:

```python
connectors = [node_state] + extra_updaters + [mos]
```

`extra_updaters` is `list(Updater._possible_updaters.values())` — all registered
updaters, in import order (`waning.py` → `vaccine.py` → `bednet.py`). This ordering
is the required step sequence: `WaningEffectUpdater` must update efficacy before
`MalariaVaccineUpdater` writes `_prob_acquire`, and `BednetUpdater` must run after
`WaningEffectUpdater` so efficacy values are current.

During `init_pre()`, each Updater resolves its target module handles and calls
`_inject_waning_states()` to create per-person arrays on the target module for all
registered prefixes — initialised to sentinel values. All agents start inactive. No
arrays are created at distribution time. Injection happens in `init_pre()` (not
`init_post()`) because connectors run before diseases in Starsim's initialisation order,
ensuring arrays exist on the disease module before `add_module()` links them to People.

---

## Connector Ordering

The full connector list is:

```
[NodeState] + [WaningEffectUpdater, MalariaVaccineUpdater, BednetUpdater] + [MosquitoConnector]
```

Each timestep the execution order is:

1. **Human disease `step_state()`** — resets `_prob_acquire_indoor` and `_prob_acquire_outdoor`
   to 1.0 for all alive agents.
2. **NodeState.`step()`** — resets `itn_biting_reduction` and `itn_kill_prob` to 0.0
   for all nodes. Serves as an ordering anchor between the human disease and subsequent
   connectors.
3. **WaningEffectUpdater.`step()`** — updates `{prefix}_efficacy` for all affected agents.
   Runs before any downstream updater so efficacy values are current.
4. **MalariaVaccineUpdater.`step()`** — reads current `vaccine_efficacy`, calls
   `disease.sync_vaccinated_fraction()` then `disease.set_acquisition_probability()`
   for all vaccinated agents.
5. **BednetUpdater.`step()`** — reads current ITN efficacy values, writes
   `_prob_acquire_indoor` on the disease module and `itn_biting_reduction` /
   `itn_kill_prob` on NodeState.
6. **MosquitoConnector.`step()`** — reads `p_inf_indoor/outdoor` and `n_humans` from
   NodeState (written by human disease in the previous step), reads `itn_biting_reduction`
   and `itn_kill_prob` (just written by BednetUpdater), advances mosquito compartments,
   writes `eir_indoor` and `eir_outdoor` to NodeState for the human module to read next
   step.
7. **Human disease `step()`** — reads EIR from NodeState (one-step lag), applies
   M→H transmission.

---

## Bednet (Insecticide-Treated Net, ITN)

### Four waning effects

`Bednet` is a `ss.Product` with four `WaningEffect` components:

| Effect | Registers prefix | Mechanism |
|--------|-----------------|-----------|
| `repelling` | `itn_repelling` | Mosquito repelled from entering house |
| `blocking`  | `itn_blocking`  | Mosquito enters house but net blocks access to person |
| `killing`   | `itn_killing`   | Mosquito contacts net and is killed by insecticide; conditioned on not repelled and on blocking |
| `usage`     | `itn_usage`     | Whether the net is in use — scales all three efficacy effects |

### Human-side effect (indoor bites only)

`BednetUpdater.apply_immediate(uids)` computes:

```python
u_rep   = eff_usage * eff_rep
u_block = eff_usage * eff_block
u_kill  = eff_usage * (1.0 - eff_rep) * eff_block * eff_kill  # killing conditioned on not repelled and on blocking

# Combined fraction of indoor bites blocked by either repelling or blocking
combined_block = 1.0 - (1.0 - u_rep) * (1.0 - u_block)
```

It then calls:

```python
itn_weights = self._disease.get_itn_weights(uids)
self._disease.set_acquisition_probability(uids, combined_block, itn_weights, target=BiteTarget.INDOOR)
```

`BiteTarget.INDOOR` means only `_prob_acquire_indoor` is written. Outdoor bites are
unaffected. A vaccinated person sleeping under an ITN has `_prob_acquire_indoor`
reduced by both effects (the ABM implementation multiplies in both reductions because
`set_acquisition_probability` uses `*=`; the ODE uses an assignment with the combined
weight).

### Mosquito-side effect (per node, via NodeState)

`BednetUpdater` calls `disease.aggregate_itn_to_node_state(uids, combined_block, u_kill, ns)`,
which writes to two `NodeState` fields:

```
itn_biting_reduction[node]  — fraction of bite attempts blocked ∈ [0,1]
itn_kill_prob[node]         — kill probability per bite attempt ∈ [0,1]
                              (mosquito module multiplies by a*dt to get per-step kill fraction)
```

Both fields are reset to 0.0 by `NodeState.step()` at the start of each tick before
`BednetUpdater` runs.

**ABM aggregation** (`MalariaHumanBase.aggregate_itn_to_node_state`): each ITN-covered
agent counts as one person:

```python
ns.itn_biting_reduction += np.bincount(node_ids, weights=combined_block, minlength=n_nodes) / n_total
ns.itn_kill_prob         += np.bincount(node_ids, weights=u_kill,         minlength=n_nodes) / n_total
```

`n_total` comes from `ns.n_humans` (per-node human counts written by the human disease
`step()` in the previous timestep).

**ODE aggregation** (`MalariaHumanODE.aggregate_itn_to_node_state`): compartments are
population-weighted by `_itn_fraction * (Sh + Ih)`:

```python
pop_w = self._itn_fraction[uids] * (self._Sh[uids] + self._Ih[uids])
ns.itn_biting_reduction += np.bincount(node_ids, weights=combined_block * pop_w, minlength=n_nodes) / n_total
ns.itn_kill_prob         += np.bincount(node_ids, weights=u_kill         * pop_w, minlength=n_nodes) / n_total
```

### NodeState ITN fields

```python
# NodeState.__init__
self.itn_biting_reduction = np.zeros(n)  # float ∈ [0,1] — fraction of bites blocked per node
self.itn_kill_prob        = np.zeros(n)  # float ∈ [0,1] — kill probability per bite attempt

# NodeState.step() — resets each tick before BednetUpdater writes
self.itn_biting_reduction[:] = 0.0
self.itn_kill_prob[:]        = 0.0
```

### Mosquito module reads

**ODE mosquito** (`MosquitoesODE`), per-node Euler step:

```python
f        = ns.grid.f_indoor[nid]         # indoor bite fraction for this node
itn_br   = ns.itn_biting_reduction[nid]
mu_itn   = a * f * ns.itn_kill_prob[nid] # only indoor-feeding mosquitoes encounter the net
a_eff_indoor = a * f * (1.0 - itn_br)
a_outdoor    = a * (1.0 - f)
lambda_m = a_eff_indoor * p_inf_indoor[nid] + a_outdoor * p_inf_outdoor[nid]
mu_tot   = mu_m + mu_itn
```

EIR written to NodeState also reflects the reduction (via `_write_node_eir`):

```python
a_eff = a * (1.0 - f_arr * itn_br)  # outdoor bites unaffected; only indoor fraction reduced
eir_dt = a_eff * Im * dt_days / n_h
ns.eir_indoor[:]  = eir_dt * f_arr
ns.eir_outdoor[:] = eir_dt * (1.0 - f_arr)
```

**Agent mosquito** (`MosquitoesAdultFemaleAgentsOnly`):

- **Killing** (`_apply_itn_killing`): extra mortality per step scaled by indoor bite
  fraction and `itn_kill_prob`:
  ```python
  p_kill = a * dt_days * f_arr[nids] * itn_kp[nids]   # clipped to [0,1]
  ```
- **Exposure** (`_expose_susceptible_mosquitoes`): indoor biters have a reduced bite
  probability via `itn_biting_reduction`:
  ```python
  p_bite_indoor = p_bite * (1.0 - itn_br)   # itn_br for this mosquito's node
  p_bite_out    = p_bite                     # outdoor unaffected
  p_exposed = where(bites_indoor, p_bite_indoor * p_inf_indoor, p_bite_out * p_inf_outdoor)
  ```
- **EIR** (`_write_node_eir`):
  ```python
  a_eff = bites_per_female_per_day * (1.0 - f_arr * itn_br)
  ```

### ODE vs ABM coverage handling for bednets

The bednet follows the same ABM/ODE coverage pattern as the vaccine, but with
`_itn_fraction` rather than `_vaccinated_fraction`:

**ABM** (`MalariaHumanBase`):
- `apply_coverage` applies a Bernoulli draw and returns `(chosen_uids, ones)`.
- `store_itn_fraction(uids, weights)` is a **no-op** — coverage is implicit in which agents were selected.
- `get_itn_weights(uids)` returns `np.ones(len(uids))` — each covered agent is fully weighted (1.0).
- `aggregate_itn_to_node_state` does plain bincount / n_total.

**ODE** (`MalariaHumanODE`):
- `apply_coverage` returns ALL eligible compartments with `weights = coverage` (a
  continuous fraction, no Bernoulli step).
- `store_itn_fraction(uids, weights)` stores `_itn_fraction[uids] = weights`.
- `get_itn_weights(uids)` returns `np.asarray(self._itn_fraction[uids])`.
- `aggregate_itn_to_node_state` does population-weighted bincount using
  `_itn_fraction * (Sh + Ih)` as the per-compartment weight.

`_itn_fraction` is an `ss.FloatArr` state on `MalariaHumanODE` (default 0.0), defined
in `MalariaHumanODE.__init__` alongside `_vaccinated_fraction`.

Note that `Bednet.administer()` does **not** call `pre_administer` (which is a vaccine
concept for tracking weighted-mean dose time). Instead it calls `store_itn_fraction`
directly and passes a scalar `t_eff = np.full(len(uids), self.sim.t.year)` to
`WaningEffectUpdater.apply()`:

```python
def administer(self, uids, weights=None):
    if weights is None:
        weights = np.ones(len(uids), dtype=float)
    self._disease.store_itn_fraction(uids, weights)
    t_eff = np.full(len(uids), self.sim.t.year, dtype=float)
    self._waning_updater.apply(PREFIX_REPELLING, uids, self.repelling, t_eff)
    self._waning_updater.apply(PREFIX_BLOCKING,  uids, self.blocking,  t_eff)
    self._waning_updater.apply(PREFIX_KILLING,   uids, self.killing,   t_eff)
    self._waning_updater.apply(PREFIX_USAGE,     uids, self.usage,     t_eff)
    if self._bednet_updater is not None:
        self._bednet_updater.apply_immediate(uids)
```

The final `apply_immediate` call is needed because `BednetUpdater` already ran this
step before the distributor fired; this ensures the effect is applied in the same
timestep as distribution.

---

## Intervention Effects in `MalariaHumanODE`

### Design principle: one campaign, all models

A single `build_campaign` function must work unchanged with every human disease model
— `MalariaHumanSIS`, `MalariaHumanSEIS`, and `MalariaHumanODE`. The user writes the
campaign once; the disease module is responsible for translating campaign intent into
the appropriate representation.

For ABM models, that means writing per-agent state arrays. For the ODE, it means
updating compartmental flow parameters. The `WaningEffect` profile, coverage value,
node targets, and age targets are identical — only what the disease module does with
them differs.

### The interface

All human disease models implement the same methods so that updaters and eligibility
objects are unaware of which model they are talking to:

```python
apply_coverage(uids, coverage)
    # → (ss.uids, weights); called by distributor after eligibility filtering
    # ABM: pars._coverage_dist.set(p=coverage).filter(uids) → (chosen, ones)
    # ODE: (uids, full(coverage)) — no bernoulli, coverage as weight

pre_administer(uids, weights, t_now)
    # → t_effective per uid; called from MalariaVaccine.administer() before WaningEffectUpdater.apply()
    # ABM: returns np.full(len(uids), t_now) — no state written
    # ODE: updates _vaccinated_fraction and _dose_time; returns weighted-mean per uid

set_acquisition_probability(uids, efficacy, weights, target=BiteTarget.ALL)
    # write _prob_acquire_indoor and/or _prob_acquire_outdoor
    # ABM: multiplicative (*= 1 - efficacy * weights)
    # ODE: assignment (= 1 - weights * efficacy)

reset_acquisition_probability()
    # reset both to 1.0 each step before connectors run

sync_vaccinated_fraction(uids, efficacy)
    # reset expired ODE tracking state; no-op for ABM

get_vaccinated_weights(uids)
    # → weights per uid; ones for ABM, _vaccinated_fraction for ODE

store_itn_fraction(uids, weights)
    # ODE: stores _itn_fraction[uids] = weights; no-op for ABM

get_itn_weights(uids)
    # → weights per uid; ones for ABM, _itn_fraction for ODE

aggregate_itn_to_node_state(uids, combined_block, u_kill, ns)
    # write itn_biting_reduction and itn_kill_prob on NodeState
    # ABM: plain bincount / n_total
    # ODE: population-weighted bincount using _itn_fraction * (Sh + Ih)

set_drug_clearance(uids, mean_duration_years, weights)
    # ABM: draws exponential duration, sets ti_recovered for infected agents
    # ODE: sets _drug_coverage, _drug_clearance_rate, _drug_clear_end_t

administer_outbreak(uids, weights)
    # ABM: infects susceptible agents via set_prognoses()
    # ODE: moves weights * Sh[uid] from susceptible to infectious

validate_eligibility(eligibility)
    # ABM: no-op
    # ODE: raises ValueError if min_age/max_age don't align to bin edges
```

`_supports_intervention_effects = True` remains the duck-type marker used to find the
module; the method names are the contract.

### ABM implementation (`MalariaHumanBase`)

Per-agent: two `ss.FloatArr` states, both with default 1.0, defined in
`define_intervention_states()`:

- `_prob_acquire_indoor` — acquisition probability multiplier for indoor bites
- `_prob_acquire_outdoor` — acquisition probability multiplier for outdoor bites

`reset_acquisition_probability()` sets both arrays to 1.0 for all alive agents.

`pre_administer(uids, weights, t_now)` returns `np.full(len(uids), t_now)` — every
agent is dosed at the current time. No state is written.

`set_acquisition_probability(uids, efficacy, weights, target=BiteTarget.ALL)` writes:

```python
reduction = 1.0 - efficacy * weights
if target in (BiteTarget.ALL, BiteTarget.INDOOR):
    self._prob_acquire_indoor[uids]  *= reduction
if target in (BiteTarget.ALL, BiteTarget.OUTDOOR):
    self._prob_acquire_outdoor[uids] *= reduction
```

`sync_vaccinated_fraction(uids, efficacy)` is a no-op.

`get_vaccinated_weights(uids)` returns `np.ones(len(uids))`.

`store_itn_fraction(uids, weights)` is a no-op.

`get_itn_weights(uids)` returns `np.ones(len(uids))`.

`apply_coverage(uids, coverage)` applies `self.pars._coverage_dist.set(p=coverage).filter(uids)` and
returns `(chosen_uids, np.ones(len(chosen_uids)))`. `_coverage_dist` is a
`ss.bernoulli(p=0.5)` stored in `define_pars` rather than created fresh each call —
this preserves Common Random Numbers (CRN).

`set_drug_clearance(uids, mean_duration_years, weights)` filters to currently-infected
agents then draws exponential clearance durations using a `dur_drug = ss.expon(scale=1.0)`
distribution stored in `define_pars` (scale is set per-call to preserve CRN):

```python
infected = np.asarray(self.infected[uids], dtype=bool)
targets  = ss.uids(np.asarray(uids)[infected])
if len(targets):
    dt_year   = self.sim.t.dt_year
    dur_steps = self.pars.dur_drug.set(scale=mean_duration_years).rvs(targets) / dt_year
    self.ti_recovered[targets] = self.sim.ti + dur_steps
```

Susceptible and exposed agents are no-ops. Weights are ignored in the ABM (Bernoulli
sampling in `apply_coverage` already selected only covered agents).

### ODE implementation (`MalariaHumanODE`)

#### Per-compartment state arrays

`MalariaHumanODE` defines the following `ss.FloatArr` states (in addition to `_Sh` and `_Ih`):

```
_prob_acquire_indoor   — default 1.0
_prob_acquire_outdoor  — default 1.0
_vaccinated_fraction   — default 0.0 — fraction of compartment currently vaccinated
_itn_fraction          — default 0.0 — ITN-covered fraction per compartment
_drug_coverage         — default 0.0 — fraction covered by active drug dose
_drug_clear_end_t      — default -1.0 — sim time (years) when drug window closes (sentinel -1)
_drug_clearance_rate   — default 0.0 — 1/mean_duration_years
_dose_time             — default 0.0 — population-weighted mean vaccine dose time
node_id                — default 0   — node index (IntArr)
```

All are sized to `n_agents = n_nodes * n_bins`.

#### UID mapping

UIDs are assigned in row-major order: `uid = nid * n_bins + bin`. In `init_post()`:

```python
self.sim.people.age[all_uids] = bin_min_ages[uid_arr % n_bins]  # enables age filtering
self.node_id[all_uids]        = uid_arr // n_bins                # enables node filtering
```

#### `apply_coverage`: ODE coverage as continuous fraction

```python
def apply_coverage(self, uids, coverage):
    return uids, np.full(len(uids), float(coverage), dtype=float)
```

No Bernoulli step. `weights=coverage` means each product receives a continuous fraction
rather than a binary selected/not flag.

#### `pre_administer`: weighted-mean dose time before waning arrays are written

`pre_administer(uids, weights, t_now)` is called by `MalariaVaccine.administer()`
**before** `WaningEffectUpdater.apply()`. `weights` here is the coverage delta `c_new`
for each compartment. The method updates `_vaccinated_fraction` and `_dose_time` in
place and returns a per-uid array of effective dose times:

```python
f_old = self._vaccinated_fraction[uids]
t_old = self._dose_time[uids]
f_new = f_old + c_new * (1.0 - f_old)
safe  = f_new > 0.0
t_eff = np.where(
    safe,
    (f_old * t_old + c_new * (1.0 - f_old) * t_now) / np.where(safe, f_new, 1.0),
    t_now,
)
self._vaccinated_fraction[uids] = f_new
self._dose_time[uids]           = t_eff
return t_eff   # shape (len(uids),) — passed to WaningEffectUpdater.apply()
```

`WaningEffectUpdater.apply()` receives `t_eff` as a per-uid array and writes
per-compartment `vaccine_box_end` and `vaccine_decay_end` from the correct effective
dose time for each compartment.

#### `set_acquisition_probability` on the ODE

```python
val = 1.0 - weights * efficacy
if target in (BiteTarget.ALL, BiteTarget.INDOOR):
    self._prob_acquire_indoor[uids]  = val
if target in (BiteTarget.ALL, BiteTarget.OUTDOOR):
    self._prob_acquire_outdoor[uids] = val
```

For the vaccine, `weights` at this call site is `_vaccinated_fraction` (returned by
`get_vaccinated_weights`). For the bednet, `weights` is `_itn_fraction` (returned by
`get_itn_weights`).

#### `sync_vaccinated_fraction`: reset expired tracking state

`sync_vaccinated_fraction(uids, efficacy)` is called by `MalariaVaccineUpdater.step()`
before `set_acquisition_probability` each step. It resets tracking state for any
compartment whose `vaccine_efficacy` has reached zero:

```python
expired = ss.uids(uids[efficacy[uids] == 0.0])
if len(expired) == 0:
    return
self._vaccinated_fraction[expired] = 0.0
self._dose_time[expired]           = 0.0
```

#### `store_itn_fraction` and `get_itn_weights`

```python
def store_itn_fraction(self, uids, weights):
    self._itn_fraction[ss.uids(np.asarray(uids))] = np.asarray(weights)

def get_itn_weights(self, uids):
    return np.asarray(self._itn_fraction[ss.uids(np.asarray(uids))])
```

These parallel `pre_administer` / `get_vaccinated_weights` for the vaccine but are
simpler — bednets do not accumulate fractions across campaigns because each campaign
writes the full coverage directly.

#### Vaccine pattern summary for ODE

```
1. t_eff = disease.pre_administer(uids, weights, t_now)
       → updates _vaccinated_fraction and _dose_time (weighted-mean)
       → returns per-uid effective dose time

2. efficacy0 = waning_updater.apply(prefix, uids, waning_effect, t_eff)
       → writes vaccine_box_end[uid]   = t_eff + box_duration
       → writes vaccine_decay_end[uid] = t_eff + box_duration + tau*ln(100)
       → writes vaccine_efficacy[uid]  = efficacy0
       → returns efficacy0

3. vax_weights = disease.get_vaccinated_weights(uids)
       → reads _vaccinated_fraction already updated by pre_administer in step 1

4. disease.set_acquisition_probability(uids, efficacy0, vax_weights, target=BiteTarget.ALL)
       → writes _prob_acquire_indoor[uids]  = 1 - _vaccinated_fraction[uids] * efficacy0
       → writes _prob_acquire_outdoor[uids] = 1 - _vaccinated_fraction[uids] * efficacy0
```

#### `MalariaVaccineUpdater.step()` per-step sequence

```python
ever_dosed = self._waning_updater.active_mask(MalariaVaccine.PREFIX)
efficacy   = self._waning_updater.current_efficacy(MalariaVaccine.PREFIX)
self._disease.sync_vaccinated_fraction(ever_dosed.uids, efficacy)
active_uids = (ever_dosed & (efficacy > 0)).uids
if len(active_uids):
    vax_weights = self._disease.get_vaccinated_weights(active_uids)
    self._disease.set_acquisition_probability(
        active_uids, efficacy[active_uids], vax_weights, target=BiteTarget.ALL
    )
```

#### Aging of `_vaccinated_fraction` and `_dose_time`

Each step, after computing the Euler fluxes for `Sh` and `Ih`, the ODE calls
`_age_vaccinated_state()` to propagate the vaccinated fraction across age bins:

```python
V        = vaccinated_fraction * (Sh + Ih)   # vaccinated count before aging
V_out    = V * (aging_dt + death_dt)
V_in[..., 1:] = V_out[..., :-1]
V_new    = np.maximum(V - V_out + V_in, 0.0)
new_fraction = V_new / (Sh_new + Ih_new)     # uses post-Euler population as denominator
T_new = where(V_new > 0,
              ((V - V_out) * dose_time + V_in * np.roll(dose_time, 1)) / V_new,
              0.0)
T_new[..., 0] = 0.0   # births are unvaccinated
```

`np.roll(dose_time, 1)` gives the dose time of the cohort aging in from the younger
bin. Bin 0 always has `T_new = 0` since births are unvaccinated.

#### Waning

The same `WaningEffectUpdater` serves both ABM and ODE. The ODE's compartment UIDs
look identical to agent UIDs from the updater's perspective.

---

## AntiMalarialDrug

### Design

`AntiMalarialDrug` is a `ss.Product` with a single parameter: `mean_duration` (an
`ss.days` value giving the exponential mean clearance time). It has no paired Updater
— the ABM delegates to SIS machinery and the ODE handles clearance inline in its own
`step()`.

```python
drug = AntiMalarialDrug(mean_duration=ss.days(1.5))
```

`administer(uids, weights)` simply calls:

```python
self._disease.set_drug_clearance(uids, self.mean_duration.years, weights)
```

### ABM (`MalariaHumanBase.set_drug_clearance`)

Filters to currently-infected agents, draws exponential clearance durations from
`pars.dur_drug` (a `ss.expon` distribution stored in `define_pars`, scale set
per-call to preserve CRN), and overwrites `ti_recovered` so the existing SIS
recovery loop clears the infection at the drawn time:

```python
infected  = np.asarray(self.infected[uids], dtype=bool)
targets   = ss.uids(np.asarray(uids)[infected])
dur_steps = self.pars.dur_drug.set(scale=mean_duration_years).rvs(targets) / dt_year
self.ti_recovered[targets] = self.sim.ti + dur_steps
```

Susceptible and exposed agents are no-ops. Weights are ignored.

### ODE (`MalariaHumanODE.set_drug_clearance`)

```python
def set_drug_clearance(self, uids, mean_duration_years, weights):
    uids = ss.uids(np.asarray(uids))
    self._drug_coverage[uids]       = np.asarray(weights)
    self._drug_clear_end_t[uids]    = self.sim.t.year + mean_duration_years
    self._drug_clearance_rate[uids] = 1.0 / mean_duration_years
```

Each timestep, `MalariaHumanODE.step()` applies the drug clearance inline before the
standard Euler update:

```python
# 1. Expire finished drug doses
expired = (self._drug_clear_end_t > 0) & (self._drug_clear_end_t <= t)
if len(expired.uids):
    self._drug_coverage[expired.uids]       = 0.0
    self._drug_clear_end_t[expired.uids]    = -1.0
    self._drug_clearance_rate[expired.uids] = 0.0

# 2. Apply clearance for active doses
dc_mat     = self._drug_coverage.raw.reshape(n_nodes, n_bins)
dr_mat     = self._drug_clearance_rate.raw.reshape(n_nodes, n_bins)
drug_clear = dr_mat * dt_year * dc_mat * Ih_mat   # (n_nodes, n_bins)

Sh_new = ... + drug_clear ...
Ih_new = ... - drug_clear ...
```

`_drug_clearance_rate` (= `1/mean_duration_years`) is stored at administration time so
`step()` can read it without parameters.

### Distribution choice: exponential

Observed parasite clearance times are right-skewed. The exponential is used as a
deliberate simplification:

- **Discretisation swamps the difference.** With 1-day timesteps the distinction
  between exponential, gamma, and log-normal is largely lost.
- **Right-skew is preserved.** The exponential is itself right-skewed.
- **ODE equivalence.** The exponential mean maps exactly to an ODE clearance rate
  `r = 1/mean_duration`, giving a clean, analytically consistent cross-model design.
- **Simplicity.** One parameter (`mean_duration`) instead of two (mean + SD).

A recommended default is `mean_duration=ss.days(1.5)`.

### Known limitation: single active drug dose per ODE compartment

The ODE tracks a single `_drug_coverage` fraction per compartment. A second drug
distribution before the first has expired will overwrite `_drug_coverage` and
`_drug_clear_end_t`, losing the first dose. This is accepted because typical campaigns
space doses far enough apart that the clearance window from the first dose has closed
before the second arrives.

---

## Timestep Walkthrough

### Before distribution (`t < 2005`)

```
diseases.step_state  — _do_disease_transitions() resets _prob_acquire = 1.0
connectors.step:
    NodeState            — resets itn_biting_reduction = 0, itn_kill_prob = 0
    WaningEffectUpdater  — decay_end arrays all sentinel (-1), nothing to do
    MalariaVaccineUpdater — vaccine_decay_end all sentinel, nothing to do
    BednetUpdater        — no active agents (itn_usage_decay_end all sentinel), nothing to do
    MosquitoConnector    — reads NodeState, advances mosquitoes, writes EIR
interventions.step:
    InterventionDistributorScheduled (2005) — t < 2005, does nothing
diseases.step        — reads EIR from NodeState, runs M→H transmission
```

### The distribution timestep (`t = 2005`)

```
diseases.step_state  — resets _prob_acquire = 1.0
connectors.step:
    NodeState            — resets itn_biting_reduction = 0, itn_kill_prob = 0
    WaningEffectUpdater  — still no active recipients, nothing to do
    MalariaVaccineUpdater — still nothing to do
    BednetUpdater        — still nothing to do
    MosquitoConnector    — reads NodeState, writes EIR
interventions.step:
    InterventionDistributorScheduled (2005) — fires:
        eligible      = eligibility(sim)
        uids, weights = disease.apply_coverage(eligible, coverage=0.50)
        calls vaccine.administer(uids, weights):
            t_eff = disease.pre_administer(uids, weights, t_now)
            efficacy = waning_updater.apply('vaccine', uids, waning, t_eff)
            vax_weights = disease.get_vaccinated_weights(uids)
            disease.set_acquisition_probability(uids, efficacy, vax_weights, target=BiteTarget.ALL)
diseases.step        — reads _prob_acquire (already adjusted), unaware of vaccines
```

### Subsequent timesteps

```
diseases.step_state  — _do_disease_transitions() resets _prob_acquire = 1.0
connectors.step:
    NodeState            — resets itn_biting_reduction = 0, itn_kill_prob = 0
    WaningEffectUpdater  — updates {prefix}_efficacy for dosed agents
                           (box phase: unchanged; decaying: exp decay; expired: 0.0)
    MalariaVaccineUpdater — reads vaccine_efficacy, calls sync_vaccinated_fraction(),
                            then set_acquisition_probability() for active recipients
    BednetUpdater        — if bednets distributed: applies per-agent effects,
                           writes itn_biting_reduction and itn_kill_prob to NodeState
    MosquitoConnector    — reads itn_biting_reduction and itn_kill_prob from NodeState,
                           advances mosquitoes, writes EIR
diseases.step        — reads _prob_acquire (already adjusted), reads EIR
```

---

## Sweeping Intervention Parameters

Since `build_campaign` is just a callable, sweeping intervention parameters requires no
new machinery. The builder function takes parameters directly; `functools.partial` binds
them per sim:

```python
def build_campaign(sim, efficacy):
    waning  = WaningEffect(
        initial_efficacy    = efficacy,
        box_duration        = ss.days(180),
        decay_time_constant = ss.days(730),
    )
    vaccine = MalariaVaccine(waning=waning)
    return Campaign(sim, [
        InterventionDistributorScheduled(name='vacc_2005', products=vaccine,
                                         start=ss.date(2005), coverage=0.80),
    ])

from functools import partial

sims = [
    ms.Sim(..., build_campaign=partial(build_campaign, efficacy=e), label=f'efficacy={e}')
    for e in [0.60, 0.70, 0.80, 0.90]
]
msim = ms.MalariaMultiSim(sims)
```

`partial(build_campaign, efficacy=0.80)` returns a callable that accepts one argument
(`sim`) — exactly what `ms.Sim` passes to `build_campaign=`. Sweeping across both
model configuration and intervention parameters composes naturally:

```python
sims = [
    ms.Sim(..., human=h, build_campaign=partial(build_campaign, efficacy=e))
    for h in ['sis', 'seis']
    for e in [0.60, 0.80]
]
```

---

## Divergence Summary

| Starsim convention | Our approach | Reason |
|---|---|---|
| One `ss.Intervention` per strategy | Product shared across multiple distributors | Biological logic defined once |
| Intervention manages its own per-step update | Updater connector processes all recipients in one pass | O(n_recipients) not O(n_distributors × n_recipients) |
| User adds interventions explicitly | `ms.Sim` auto-adds Updaters as connectors | Updaters are framework internals |
| No registration guard | `Updater.get()` raises if called outside `build_campaign()` window | Prevents silent registration into stale updater |
| Connector ordering incidental | Updater insertion order in `ms.Sim` is significant | Waning must be current before `_prob_acquire` is adjusted; NodeState must reset ITN fields before BednetUpdater writes them |
| State injection in `init_post()` | State injection in `init_pre()` | Connectors run before diseases; arrays must exist before `add_module()` |
| Single mosquito-side effect path | `itn_biting_reduction` and `itn_kill_prob` on NodeState | Decouples human ITN coverage from mosquito module implementation |
