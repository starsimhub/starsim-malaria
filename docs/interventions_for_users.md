# Malaria Interventions — User Guide

This guide explains how to add interventions — vaccines, outbreak seeding, bed nets, drugs — to a malariasim simulation. It is written for malaria researchers; no prior knowledge of Starsim is required.

## 1. Overview

#### Supported interventions

| Intervention | Target | Mechanism |
|---|---|---|
| Vaccine (e.g. RTS,S, R21) | Person | Reduces probability of infection per bite |
| Outbreak (infection seeding) | Person | Moves susceptible individuals to infectious immediately |
| Insecticide-treated bed net (ITN) | Person | Reduces biting rate, kills mosquitoes |
| Antimalarial drug | Person | Clears infection, provides prophylaxis |
| Diagnostic test (RDT, microscopy) | Person | Identifies infected individuals, triggers treatment |
| Ivermectin | Person | Makes blood meal toxic to feeding mosquitoes |
| Larvicide | Node | Reduces mosquito emergence rate |
| Indoor residual spraying (IRS) | Node | Increases indoor mosquito death rate |
| Attractive targeted sugar bait (ATSB) | Node | Kills mosquitoes at sugar sources |
| Mosquito release (SIT, Wolbachia) | Node | Modifies mosquito population |

Vaccines and `Outbreak` are fully implemented. The remaining interventions are planned.

#### One campaign, all models

A core design goal of malariasim is that a single campaign definition works unchanged across every combination of human and mosquito model — agent-based (ABM) or compartmental (ODE). You write your intervention code once and run it against SIS, SEIS, or ODE humans, and against EIR-only, ODE, or agent-based mosquitoes. The framework translates your intent — "vaccinate 80% of under-5s in node 2 starting in 2005" — into the appropriate representation for whichever model is running. Comparing an ABM and an ODE under identical intervention scenarios requires no changes to the campaign code.

#### How it works

You define a `build_campaign` function that describes what interventions to deliver, to whom, and when. You pass this function to `ms.Sim`; the simulation calls it during setup and handles all per-step delivery and waning automatically. A campaign can include any number of delivery events — the same vaccine given in multiple rounds, different vaccines, seeding followed by vaccination — with no limit.

## 2. Quick Start

The minimal pattern is: create a product, wrap it in a distributor with targeting and coverage, add it to a campaign, and pass the campaign builder to `ms.Sim`.

```python
import starsim as ss
import malariasim as ms

def build_campaign(sim, eir_only=False):
    vaccine = ms.MalariaVaccine(
        name   = 'rtsS',
        waning = ms.WaningEffect(
            initial_efficacy    = 0.80,
            box_duration        = ss.days(180),
            decay_time_constant = ss.days(730),
        ),
    )
    campaign = ms.Campaign(sim)
    campaign.add(
        ms.InterventionDistributorScheduled(
            name        = 'vacc_2005_u5',
            products    = vaccine,
            start       = 2005,
            coverage    = 0.80,
            eligibility = ms.HumanEligibility(
                min_age  = ss.years(0.5),
                max_age  = ss.years(5),
                node_ids = [0, 2],
            ),
        )
    )
    return campaign

sim = ms.Sim(
    human          = 'sis',
    mosquito       = 'agents',
    build_campaign = build_campaign,
    start          = 2000,
    stop           = 2015,
)
sim.run()
```

This vaccinates 80% of agents aged 6 months to 5 years in nodes 0 and 2 in 2005, with an RTS,S-like profile: full efficacy for six months, then exponential decay over two years.

> **Note:** `build_campaign` always receives an `eir_only` argument — accept it even if you don't use it. See Section 5 for details.

## 3. Building a Campaign

A campaign is a collection of intervention delivery events. You define it inside a `build_campaign` function, which `ms.Sim` calls during setup.

#### The `Campaign` object

`Campaign(sim)` is a container. It requires `sim` at construction time so it can validate start years immediately and set up internal bookkeeping objects that the user never interacts with directly. Call `add()` once per delivery event — each item is either an `InterventionDistributor` (a timed or triggered delivery) or an `InterventionProgram` (a named multi-delivery strategy like seasonal malaria chemoprevention that expands into multiple distributors internally):

```python
def build_campaign(sim, eir_only=False):
    campaign = ms.Campaign(sim)
    campaign.add(distributor_or_program_1)
    campaign.add(distributor_or_program_2)
    return campaign
```

Each distributor has a unique `name` — required, and used for bookkeeping. Two distributors with the same name in the same campaign raise an error.

#### Multiple rounds of the same vaccine

One product instance can be shared across any number of distributors. A second dose to the same person overwrites the first — the new waning profile replaces the old one:

```python
def build_campaign(sim, eir_only=False):
    vaccine = ms.MalariaVaccine(name='rtsS', waning=ms.WaningEffect(
        initial_efficacy=0.80, box_duration=ss.days(180), decay_time_constant=ss.days(730),
    ))
    campaign = ms.Campaign(sim)
    campaign.add(ms.InterventionDistributorScheduled(
        name='round_1', products=vaccine, start=2005, coverage=0.80,
        eligibility=ms.HumanEligibility(min_age=ss.years(0.5), max_age=ss.years(5)),
    ))
    campaign.add(ms.InterventionDistributorScheduled(
        name='round_2', products=vaccine, start=2007, coverage=0.60,
        eligibility=ms.HumanEligibility(min_age=ss.years(0.5), max_age=ss.years(5)),
    ))
    return campaign
```

#### Seeding infections with `Outbreak`

`Outbreak` immediately moves a fraction of susceptible individuals to infectious. Its most common use is seeding an initial infection load at the start of a simulation, but it can also be used to model importation events at any point during a run.

Whether seeding is needed at the start depends on the mosquito model. When `mosquito='eir_only'`, the externally-specified EIR drives infections from the first timestep — the population will naturally accumulate infections. When mosquitoes are modelled explicitly (`mosquito='agents'` or `mosquito='ode'`), the mosquito population starts uninfected and the feedback loop between humans and mosquitoes takes time to build up; seeding with `Outbreak` establishes a realistic starting prevalence immediately. The `eir_only` argument passed to `build_campaign` lets you handle both cases in one function:

```python
def build_campaign(sim, eir_only=False):
    campaign = ms.Campaign(sim)
    if not eir_only:
        campaign.add(ms.InterventionDistributorScheduled(
            name='seed', products=ms.Outbreak(), start=2000, coverage=1.0,
        ))
    # add vaccines, bednets, etc.
    return campaign
```

`ms.Sim` sets `eir_only` automatically based on the `mosquito=` argument — you never pass it yourself.

#### Validation

`Campaign.add()` validates each distributor immediately:
- The `start` year must fall within the simulation's time range — an out-of-range year raises an error at `add()` time, pointing directly at the offending line.
- Duplicate distributor names are caught when `ms.Sim` collects the campaign, listing each conflict by name, start year, and product type.

## 4. Targeting: Who Gets the Intervention

Targeting is specified via an `eligibility` argument on the distributor. If omitted, all alive individuals in all nodes are eligible. Eligibility filtering is separate from coverage — eligibility defines the pool, coverage determines what fraction of that pool is actually dosed.

#### `HumanEligibility`

```python
ms.HumanEligibility(
    min_age  = ss.years(0.5),   # inclusive lower bound (optional)
    max_age  = ss.years(5),     # exclusive upper bound (optional)
    node_ids = [0, 2],          # spatial nodes to target (optional; default = all nodes)
)
```

All three arguments are optional and can be combined freely. Omitting `min_age` and `max_age` targets all ages. Omitting `node_ids` targets all nodes.

Age bounds follow the standard half-open convention: `min_age <= age < max_age`. A campaign targeting `[0.5, 5)` years selects agents aged 6 months up to but not including their 5th birthday.

#### Age targeting with ODE models

For ABM models, each agent has an exact age. For ODE models, the population is divided into age bins (e.g. [0–5), [5–15), [15–30), ...) and each bin is treated as a unit — it cannot be partially targeted. Age filters select whole bins: a bin is included if its lower edge falls within `[min_age, max_age)`.

**Age ranges must align to bin edges when using an ODE human model.** A range that falls mid-bin selects an unexpected set of bins. If finer age targeting is needed, use narrower bins.

#### Node targeting

`node_ids` is a list of integer node indices. Nodes are numbered from 0 in the order they were defined in `NodeGrid`. Omitting `node_ids` targets all nodes.

```python
# Target only rural nodes (indices 1 and 3)
ms.HumanEligibility(node_ids=[1, 3])

# Target under-5s in urban nodes only
ms.HumanEligibility(min_age=ss.years(0), max_age=ss.years(5), node_ids=[0])
```

## 5. Scheduling and Coverage

#### `InterventionDistributorScheduled`

`InterventionDistributorScheduled` fires exactly once at a specified year. It is the primary tool for mass campaigns, pulse vaccinations, and one-time seeding events.

```python
ms.InterventionDistributorScheduled(
    name        = 'vacc_2005_u5',   # unique name (required)
    products    = vaccine,           # product or list of products
    start       = 2005,              # year to fire (float or ss.date)
    coverage    = 0.80,              # fraction of eligible targets to dose
    eligibility = ms.HumanEligibility(...),  # optional; default = all alive
)
```

`start` can be a plain float year (`2005`, `2005.5`) or an `ss.date`. The distributor fires on the timestep nearest to `start` — for daily timesteps this is exact to the day.

`coverage` is the fraction of the eligible pool that receives the intervention. For ABM models each eligible agent is selected independently with probability `coverage` (Bernoulli draw). For ODE models `coverage` is applied as a continuous fraction directly to each compartment — no randomness is involved.

Repetition is handled by adding multiple distributors with different `start` years. There is no `repeat` or `interval` argument — each round is an explicit `add()` call, which keeps campaign logic transparent and easy to audit.

#### Planned distributors

- **`InterventionDistributorTriggered`** — fires per-person on an event such as an age birthday or a positive diagnostic result. Planned; not yet implemented.
- **`InterventionDistributorReferenceTracking`** — maintains a target coverage level in a population by continuously dosing uncovered or lapsed individuals. Planned; not yet implemented.

## 6. Products

A product defines what an intervention does biologically. The same product instance can be reused across multiple distributors — the biological definition is written once regardless of how many delivery rounds use it.

#### `MalariaVaccine`

`MalariaVaccine` reduces an individual's probability of acquiring infection per infectious bite. Efficacy wanes over time according to a `WaningEffect` profile.

```python
ms.MalariaVaccine(
    name   = 'rtsS',
    waning = ms.WaningEffect(
        initial_efficacy    = 0.80,          # efficacy at time of dosing, in [0, 1]
        box_duration        = ss.days(180),  # constant-efficacy phase
        decay_time_constant = ss.days(730),  # exponential decay time constant τ
    ),
)
```

A second dose to the same person overwrites the first — the new efficacy and waning profile replace the old ones entirely. See Section 7 for details on waning profiles.

#### `Outbreak`

`Outbreak` immediately moves a fraction of susceptible individuals to infectious. It takes no parameters:

```python
ms.Outbreak()
```

Coverage controls what fraction of the eligible pool is infected. `coverage=1.0` with no eligibility filter infects the entire susceptible population. See Section 3 for guidance on when to use `Outbreak`.

#### Planned products

- **`Bednet`** — reduces biting rate and kills mosquitoes
- **`AntimalarialDrug`** — clears infection and provides prophylaxis
- **`Larvicide`** — reduces mosquito emergence rate at a node
- **`IndoorResidualSpraying`** — increases indoor mosquito death rate at a node

## 7. Waning Efficacy

Waning describes how a product's biological effect changes over time after delivery. Any product whose effect diminishes over time — a vaccine, a bed net treatment, a drug, ivermectin — takes a `WaningEffect` object to define that profile.

#### Parameters

```python
ms.WaningEffect(
    initial_efficacy    = 0.80,          # efficacy at delivery time, in [0, 1]
    box_duration        = ss.days(180),  # flat phase before decay begins
    decay_time_constant = ss.days(730),  # τ: exponential decay rate
)
```

The profile has two phases:

1. **Box phase** — efficacy stays constant at `initial_efficacy` for `box_duration` days after delivery.
2. **Decay phase** — efficacy decays exponentially: `E(t) = initial_efficacy × exp(−t / τ)`, where `t` is time since the box phase ended.

Efficacy is set to zero once it falls below 1% of `initial_efficacy` and is not updated further.

#### Common profiles

| Profile | `box_duration` | `decay_time_constant` | Effect |
|---|---|---|---|
| Immediate decay | `ss.days(0)` | `ss.days(730)` | Decays from delivery time with τ = 2 years |
| Box then decay | `ss.days(180)` | `ss.days(730)` | Flat for 6 months, then decays |
| Permanent box | `ss.days(inf)` | any | Efficacy never decays |
| Step function | `ss.days(180)` | `ss.days(0)` | Full efficacy for 6 months, then drops to zero |

#### Example: RTS,S-like vaccine profile

```python
ms.MalariaVaccine(
    name   = 'rtsS',
    waning = ms.WaningEffect(
        initial_efficacy    = 0.80,
        box_duration        = ss.days(180),
        decay_time_constant = ss.days(730),
    ),
)
```

At 6 months: 80%. At 12 months (6 months into decay): `0.80 × exp(−0.5 / 2) ≈ 70%`. At 24 months: `0.80 × exp(−1 / 2) ≈ 48%`. At approximately 6.6 years after the box phase ends, efficacy falls below 1% and is set to zero.

## 8. Sweeping Parameters

Since `build_campaign` is a plain Python function, sweeping intervention parameters requires no special machinery. Pass parameters directly into the builder and use `functools.partial` to bind them per sim:

```python
from functools import partial
import malariasim as ms
import starsim as ss

def build_campaign(sim, eir_only=False, efficacy=0.80, coverage=0.80):
    vaccine = ms.MalariaVaccine(
        name   = 'rtsS',
        waning = ms.WaningEffect(
            initial_efficacy    = efficacy,
            box_duration        = ss.days(180),
            decay_time_constant = ss.days(730),
        ),
    )
    campaign = ms.Campaign(sim)
    campaign.add(ms.InterventionDistributorScheduled(
        name='vacc_2005', products=vaccine, start=2005, coverage=coverage,
    ))
    return campaign

sims = [
    ms.Sim(
        human          = 'sis',
        mosquito       = 'agents',
        build_campaign = partial(build_campaign, efficacy=e, coverage=c),
        label          = f'efficacy={e} coverage={c}',
        start          = 2000,
        stop           = 2015,
    )
    for e in [0.60, 0.80]
    for c in [0.50, 0.80]
]
msim = ms.MalariaMultiSim(sims)
msim.run()
```

`partial(build_campaign, efficacy=0.80, coverage=0.50)` returns a callable that accepts one argument (`sim`) — exactly what `ms.Sim` expects.

#### Sweeping model configuration and interventions together

Model configuration and intervention parameters compose naturally in the same loop:

```python
sims = [
    ms.Sim(
        human          = human,
        mosquito       = 'agents',
        build_campaign = partial(build_campaign, efficacy=efficacy),
        label          = f'{human} efficacy={efficacy}',
        start          = 2000,
        stop           = 2015,
    )
    for human    in ['sis', 'seis', 'ode']
    for efficacy in [0.60, 0.80]
]
```

Because the same `build_campaign` function works across all human models, adding a new model to the sweep requires no changes to the campaign code.

## 9. Compatibility: ABM vs ODE Human Models

The same `build_campaign` function works unchanged with all human models — `'sis'`, `'seis'`, and `'ode'`. You do not need to check which model is running inside your campaign code. The framework translates campaign intent into the appropriate representation automatically:

| | ABM (`'sis'`, `'seis'`) | ODE (`'ode'`) |
|---|---|---|
| Coverage sampling | Bernoulli draw per agent | Continuous fraction per compartment |
| Age filtering | Exact agent age | Bin minimum age — whole bins selected |
| `Outbreak` | Infects individual susceptible agents | Moves `coverage × Sh` from susceptible to infectious |
| Waning | Per-agent efficacy arrays | Per-compartment efficacy arrays |

#### Second doses and re-vaccination

In the **ABM**, a second dose to the same agent overwrites the first — the new efficacy and waning profile replace the old ones from the new dose time, regardless of how much protection remained.

In the **ODE**, a second dose blends old and new. If a fraction `f` of a compartment is already vaccinated and a new round delivers coverage `c`, the resulting vaccinated fraction is `f + c × (1 − f)` — only the previously unvaccinated portion of the compartment is newly dosed. The effective dose time used to anchor the waning profile is a population-weighted mean of the old and new dose times. This preserves the contribution of the earlier cohort rather than discarding it.

#### The one practical constraint

When using an ODE human model, **age ranges in `HumanEligibility` must align to the bin edges defined in `age_bin_edges`**. This is enforced: `Campaign.add()` raises `ValueError` immediately if `min_age` or `max_age` does not land on a bin edge, pointing directly at the offending `add()` call. If you are running campaigns across both ABM and ODE models, define your age ranges to match your ODE bin edges — the ABM will handle them correctly regardless.

For example, with bins `[0, 5, 15, 30, 60]`:

```python
# Correct — aligns to bin edges
ms.HumanEligibility(min_age=ss.years(5), max_age=ss.years(15))

# Raises ValueError at Campaign.add() — 7 falls mid-bin
ms.HumanEligibility(min_age=ss.years(7), max_age=ss.years(15))
```

## 10. Assumptions and Limitations

#### One active instance per intervention type per person

A person (or ODE compartment) can hold at most one active dose of each intervention type at a time. A second vaccine dose overwrites the first; a person can hold a vaccine and a bed net simultaneously, but not two vaccines. See Section 9 for how ABM and ODE handle re-dosing differently.

#### ODE age targeting must align to bin edges

When using `human='ode'`, eligibility age ranges must align to the bin edges defined in `age_bin_edges`. This is enforced: `Campaign.add()` raises `ValueError` immediately if `min_age` or `max_age` does not land on a bin edge. See Section 9 for details.

#### `InterventionDistributorScheduled` fires exactly once

There is no built-in repeat or interval mechanism. Each delivery round requires its own distributor with an explicit `start` year. This is intentional — campaigns are explicit and auditable.

#### Planned but not yet implemented

The following are designed and documented but not yet available:

- `InterventionDistributorTriggered` — per-person event-driven delivery
- `InterventionDistributorReferenceTracking` — target-coverage maintenance
- `InterventionProgram` — named multi-distributor strategies (e.g. SMC)
- `Bednet`, `AntimalarialDrug`, `Larvicide`, `IndoorResidualSpraying`, `Ivermectin`, and other products listed in Section 1

#### ODE approximation for multiple campaigns

When multiple campaigns target the same ODE compartment at different times, the vaccinated fraction and dose time are tracked as population-weighted means. This loses the distinction between early-dosed and late-dosed subgroups, introducing a small error in the waning trajectory. The ODE is best used for fast parameter screening; the ABM provides more precise answers on selected parameter combinations.
