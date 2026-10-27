# Malaria Interventions — User Guide

This guide explains how to add interventions — vaccines, outbreak seeding, bed nets, drugs — to a malariasim simulation. It is written for malaria researchers; no prior knowledge of Starsim is required.

## 1. Overview

#### Supported interventions

| Intervention | Target | Mechanism |
|---|---|---|
| Vaccine (e.g. RTS,S, R21) | Person | Reduces probability of infection per bite |
| Outbreak (infection seeding) | Person | Moves susceptible individuals to infectious immediately |
| Insecticide-treated bed net (ITN) | Person | Reduces biting rate, kills mosquitoes |
| Antimalarial drug | Person | Clears infection |
| Diagnostic test (RDT, microscopy) | Person | Identifies infected individuals, triggers treatment |
| Ivermectin | Person | Makes blood meal toxic to feeding mosquitoes |
| Larvicide | Node | Reduces mosquito emergence rate |
| Indoor residual spraying (IRS) | Node | Increases indoor mosquito death rate |
| Attractive targeted sugar bait (ATSB) | Node | Kills mosquitoes at sugar sources |
| Mosquito release (SIT, Wolbachia) | Node | Modifies mosquito population |

Vaccines, outbreaks, bed nets, and drugs are implemented. A diagnostic test is Starsim's `ss.Diagnostic`, delivered by `ss.screen()`, and can trigger treatment. The remaining interventions are planned.

#### One campaign, all models

A core design goal of malariasim is that a single campaign definition works unchanged across every combination of human and mosquito model. The malaria model is one declared Starsim disease (`ms.Malaria`), and the interventions are Starsim's own deliveries (`ss.vaccinate()`, `ss.treat()`, `ss.Delivery()`), so the same campaign runs with SIS or SEIS humans, with EIR-only, ODE, or agent-based mosquitoes, and with people as agents or as an ODE (`solver='ode'`). The one exception is protection that wanes by a function of time other than a box (`ss.box()`, full protection for a time, then an exponential decay), which only agents keep (see Section 9).

#### How it works

You give `ms.Sim` the interventions: a list of deliveries, or a function of the disease that returns them (needed when a delivery names the disease's states, as an outbreak does). Each delivery says what is delivered, to whom, when, and to what coverage. A campaign can include any number of deliveries — the same vaccine given in multiple rounds, different vaccines, seeding followed by vaccination — with no limit.

## 2. Quick Start

The minimal pattern is: create a product, deliver it with targeting and coverage, and pass the delivery to `ms.Sim`.

```python
import starsim as ss
import malariasim as ms

vaccine = ss.Vaccine(
    name     = 'rtsS',
    efficacy = 0.80,
    waning   = ss.box(dur=ss.days(180), decay=ss.days(730)),
)
nodes = ss.geo.geography(names=['Urban', 'Peri-urban', 'Rural'], pop=[6000, 3000, 1000])
vacc_2005_u5 = ss.vaccinate(
    vaccine,
    at       = '2005-01-01',
    coverage = 0.80,
    where    = (ss.ppl.age >= 0.5) & (ss.ppl.age < 5) & (ss.ppl.patch != 'Peri-urban'),
)

sim = ms.Sim(
    human         = 'sis',
    mosquito      = 'agents',
    nodes         = nodes,
    interventions = vacc_2005_u5,
    start         = 2000,
    stop          = 2015,
)
sim.run()
```

This vaccinates 80% of people aged 6 months to 5 years in the Urban and Rural nodes on 1 January 2005, with an RTS,S-like profile: full efficacy for six months, then an exponential decay with a mean of two years.

## 3. Building a Campaign

A campaign is a list of deliveries, given to `ms.Sim(interventions=...)`.

#### Interventions that name the disease's states

An outbreak moves people between the disease's states, so it needs the disease. Give `ms.Sim` a function of the disease instead of a list; it is called with the `ms.Malaria` the sim is made with:

```python
def campaign(mal):
    return [
        ss.Delivery(move=mal.S >> mal.I, at='2000-02-01', coverage=0.15, name='seed'),
        ss.vaccinate(vaccine, at='2005-01-01', coverage=0.8, where=ss.ppl.age < 5),
    ]

sim = ms.Sim(mosquito='ode', interventions=campaign)
```

The function can also tell the mosquito model apart: `mal.mosquito` is `'ode'`, `'eir_only'`, or `'agents'`.

Each delivery has a `name`, used for its results (e.g. `sim.results.seed.delivered`); if none is given, it is named after the variable it is assigned to or its kind. Names must be unique, including between a delivery and its product.

#### Multiple rounds of the same vaccine

A delivery can have several rounds (`at=[...]`), or one product can be given by several deliveries. A second dose to the same person restarts their protection: the waning starts again from the new dose:

```python
rounds = ss.vaccinate(vaccine, at=['2005-01-01', '2007-01-01'], coverage=0.8, where=(ss.ppl.age >= 0.5) & (ss.ppl.age < 5), name='rounds')
```

#### Seeding infections with an outbreak

An outbreak immediately moves a fraction of susceptible individuals to infectious. Its most common use is seeding an initial infection load at the start of a simulation, but it can also be used to model importation events at any point during a run.

Whether seeding is needed at the start depends on the mosquito model. When `mosquito='eir_only'`, the externally-specified EIR drives infections from the first timestep — the population will naturally accumulate infections. When mosquitoes are modelled explicitly (`mosquito='agents'` or `mosquito='ode'`), the mosquito population starts uninfected and the feedback loop between humans and mosquitoes takes time to build up; seeding with an outbreak (or `init_prev=`) establishes a realistic starting prevalence immediately. One function handles both cases:

```python
def campaign(mal):
    out = []
    if mal.mosquito != 'eir_only':
        out.append(ss.Delivery(move=mal.S >> mal.I, at='2000-01-01', coverage=1.0, name='seed'))
    # add vaccines, bednets, etc.
    return out
```

#### Validation

Starsim checks each delivery when the sim is initialized: a time outside the simulation raises an error naming the delivery, and so does a name used twice.

## 4. Targeting: Who Gets the Intervention

Targeting is a condition on people, given as `where=`. If omitted, all alive individuals in all nodes are eligible. Eligibility is separate from coverage — eligibility defines the pool, coverage determines what fraction of that pool is actually dosed.

```python
where = (ss.ppl.age >= 0.5) & (ss.ppl.age < 5)        # an age range (in years)
where = ss.ppl.patch == 'Rural'                        # one node, by name
where = (ss.ppl.patch != 'Urban') & (ss.ppl.age < 15)  # combined
```

Conditions combine with `&` (and), `|` (or), and `~` (not). Age bounds follow the half-open convention when written as above: `0.5 <= age < 5` selects people aged 6 months up to but not including their 5th birthday.

#### Age targeting with ODE models

For agents, each person has an exact age. For an ODE (`solver='ode'`), people are in age groups (`ms.Sim(age_bin_edges=...)`, e.g. [0–5), [5–10), ...), and a condition on age must not split a group: give age bounds on the edges of the groups.

#### Node targeting

Nodes are the places of an `ss.geo.geography`, named when it is made (`ss.geo.geography(names=['Urban', 'Rural'], pop=[6000, 4000])`), and targeted by name: `where=ss.ppl.patch == 'Rural'`.

## 5. Scheduling and Coverage

A delivery's timing is `at=` (one date or a list, for rounds), or `start=`, `stop=`, and `interval=` for regular rounds:

```python
ss.vaccinate(
    vaccine,
    at       = '2005-01-01',               # or a list of dates, one per round
    coverage = 0.80,                       # fraction of eligible targets to dose
    where    = ss.ppl.age < 5,             # optional; default = all alive
    name     = 'vacc_2005_u5',
)
```

A date can be a string (`'2005-07-01'`) or an `ss.date`. The delivery happens on the step that starts on that date.

`coverage` is the fraction of the eligible pool that receives the intervention. For agents, each eligible person is selected independently with probability `coverage`. For an ODE, `coverage` is applied as a continuous fraction of each compartment — no randomness is involved.

Each delivery keeps results: the number `delivered`, `eligible`, and the `coverage` reached, at each step (e.g. `sim.results.vacc_2005_u5.delivered`).

## 6. Products

A product defines what an intervention does biologically. The same product instance can be reused across multiple deliveries — the biological definition is written once regardless of how many delivery rounds use it.

#### Vaccine

A vaccine is Starsim's `ss.Vaccine`. Against malaria, it reduces an individual's probability of acquiring infection per infectious bite, indoors and outdoors (its `efficacy`, against acquisition). Its efficacy wanes over time as its `waning=` says.

```python
ss.Vaccine(
    name     = 'rtsS',
    efficacy = 0.80,                                          # efficacy at time of dosing, in [0, 1]
    waning   = ss.box(dur=ss.days(180), decay=ss.days(730)),  # full for 180 days, then an exponential decay with a mean of 730 days
)
```

A second dose to the same person restarts the waning from the new dose. See Section 7 for details on waning profiles.

#### Outbreak

An outbreak is `ss.Delivery(move=mal.S >> mal.I, ...)`: it moves the susceptible people it reaches to infectious (see Section 3). Coverage controls what fraction of the eligible susceptible pool is infected: `coverage=1.0` with no `where=` infects the entire susceptible population.

#### `Bednet`

`ms.Bednet` is an insecticide-treated net with four effects, each a probability that can wane: repelling, blocking, killing, and usage (which scales the other three). Its holder is protected from indoor bites by `1 − (1 − usage × repelling)(1 − usage × blocking)`. The mosquitoes feel the nets too: in each node, the mean of the same over people blocks indoor feeding on them, and `usage × (1 − repelling) × blocking × killing` is the chance per indoor bite that a mosquito is killed. A repelled mosquito is not killed.

```python
ms.Bednet(repelling=0.5, blocking=0.9, killing=0.6, usage=0.8, waning=ss.box(dur=ss.years(3))) # Every effect lasts three years
```

`waning=` is one waning for every effect, or a dict with one for each (an effect left out does not wane):

```python
ms.Bednet(
    repelling = 0.5,
    blocking  = 0.9,
    killing   = 0.6,
    usage     = 0.8,
    waning    = dict(
        repelling = ss.box(dur=ss.days(0),   decay=ss.days(730)),
        blocking  = ss.box(dur=ss.days(365), decay=ss.days(365)),
        killing   = ss.box(dur=ss.days(180), decay=ss.days(365)),
        usage     = ss.box(dur=ss.days(365), decay=ss.days(365)),
    ),
)
```

It is delivered by `ss.vaccinate()`, like a vaccine. The mosquitoes read the nets' effects from their coverage (`itn.coverage`, the share of the people in the same node who hold one, each counted by the protection they have left), which `ms.Sim` adds to the mosquitoes' rates for every net delivered.

#### Antimalarial drug

A drug that clears infection is Starsim's `ss.Treatment`, delivered by `ss.treat()`. `cure=dict(I=1.0)` clears the infectious (in the SEIS model, those still incubating are not cleared):

```python
mda = ss.treat(ss.Treatment(cure=dict(I=1.0)), at='2007-01-01', coverage=0.5, where=ss.ppl.age < 20)
```

Clearance is immediate. A treatment has the outcomes `cured`, `failed`, and `unneeded` (not infected), which are results of its delivery.

#### Planned products

- **`Larvicide`** — reduces mosquito emergence rate at a node
- **`IndoorResidualSpraying`** — increases indoor mosquito death rate at a node

## 7. Waning Efficacy

Waning describes how a product's biological effect changes over time after delivery. A product's `waning=` is one of:

- `None` (the default): permanent
- a duration, e.g. `ss.days(730)`: each person loses all their protection at a random time, with that mean
- `ss.box(dur=..., decay=...)`: full efficacy for `dur`, then (with `decay=`) an exponential decay with that mean, or (without it) none
- any function of the time since delivery `t` (a duration, so `t.years` and `t < ss.days(180)` read the same in any sim), giving the share of the protection left, e.g. `lambda t: np.clip(1 - t.years/5, 0, 1)`

The box has two phases:

1. **Box phase**: efficacy stays at its value at delivery (the product's `efficacy`) for `dur` after delivery.
2. **Decay phase**: efficacy decays exponentially: `E(t) = efficacy × exp(−t / decay)`, where `t` is time since the box phase ended.

#### Common profiles

| Profile | `waning=` | Effect |
|---|---|---|
| Immediate decay | `ss.box(dur=ss.days(0), decay=ss.days(730))` | Decays from delivery time, with a mean of 2 years (as agents) |
| Box then decay | `ss.box(dur=ss.days(180), decay=ss.days(730))` | Flat for 6 months, then decays |
| Step function | `ss.box(dur=ss.days(180))` | Full efficacy for 6 months, then drops to zero |
| Permanent | `None` (the default) | Efficacy never decays |

A box (`ss.box(...)`) is also the share of the protection left: `ss.box(dur=ss.days(180), decay=ss.days(730))(ss.days(365))` is `exp(−185/730)`.

#### Example: RTS,S-like vaccine profile

```python
ss.Vaccine(
    name     = 'rtsS',
    efficacy = 0.80,
    waning   = ss.box(dur=ss.days(180), decay=ss.days(730)),
)
```

At 6 months: 80%. At 12 months (6 months into decay): `0.80 × exp(−0.5 / 2) ≈ 62%`. At 24 months: `0.80 × exp(−1.5 / 2) ≈ 38%`.

## 8. Sweeping Parameters

Since the interventions can be made by a plain Python function, sweeping intervention parameters requires no special machinery. Pass parameters into the function and use `functools.partial` to bind them per sim:

```python
from functools import partial
import malariasim as ms
import starsim as ss

def campaign(mal, efficacy=0.80, coverage=0.80):
    vaccine = ss.Vaccine(
        name     = 'rtsS',
        efficacy = efficacy,
        waning   = ss.box(dur=ss.days(180), decay=ss.days(730)),
    )
    return [
        ss.Delivery(move=mal.S >> mal.I, at='2000-02-01', coverage=0.15, name='seed'),
        ss.vaccinate(vaccine, at='2005-01-01', coverage=coverage, name='vacc_2005'),
    ]

sims = [
    ms.Sim(
        human         = 'sis',
        mosquito      = 'agents',
        interventions = partial(campaign, efficacy=e, coverage=c),
        label         = f'efficacy={e} coverage={c}',
        start         = 2000,
        stop          = 2015,
    )
    for e in [0.60, 0.80]
    for c in [0.50, 0.80]
]
msim = ms.MalariaMultiSim(sims)
msim.run()
```

`partial(campaign, efficacy=0.80, coverage=0.50)` returns a callable that accepts one argument (the disease) — exactly what `ms.Sim` expects.

#### Sweeping model configuration and interventions together

Model configuration and intervention parameters compose naturally in the same loop:

```python
sims = [
    ms.Sim(
        human         = human,
        mosquito      = 'agents',
        interventions = partial(campaign, efficacy=efficacy),
        label         = f'{human} efficacy={efficacy}',
        start         = 2000,
        stop          = 2015,
    )
    for human    in ['sis', 'seis']
    for efficacy in [0.60, 0.80]
]
```

Because the same campaign function works across all human models, adding a new model to the sweep requires no changes to the campaign code.

## 9. Compatibility: Agents vs ODE

The same campaign works unchanged with people as agents (`solver='abm'`, the default) and as an ODE (`solver='ode'`), with one exception: protection that wanes by a function of time other than a box.

| | Agents | ODE (`solver='ode'`) |
|---|---|---|
| Coverage sampling | Bernoulli draw per person | Continuous fraction per compartment |
| Age filtering | Exact age | Age groups — bounds on group edges |
| Outbreak | Infects individual susceptible people | Moves `coverage × S` from susceptible to infectious |
| Drug | Cures individual infectious people | Moves `coverage × I` from infectious to susceptible |
| Waning by `ss.box()` | Per-person efficacy from the time since their dose | Levels of holding the product: the box as 10 stages (so its end is spread over about `dur/√10`), then a decaying level |
| Waning by another function of time | Per-person efficacy from the time since their dose | Not available: refused before the run |

#### Second doses and re-vaccination

A second dose to the same person restarts their protection from the new dose time, regardless of how much protection remained.

#### The one practical constraint

A vaccine that wanes by a function other than a box, and a bed net whose effects wane differently (or by a box with a decay, since its protection then is not in proportion to the box), runs as agents only: with `solver='ode'`, Starsim refuses to run, saying why. Vaccines that are permanent or wane by a box, bed nets that are permanent or whose effects all wane by one box without a decay, outbreaks, and drugs run under both.

## 10. Assumptions and Limitations

#### One active instance per intervention type per person

A person holds at most one active dose of each product. A second vaccine dose restarts the first; a person can hold a vaccine and a bed net simultaneously.

#### Waning by a function of time other than a box runs as agents only

See Section 9.

#### Planned but not yet implemented

The following are designed and documented but not yet available:

- `Larvicide`, `IndoorResidualSpraying`, `Ivermectin`, and other node-level products listed in Section 1
