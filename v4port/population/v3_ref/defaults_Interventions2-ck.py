"""
defaults.py — canonical demographic and transmission parameters for malariasim.

Centralises values that would otherwise be duplicated across run scripts,
tests, and sweep scripts.  Import from here rather than copying constants.

Age structure
-------------
``AGE_BIN_EDGES`` and ``AGE_BIN_FRACTIONS`` define a Sub-Saharan Africa-style
age pyramid with 20 five-year bins from 0 to 100 years.

Transmission settings (3-node canonical grid)
----------------------------------------------
Named constants for the three nodes used in ``run_malariasim.py``:
  URBAN_*      — high transmission, mostly indoor biting
  PERI_URBAN_* — moderate wet-season transmission
  RURAL_*      — lower transmission with a sharp seasonal peak

Helper functions
----------------
``make_steady_state_demographics()``
    Returns a fresh ``[ss.Births, ss.Deaths]`` pair calibrated so that births
    exactly balance deaths at the initial age distribution.  Must be called
    fresh for each simulation (modules cannot be shared).

``make_age_data()``
    Returns a per-year age-distribution dataframe whose smooth density is
    consistent with the mortality rates passed to ``ss.Deaths``,
    so agent initialisation and mortality are in demographic equilibrium.
"""

from dataclasses import dataclass

import numpy as np
import numba as nb
import pandas as pd
import starsim as ss

from .demographics import compute_mortality_schedule


@nb.njit(cache=True)
def _death_draw_kernel(auids, age, p_by_age, u):
    """
    Single raw-array pass selecting agents who die this step: agent k dies iff
    ``u[k] < p_by_age[clip(floor(age), 0, n_ages-1)]``. Returns the raw indices
    (UIDs) of the dying agents. Folds ``ss.Deaths.make_p_death``'s age gather +
    lookup and the per-agent Bernoulli ``filter()`` into one loop, avoiding the
    ss.Arr ``.values`` gather and the ss.Distribution draw machinery. Valid under
    ``crn=False`` (the per-agent RNG slots carry no benefit then).
    """
    n_ages = p_by_age.shape[0]
    out = np.empty(auids.shape[0], dtype=np.int64)
    m = 0
    for k in range(auids.shape[0]):
        i = auids[k]
        a = int(age[i])
        if a < 0:
            a = 0
        elif a >= n_ages:
            a = n_ages - 1
        if u[k] < p_by_age[a]:
            out[m] = i
            m += 1
    return out[:m]


# ------------------------------------------------------------------
# Shared biological parameters
# ------------------------------------------------------------------

@dataclass
class CommonParams:
    """
    Canonical biological parameters shared across all MalariaHuman model variants.

    Each human disease module has a ``pars_from(cp)`` classmethod that translates
    these values into the model-specific parameter dict (e.g. converting
    ``infection_duration_mean_days`` into ``ss.lognorm_ex`` for ABM models or ``ss.days``
    for the ODE model).

    Parameters
    ----------
    prob_human_infected_on_bite : float
        Probability of human infection per infectious mosquito bite
        (Ross-Macdonald *b*). Default: 1.0.
    prob_mosquito_infected_on_bite : float
        Probability of mosquito infection per bite on an infectious human
        (Ross-Macdonald *c*). Default: 1.0.
    infection_duration_mean_days : float
        Arithmetic mean of the infectious period in days. Default: 200.
    infection_duration_std_days : float
        Arithmetic standard deviation of the infectious period in days.
        Used by ABM models to parameterize a log-normal duration distribution.
        Ignored by ODE models which use the mean only. Default: 200.
    """
    prob_human_infected_on_bite:       float = 0.4     # Ross-Macdonald *b*
    prob_mosquito_infected_on_bite:    float = 0.3     # Ross-Macdonald *c*
    infection_duration_mean_days:      float = 200.0   # arithmetic mean of the infectious period
    infection_duration_std_days:       float = 200.0   # arithmetic std dev (log-normal for ABM; ignored by ODE)
    bites_per_female_per_day:          float = 0.03    # mosquito biting rate (a)
    mosquito_life_expectancy_days:     float = 10.0    # mean adult female lifespan (1/μ)
    mosquito_eip_days:                 float = 12.0    # extrinsic incubation period (1/σ)


# ------------------------------------------------------------------
# Age structure
# ------------------------------------------------------------------

AGE_BIN_EDGES = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
]

AGE_BIN_FRACTIONS = [
    0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
    0.035, 0.03, 0.025, 0.02, 0.015, 0.01,
    0.006, 0.0036, 0.0022, 0.0013,
]

# ------------------------------------------------------------------
# Canonical 3-node transmission parameters
# ------------------------------------------------------------------

# Monthly EIR (bites / person / month) for each node
URBAN_EIR      = [1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1,  1]
PERI_URBAN_EIR = [2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2]
RURAL_EIR      = [5,  5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5]

# Monthly emergence (new adult females / human / day)
URBAN_EMERGENCE      = [0.008, 0.008, 0.015, 0.030, 0.045, 0.060,
                        0.060, 0.045, 0.030, 0.015, 0.008, 0.008]
PERI_URBAN_EMERGENCE = [0.005, 0.005, 0.010, 0.020, 0.030, 0.040,
                        0.040, 0.030, 0.020, 0.010, 0.005, 0.005]
RURAL_EMERGENCE      = [0.002, 0.002, 0.005, 0.010, 0.018, 0.025,
                        0.025, 0.018, 0.010, 0.005, 0.002, 0.002]

# Fraction of bites that occur indoors
URBAN_F_INDOOR      = 0.80
PERI_URBAN_F_INDOOR = 0.70
RURAL_F_INDOOR      = 0.55

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

class FastDeaths(ss.Deaths):
    """
    ``ss.Deaths`` variant that avoids the per-timestep pandas lookup.

    The base ``ss.Deaths.make_p_death`` rebuilds each agent's death probability
    every step by locating the nearest data year and running ``np.digitize`` of
    all ages against the age-rate table — an O(n_alive) cost that dominates the
    demographics runtime. Here the mortality schedule is static (one integer-age
    hazard per year), so the per-integer-age daily death probability is a fixed
    vector: it is computed once and looked up by ``floor(age)``. The returned
    probabilities are identical to the base class, so dynamics are unchanged.
    """

    def __init__(self, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu = np.asarray(mu, dtype=float)  # per-year hazard by integer age
        self._p_by_age = None                    # daily P(death) by integer age (lazy)
        self._rng = None                         # dedicated Generator (crn=False path)

    def _ensure_tables(self):
        if self._p_by_age is None:
            rate = self._mu * self.pars.rate_units * self.pars.rel_death
            self._p_by_age = np.asarray(ss.peryear(rate).to_prob(self.t.dt), dtype=float)
        if self._rng is None:
            # Dedicated deterministic stream (crn=False → per-agent RNG slots
            # give nothing; a single bulk uniform draw is far cheaper than the
            # ss.Distribution per-agent Bernoulli filter).
            seed = int(self.sim.pars.rand_seed or 0)
            self._rng = np.random.default_rng(seed + 20259)

    def make_p_death(self):
        # Retained for API compatibility / external callers; the hot path is the
        # numba kernel in step(). Integer-age lookup matches the base class exactly.
        self._ensure_tables()
        ages = np.asarray(self.sim.people.age.values)  # alive agents, auids order
        idx  = np.clip(ages.astype(np.int64), 0, self._p_by_age.size - 1)
        return self._p_by_age[idx]

    def step(self):
        """Select agents to die via one numba pass (no per-agent ss.dist Bernoulli)."""
        self._ensure_tables()
        auids = np.asarray(self.sim.people.auids)
        u = self._rng.random(auids.shape[0])
        death_uids = _death_draw_kernel(auids, self.sim.people.age.raw, self._p_by_age, u)
        self.sim.people.request_death(ss.uids(death_uids))
        self.n_deaths = len(death_uids)
        return self.n_deaths

    def finalize(self):
        # ss.Deaths.finalize computes a crude mortality rate from sim.results.n_alive,
        # which is absent when the sim runs with people_results=False (this model tracks
        # populations via its own analyzer). Do the standard finalize + cumulative-deaths
        # sum and skip the unused cmr diagnostic so people_results=False is supported.
        super(ss.Deaths, self).finalize()   # Demographics/Module.finalize (result scaling, ti fix)
        self.results.cumulative[:] = np.cumsum(self.results.new)
        return


class FastBirths(ss.Births):
    """
    ``ss.Births`` variant that draws the birth *count* directly instead of
    running a per-agent Bernoulli filter over the whole population each step.

    The base ``ss.Births.get_births`` samples one Bernoulli per alive agent (an
    O(n_alive) draw) purely to preserve common random numbers — its own source
    even notes the faster ``np.random.binomial`` alternative is "not CRN safe".
    With a constant per-capita birth rate the number of births per step is
    ``Poisson(n_alive * p)``, which is statistically identical to the sum of the
    per-agent Bernoullis but O(1). This is only used with CRN disabled (the
    per-agent RNG slots carry no benefit then), so nothing is lost.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_new_dist = ss.poisson(lam=1.0)  # count draw; lam set per step
        self._scaled_p   = None                 # daily P(birth) per capita (lazy, constant)

    def get_births(self):
        # Fall back to the base per-agent path for data-driven (time-varying) rates.
        if not isinstance(self.pars.birth_rate, ss.Rate):
            return super().get_births()
        if self._scaled_p is None:
            br   = self.pars.birth_rate
            rate = ss.prob(br.value * self.pars.rate_units * self.pars.rel_birth,
                           br.unit).to_prob(ss.years(1))
            self._scaled_p = float(np.clip(ss.prob.array_to_prob(np.array([rate]), self.t.dt)[0],
                                           0.0, 1.0))
        auids = self.sim.people.auids
        self._n_new_dist.set(lam=len(auids) * self._scaled_p)
        n_new = min(int(self._n_new_dist.rvs(1)[0]), len(auids))
        # `parent` is unused in malariasim (no maternal network), so any n_new
        # existing agents serve as nominal parents.
        return auids[:n_new]


def make_steady_state_demographics(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS):
    """
    Return a fresh ``[FastBirths, FastDeaths]`` demographics list calibrated for
    a stable, representative population.

    The birth rate is derived from the age-specific mortality schedule so
    that births exactly balance deaths at the initial age distribution.
    This is the standard approach for malaria modelling, where simulations
    typically use a representative constant population rather than actual
    census birth/death data.

    A new list must be created for each simulation — modules cannot be
    shared across ``ss.Sim`` instances.

    Returns
    -------
    list
        ``[FastBirths, FastDeaths]`` ready to pass as ``demographics=`` to
        ``ms.Sim``.  ``FastDeaths`` is a drop-in ``ss.Deaths`` that precomputes
        the per-age death probability (identical dynamics, much faster).
    """
    mu, _, birth_rate, max_age = compute_mortality_schedule(edges, fractions)
    births = FastBirths(birth_rate=ss.peryear(birth_rate * 1000))
    df = pd.DataFrame({
        'Time':        2000,
        'AgeGrpStart': np.arange(max_age),
        'mx':          mu,
    })
    metadata = {'data_cols': dict(year='Time', age='AgeGrpStart', value='mx')}
    return [births, FastDeaths(mu, death_rate=df, rate_units=1, metadata=metadata)]


def make_age_data(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS):
    """
    Return a per-year age-distribution dataframe for initialising agent ages.

    The smooth density n(x) is derived from the same log-linear interpolation
    used to compute per-year mortality rates, so agent initialisation and
    mortality are at the same demographic equilibrium.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age`` (int, 0–99) and ``value`` (float, relative weight).
        Suitable for ``ss.People(age_data=...)``.
    """
    _, n, _, max_age = compute_mortality_schedule(edges, fractions)
    return pd.DataFrame({'age': np.arange(max_age, dtype=int), 'value': n})


def make_mortality_dataframe():
    """
    Return a ``(dataframe, birth_rate)`` pair suitable for ``ss.Deaths``.

    The dataframe has columns ``Time``, ``AgeGrpStart``, ``mx`` — one row
    per integer age 0 to max_age-1.  Rates are per-year (use ``rate_units=1``).

    Returns
    -------
    df : pandas.DataFrame
        Age-specific mortality rates for ``ss.Deaths(death_rate=df, rate_units=1)``.
    birth_rate : float
        Calibrated births per person per year.
    """
    mu, _, birth_rate, max_age = compute_mortality_schedule(
        AGE_BIN_EDGES, AGE_BIN_FRACTIONS
    )
    df = pd.DataFrame({
        'Time':        2000,
        'AgeGrpStart': np.arange(max_age),
        'mx':          mu,
    })
    return df, birth_rate
