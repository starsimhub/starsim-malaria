""" FastBirths and FastDeaths from malariasim branch Interventions2-ck (6ca8d1d), malariasim/defaults.py, copied verbatim apart from imports """
import numpy as np, numba as nb, pandas as pd, starsim as ss
from common import compute_mortality_schedule, AGE_BIN_EDGES, AGE_BIN_FRACTIONS
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
