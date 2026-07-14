"""
MosquitoesODE: SEI compartmental (ODE) mosquito module for malaria.

Drop-in connector replacement for MosquitoesAdultFemaleAgentsOnly that
integrates the same SEI mosquito dynamics using Euler-stepped ODEs instead
of individual mosquito agents.

Use this module when:
  - Speed is a priority — no per-agent stochasticity, no large arrays
  - You want a deterministic baseline to compare against the ABM
  - Population sizes are large enough that stochastic effects are small

Trade-offs vs MosquitoesAdultFemaleAgentsOnly
----------------------------------------------
  + Much faster — O(n_nodes) per step instead of O(n_mosquitoes)
  + Fully deterministic given the same parameters and initial conditions
  - No individual-level heterogeneity in mosquito lifespan or EIP
  - Stochastic fade-out near zero is not captured (Im never reaches 0)

Parameter API
-------------
Parameter names are identical to MosquitoesAdultFemaleAgentsOnly so that
the two modules are interchangeable in ms.Sim(mosquito=...) and in code
that passes mosquito_pars dicts.

NodeState interface
-------------------
Reads  : p_inf_indoor, p_inf_outdoor, n_humans  (written by human module)
Writes : eir_indoor, eir_outdoor                (read by human next step)

Force of infection on mosquitoes (per susceptible per day):

    lambda_m = a * (f * p_inf_indoor + (1-f) * p_inf_outdoor)

where a = bites_per_female_per_day.  This matches the per-bite exposure
probability used by the ABM: each mosquito bites either indoors (prob f)
or outdoors (prob 1-f), and p_inf is the transmission probability per bite.

Per-timestep EIR written to NodeState:

    eir_dt = a * Im[nid] * dt_days / n_humans[nid]
"""

import numpy as np
import starsim as ss

from .utils import interpolate_monthly

# Same default seasonal emergence profile as MosquitoesAdultFemaleAgentsOnly.
_DEFAULT_EMERGENCE = np.array(
    [0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005],
    dtype=float,
)


class MosquitoesODE(ss.Module):
    """
    SEI ODE (compartmental) adult female mosquito module.

    Deterministic drop-in replacement for MosquitoesAdultFemaleAgentsOnly.
    Maintains per-node Sm, Em, Im compartment counts as floats and advances
    them with an Euler step each timestep.

    Parameters
    ----------
    emergence : array-like of length 12
        Monthly adult female emergence rate in new mosquitoes per human per
        day. One value per calendar month (index 0 = January). Values between
        months are linearly interpolated; December wraps to January.
        Default: seasonal profile (same as MosquitoesAdultFemaleAgentsOnly).

    bites_per_female_per_day : float
        Average bites per adult female per day (Ross-MacDonald *a*).
        Default: 0.3.

    life_expectancy : ss.days
        Mean adult female lifespan. Default: ``ss.days(10.0)``.

    eip_days : ss.days
        Extrinsic incubation period (time from exposure to infectious).
        Default: ``ss.days(12.0)``.

    f_indoor : float
        Fraction of bites occurring indoors. Must be in [0, 1]. Default: 0.7.

    Examples
    --------
    Drop-in replacement via ms.Sim::

        import malariasim as ms
        sim = ms.Sim(mosquito='ode')
        sim.run()

    Direct construction with custom parameters::

        from malariasim import MosquitoesODE
        mos = MosquitoesODE(
            emergence=[0.1, 0.1, 0.2, 0.5, 1.5, 3.0, 3.0, 1.5, 0.5, 0.2, 0.1, 0.1],
            life_expectancy=ss.days(14.0),
            eip_days=ss.days(10.0),
        )
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            emergence                = _DEFAULT_EMERGENCE.copy(),
            bites_per_female_per_day = 0.3,
            life_expectancy          = ss.days(10.0),
            eip_days                 = ss.days(12.0),
            f_indoor                 = 0.7,
        )
        self.update_pars(pars, **kwargs)
        self._validate()

        # Per-node SEI compartments — allocated in init_post.
        self._Sm = None   # ndarray(n_nodes): susceptible mosquitoes
        self._Em = None   # ndarray(n_nodes): exposed mosquitoes
        self._Im = None   # ndarray(n_nodes): infectious mosquitoes
        self._ns_cache = None
        return

    # ------------------------------------------------------------------
    # Starsim module lifecycle
    # ------------------------------------------------------------------

    def init_post(self):
        """Allocate per-node SEI arrays and seed at disease-free equilibrium."""
        super().init_post()
        ns      = self._get_node_state()
        grid    = ns.grid
        p       = self.pars
        t_frac  = self.t.year % 1

        mu_m           = 1.0 / p.life_expectancy.days
        n_nodes        = grid.n_nodes
        n_humans_total = int(self.sim.people.alive.sum())

        self._Sm = np.zeros(n_nodes)
        self._Em = np.zeros(n_nodes)
        self._Im = np.zeros(n_nodes)

        # Seed at disease-free equilibrium: Sm = emergence_rate * n_h / mu_m
        for nid in range(n_nodes):
            n_h           = max(int(round(n_humans_total * grid.pop_fractions[nid])), 0)
            em_r          = self._emergence_rate_for_node(grid, nid, t_frac)
            self._Sm[nid] = em_r * n_h / mu_m
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_susceptible',      dtype=float, label='Mosquitoes — susceptible (ODE)'),
            ss.Result('n_exposed',          dtype=float, label='Mosquitoes — exposed (ODE)'),
            ss.Result('n_infectious',       dtype=float, label='Mosquitoes — infectious (ODE)'),
            ss.Result('n_total',            dtype=float, label='Mosquitoes — total (ODE)'),
            ss.Result('eir_annual',         dtype=float, label='EIR — total (bites/person/year)'),
            ss.Result('eir_indoor_annual',  dtype=float, label='EIR — indoor (bites/person/year)'),
            ss.Result('eir_outdoor_annual', dtype=float, label='EIR — outdoor (bites/person/year)'),
        )
        return

    def step(self):
        """
        Advance the mosquito compartments by one timestep (Euler integration).

        Order of operations:
          1. For each node: compute births and lambda_m, Euler-step Sm/Em/Im.
          2. Record aggregate results.
          3. Write per-node EIR to NodeState for the human module.
        """
        ns      = self._get_node_state()
        grid    = ns.grid
        p       = self.pars
        dt_days = self.dt.days
        t_frac  = self.t.year % 1

        a       = p.bites_per_female_per_day
        mu_m    = 1.0 / p.life_expectancy.days
        sigma_m = 1.0 / p.eip_days.days

        n_humans = np.asarray(ns.n_humans)

        for nid in range(grid.n_nodes):
            n_h = n_humans[nid]
            if n_h == 0:
                continue

            em_r   = self._emergence_rate_for_node(grid, nid, t_frac)
            births = em_r * n_h  # new susceptible mosquitoes per day

            # H→M force of infection: each mosquito bites indoors (prob f)
            # or outdoors (prob 1-f); p_inf is transmission prob per bite.
            f        = (ns.grid.f_indoor[nid]
                        if ns.grid.f_indoor is not None
                        else p.f_indoor)
            itn_br       = ns.itn_biting_reduction[nid]
            mu_itn       = a * f * ns.itn_kill_prob[nid]   # only indoor-feeding mosquitoes encounter the net
            a_eff_indoor = a * f * (1.0 - itn_br)
            a_outdoor    = a * (1.0 - f)
            lambda_m     = a_eff_indoor * ns.p_inf_indoor[nid] + a_outdoor * ns.p_inf_outdoor[nid]

            Sm = self._Sm[nid]
            Em = self._Em[nid]
            Im = self._Im[nid]

            mu_tot = mu_m + mu_itn
            dSm = births        - lambda_m * Sm - mu_tot * Sm
            dEm = lambda_m * Sm - sigma_m * Em  - mu_tot * Em
            dIm = sigma_m * Em  - mu_tot * Im

            self._Sm[nid] = max(Sm + dSm * dt_days, 0.0)
            self._Em[nid] = max(Em + dEm * dt_days, 0.0)
            self._Im[nid] = max(Im + dIm * dt_days, 0.0)

        # Record aggregate results.
        # Annual EIR = per-step EIR / dt_years, where per-step EIR = a*Im*dt_days/n_h.
        # Equivalently: a * Im * dt_days / (n_h * dt_years) = a * Im * 365.25 / n_h.
        Im_total  = self._Im.sum()
        n_h_total = n_humans.sum()
        eir_per_step = a * Im_total * dt_days / n_h_total if n_h_total > 0 else 0.0
        annual_eir   = eir_per_step / self.dt.years       if n_h_total > 0 else 0.0

        self.results.n_susceptible[self.ti]      = self._Sm.sum()
        self.results.n_exposed[self.ti]          = self._Em.sum()
        self.results.n_infectious[self.ti]       = Im_total
        self.results.n_total[self.ti]            = (self._Sm + self._Em + self._Im).sum()
        self.results.eir_annual[self.ti]         = annual_eir
        self.results.eir_indoor_annual[self.ti]  = annual_eir * p.f_indoor
        self.results.eir_outdoor_annual[self.ti] = annual_eir * (1.0 - p.f_indoor)

        self._write_node_eir(ns)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_node_eir(self, ns):
        """Write per-node per-timestep EIR to NodeState."""
        p      = self.pars
        a      = p.bites_per_female_per_day
        dt_days = self.dt.days
        n_h    = ns.n_humans
        _f    = ns.grid.f_indoor if ns.grid.f_indoor is not None else p.f_indoor
        n_nodes = ns.grid.n_nodes
        f_arr = np.full(n_nodes, _f) if np.isscalar(_f) else np.asarray(_f, dtype=float)
        live           = n_h > 0
        itn_br         = np.asarray(ns.itn_biting_reduction)
        eir_indoor_dt  = np.zeros(ns.grid.n_nodes)
        eir_outdoor_dt = np.zeros(ns.grid.n_nodes)
        eir_indoor_dt[live]  = a * f_arr[live] * (1.0 - itn_br[live]) * self._Im[live] * dt_days / n_h[live]
        eir_outdoor_dt[live] = a * (1.0 - f_arr[live])                 * self._Im[live] * dt_days / n_h[live]
        ns.eir_indoor[:]  = eir_indoor_dt
        ns.eir_outdoor[:] = eir_outdoor_dt
        return

    def _emergence_rate_for_node(self, grid, nid, t):
        """Return the emergence rate (new females per human per day) for node nid."""
        if grid.emergence is not None:
            profile = grid.emergence[nid]
        else:
            profile = self.pars.emergence
        return interpolate_monthly(profile, t)

    def _get_node_state(self):
        """Return NodeState, raising RuntimeError if not found."""
        if self._ns_cache is None:
            self._ns_cache = next(
                (m for m in self.sim.modules if getattr(m, '_is_node_state', False)),
                None,
            )
            if self._ns_cache is None:
                raise RuntimeError(
                    f'{type(self).__name__}._get_node_state(): no NodeState '
                    'module found. Use ms.Sim() or add NodeState manually.'
                )
        return self._ns_cache

    def _validate(self):
        """Raise informative errors for malformed parameters."""
        em = np.asarray(self.pars.emergence, dtype=float)
        if em.shape != (12,):
            raise ValueError(
                f'emergence must be a 12-element array (one per calendar month), '
                f'got shape {em.shape}'
            )
        if np.any(em < 0):
            raise ValueError('All emergence values must be non-negative')
        self.pars.emergence = em

        if self.pars.life_expectancy <= 0:
            raise ValueError(f'life_expectancy must be positive, got {self.pars.life_expectancy}')
        if self.pars.eip_days <= 0:
            raise ValueError(f'eip_days must be positive, got {self.pars.eip_days}')
        if self.pars.bites_per_female_per_day < 0:
            raise ValueError(f'bites_per_female_per_day must be non-negative, got {self.pars.bites_per_female_per_day}')
        f = self.pars.f_indoor
        if not (0.0 <= f <= 1.0):
            raise ValueError(f'f_indoor must be in [0, 1], got {f}')
        return
