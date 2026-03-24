"""
MosquitoesAdultFemaleAgentsOnly: SEI agent-based adult female mosquito module.

Female Anopheles mosquitoes are modelled as individual agents with three
states: Susceptible, Exposed, and Infectious. The mosquito population is
managed as numpy arrays within this module — mosquitoes are kept separate
from the human People object to avoid mixing species.

Decoupling from the human model
--------------------------------
This module is fully decoupled from the human disease module. Both sides
communicate through a shared ``NodeState`` object discovered at runtime via
the ``_is_node_state`` duck-type marker. The human module writes
``p_inf_indoor`` / ``p_inf_outdoor`` / ``n_humans`` to NodeState each
timestep; this module reads those fields and writes back ``eir_indoor`` /
``eir_outdoor``.

Population dynamics
-------------------
The mosquito population is dynamic and seasonal, driven by a monthly emergence
rate (new adult females per human per day). Values between months are linearly
interpolated; December wraps to January.

Each timestep, new susceptible mosquitoes emerge at:

    n_new ~ Poisson(emergence_rate * n_humans * dt_days)

At equilibrium the expected mosquito-to-human ratio is:

    N_mosquitoes / N_humans ≈ emergence_rate / (1 / life_expectancy)

When a mosquito dies (pre-scheduled from an exponential lifespan at birth),
its array slot is freed and reused by new arrivals. Arrays are pre-allocated
to 3× the maximum equilibrium population to avoid runtime resizing.

Human → mosquito transmission
------------------------------
Each susceptible mosquito is assigned indoors or outdoors (Bernoulli with
p = f_indoor), bites with probability bites_per_female_per_day * dt_days, and
if it bites becomes exposed with probability read from NodeState:

    P(exposed | indoors)  = NodeState.p_inf_indoor   = I_h * c
    P(exposed | outdoors) = NodeState.p_inf_outdoor  = I_h * c

Both are equal in the base model and will differ once interventions are added.

Exposed → infectious transition
---------------------------------
Scheduled at ti_exposed + eip_days / dt_days (EIP in timestep units).

Mosquito → human transmission (EIR)
-------------------------------------
Per-node EIR values are written to NodeState (``eir_indoor``,
``eir_outdoor``) at the end of each step for the human module to read.
"""

import numpy as np
import starsim as ss

from .utils import interpolate_monthly

DAYS_PER_YEAR = 365.25

# Default monthly emergence profile: seasonal with a wet-season peak.
# Values are new adult female mosquitoes per human per day.
# With default life_expectancy=10 days (mu=0.1/day):
#   peak (1.0)  → ~10 mosquitoes per human
#   trough (0.2) → ~2 mosquitoes per human
_DEFAULT_EMERGENCE = np.array(
    [0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005],
    dtype=float,
)


class MosquitoesAdultFemaleAgentsOnly(ss.Module):
    """
    SEI agent-based adult female mosquito module for malaria.

    The mosquito population is dynamic and seasonal, driven by a monthly
    emergence rate. Mosquitoes are individual agents with Susceptible,
    Exposed, and Infectious states, kept separate from the human People
    object.

    This module is decoupled from any specific human disease model. It
    communicates with the human module through a shared NodeState object.

    Parameters
    ----------
    emergence : array-like of length 12
        Monthly adult female emergence rate in new mosquitoes per human per
        day. One value per calendar month (index 0 = January). Values between
        months are linearly interpolated; December wraps to January.

        At equilibrium the expected mosquito-to-human ratio is
        emergence / (1 / life_expectancy). Default: seasonal profile
        peaking at 1.0 in Jul-Aug (ratio ≈ 10) and troughing at 0.2 in
        Jan-Feb (ratio ≈ 2), assuming default life_expectancy=10.

    bites_per_female_per_day : float
        Average number of bites per adult female per day. Default: 0.3.

    life_expectancy : float
        Mean adult female lifespan in days. Default: 10.0 (corresponding to
        a daily mortality rate of 0.1).

    eip_days : float
        Extrinsic incubation period (EIP) in days — the time from a mosquito
        becoming exposed to becoming infectious. Default: 12.0.

    f_indoor : float
        Fraction of bites occurring indoors. Must be in [0, 1]. Default: 0.7.

    Examples
    --------
    Default seasonal profile::

        sim = ss.Sim(
            diseases=MalariaHumanSIS(),
            connectors=MosquitoesAdultFemaleAgentsOnly(),
        )

    Custom high-transmission wet-season profile::

        sim = ss.Sim(
            diseases=MalariaHumanSIS(),
            connectors=MosquitoesAdultFemaleAgentsOnly(
                emergence=[0.1, 0.1, 0.2, 0.5, 1.5, 3.0, 3.0, 1.5, 0.5, 0.2, 0.1, 0.1],
                life_expectancy=14.0,
                eip_days=10.0,
            ),
        )
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            emergence                = _DEFAULT_EMERGENCE.copy(),  # new females/human/day, by month
            bites_per_female_per_day = 0.3,                        # biting rate (bites/female/day)
            life_expectancy          = 10.0,                       # mean adult lifespan (days)
            eip_days                 = 12.0,                       # extrinsic incubation period (days)
            f_indoor                 = 0.7,                        # fraction of bites indoors
        )
        self.update_pars(pars, **kwargs)
        self._validate()

        # Mosquito agent arrays — allocated on the first step.
        self._max_mos      = 0
        self.alive         = None   # bool: slot is occupied by a live mosquito
        self.susceptible   = None   # bool: susceptible
        self.exposed       = None   # bool: exposed (not yet infectious)
        self.infectious    = None   # bool: infectious
        self.ti_infectious = None   # float: timestep when exposed → infectious
        self.ti_dead       = None   # float: timestep when mosquito dies
        self.node_id       = None   # int: spatial node index

        self._initialized = False
        self._ns_cache    = None
        return

    # ------------------------------------------------------------------
    # Starsim module lifecycle
    # ------------------------------------------------------------------

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_susceptible',      dtype=int,   label='Mosquitoes — susceptible'),
            ss.Result('n_exposed',          dtype=int,   label='Mosquitoes — exposed'),
            ss.Result('n_infectious',       dtype=int,   label='Mosquitoes — infectious'),
            ss.Result('n_total',            dtype=int,   label='Mosquitoes — total alive'),
            ss.Result('eir_annual',         dtype=float, label='EIR — total (bites/person/year)'),
            ss.Result('eir_indoor_annual',  dtype=float, label='EIR — indoor (bites/person/year)'),
            ss.Result('eir_outdoor_annual', dtype=float, label='EIR — outdoor (bites/person/year)'),
        )
        return

    def step(self):
        """
        Advance the mosquito population by one timestep.

        Order of operations:
          1. Allocate arrays on the first step (once n_humans is known).
          2. Kill mosquitoes whose scheduled death time has arrived.
          3. Birth new susceptible mosquitoes (per node from NodeGrid).
          4. Transition exposed → infectious when EIP has elapsed.
          5. Apply human → mosquito transmission (per node from NodeState).
          6. Record results and write per-node EIR to NodeState.
        """
        sim = self.sim
        ns  = self._get_node_state()

        # ------------------------------------------------------------------------
        # 1. Initialize mosquito arrays on the first step, once n_humans is known.
        # ------------------------------------------------------------------------
        if not self._initialized:
            self._init_mosquitoes(ns)
            self._initialized = True

        p       = self.pars
        ti      = sim.ti
        dt_days = sim.t.dt_year * DAYS_PER_YEAR

        # ----------
        # 2. Deaths
        # ----------
        dying = self.alive & (self.ti_dead <= ti)
        if dying.any():
            self._kill_mosquitoes(dying)

        # --------------------------------------------------
        # 3. Births — per node using n_humans from NodeState
        # --------------------------------------------------
        for nid in range(ns.grid.n_nodes):
            n_humans_node  = int(ns.n_humans[nid])
            emergence_rate = self._emergence_rate_for_node(ns.grid, nid, sim.t.year % 1)
            n_new          = int(np.random.poisson(emergence_rate * n_humans_node * dt_days))
            if n_new > 0:
                self._birth_mosquitoes(ti, dt_days, n_new, nid)

        # -----------------------
        # 4. Exposed → infectious
        # -----------------------
        newly_infectious = self.alive & self.exposed & (self.ti_infectious <= ti)
        if newly_infectious.any():
            self.exposed[newly_infectious]    = False
            self.infectious[newly_infectious] = True

        # ----------------------------------------------------------
        # 5. Human → mosquito transmission — per node from NodeState
        # ----------------------------------------------------------
        for nid in range(ns.grid.n_nodes):
            sus_idx = np.where(self.alive & self.susceptible & (self.node_id == nid))[0]
            if len(sus_idx) == 0:
                continue
            f             = ns.grid.f_indoor[nid] if ns.grid.f_indoor is not None else p.f_indoor
            p_inf_indoor  = ns.p_inf_indoor[nid]
            p_inf_outdoor = ns.p_inf_outdoor[nid]
            p_bite        = min(p.bites_per_female_per_day * dt_days, 1.0)
            bites_indoor  = np.random.random(len(sus_idx)) < f
            p_exposed     = np.where(bites_indoor, p_bite * p_inf_indoor, p_bite * p_inf_outdoor)
            gets_exposed  = np.random.random(len(sus_idx)) < p_exposed
            new_idx       = sus_idx[gets_exposed]
            if len(new_idx):
                eip_steps                   = p.eip_days / dt_days
                self.susceptible[new_idx]   = False
                self.exposed[new_idx]       = True
                self.ti_infectious[new_idx] = ti + eip_steps

        # -----------------------------------
        # 6. Results + write EIR to NodeState
        # -----------------------------------
        n_humans_total = int(sim.people.alive.sum())
        alive_mask     = self.alive
        n_alive        = int(alive_mask.sum())
        n_inf          = int((alive_mask & self.infectious).sum())
        annual_eir     = (
            n_inf * p.bites_per_female_per_day * DAYS_PER_YEAR / float(n_humans_total)
            if n_humans_total > 0 else 0.0
        )
        self.results.n_susceptible[ti]      = int((alive_mask & self.susceptible).sum())
        self.results.n_exposed[ti]          = int((alive_mask & self.exposed).sum())
        self.results.n_infectious[ti]       = n_inf
        self.results.n_total[ti]            = n_alive
        self.results.eir_annual[ti]         = annual_eir
        self.results.eir_indoor_annual[ti]  = annual_eir * p.f_indoor
        self.results.eir_outdoor_annual[ti] = annual_eir * (1.0 - p.f_indoor)

        for nid in range(ns.grid.n_nodes):
            n_h = int(ns.n_humans[nid])
            if n_h == 0:
                ns.eir_indoor[nid]  = 0.0
                ns.eir_outdoor[nid] = 0.0
                continue
            n_inf_node = int((alive_mask & self.infectious & (self.node_id == nid)).sum())
            f          = ns.grid.f_indoor[nid] if ns.grid.f_indoor is not None else p.f_indoor
            eir_dt     = n_inf_node * p.bites_per_female_per_day * dt_days / n_h
            ns.eir_indoor[nid]  = eir_dt * f
            ns.eir_outdoor[nid] = eir_dt * (1.0 - f)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_mosquitoes(self, ns):
        """
        Allocate mosquito agent arrays sized to 3× the maximum equilibrium
        population. Seed with the equilibrium count for the simulation start
        month, distributing across nodes proportionally to pop_fractions.
        """
        sim     = self.sim
        p       = self.pars
        dt_days = sim.t.dt_year * DAYS_PER_YEAR
        grid    = ns.grid

        n_humans = int(sim.people.alive.sum())

        # Maximum emergence across all nodes.
        max_em = float(np.max(p.emergence))
        if grid.emergence is not None:
            max_em = max(max_em, float(np.max(grid.emergence)))

        max_equil_per_human = max_em * p.life_expectancy
        self._max_mos = max(100, int(max_equil_per_human * n_humans * 3))

        self.alive         = np.zeros(self._max_mos, dtype=bool)
        self.susceptible   = np.zeros(self._max_mos, dtype=bool)
        self.exposed       = np.zeros(self._max_mos, dtype=bool)
        self.infectious    = np.zeros(self._max_mos, dtype=bool)
        self.ti_infectious = np.full(self._max_mos,  np.inf)
        self.ti_dead       = np.full(self._max_mos,  np.inf)
        self.node_id       = np.zeros(self._max_mos, dtype=int)

        # Seed starting population distributed across nodes.
        t_frac   = sim.t.year % 1
        mean_ls  = p.life_expectancy / dt_days
        slot     = 0
        for nid in range(grid.n_nodes):
            n_h_node       = int(round(n_humans * grid.pop_fractions[nid]))
            em_rate        = self._emergence_rate_for_node(grid, nid, t_frac)
            n_start_node   = min(int(round(em_rate * p.life_expectancy * n_h_node)),
                                 self._max_mos - slot)
            end            = slot + n_start_node
            self.alive[slot:end]       = True
            self.susceptible[slot:end] = True
            self.node_id[slot:end]     = nid
            self.ti_dead[slot:end]     = np.random.exponential(mean_ls, n_start_node).astype(int)
            slot = end
        return

    def _kill_mosquitoes(self, dying):
        """Mark dead mosquitoes as inactive, freeing their slots for new births."""
        self.alive[dying]         = False
        self.susceptible[dying]   = False
        self.exposed[dying]       = False
        self.infectious[dying]    = False
        self.ti_infectious[dying] = np.inf
        self.ti_dead[dying]       = np.inf
        return

    def _birth_mosquitoes(self, ti, dt_days, n_new, nid):
        """
        Place n_new susceptible mosquitoes into free array slots for node nid.

        If the buffer is exhausted a warning is printed and births are capped.
        The caller is responsible for computing n_new from the emergence rate.
        """
        if n_new == 0:
            return

        free_slots = np.where(~self.alive)[0]
        if len(free_slots) < n_new:
            print(
                f'Warning (MosquitoesAdultFemaleAgentsOnly): mosquito buffer full at '
                f'timestep {ti}. {n_new - len(free_slots)} births dropped. '
                f'Consider increasing the buffer by raising the emergence values.'
            )
            n_new = len(free_slots)
            if n_new == 0:
                return

        idx             = free_slots[:n_new]
        mean_life_steps = self.pars.life_expectancy / dt_days

        self.alive[idx]         = True
        self.susceptible[idx]   = True
        self.exposed[idx]       = False
        self.infectious[idx]    = False
        self.node_id[idx]       = nid
        self.ti_infectious[idx] = np.inf
        self.ti_dead[idx]       = ti + np.random.exponential(mean_life_steps, n_new).astype(int)
        return

    def _emergence_rate_for_node(self, grid, nid, t):
        """
        Return the emergence rate (new females per human per day) for node nid.

        Uses NodeGrid.emergence[nid] if provided, otherwise falls back to the
        module-level emergence parameter (same for all nodes).
        """
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
