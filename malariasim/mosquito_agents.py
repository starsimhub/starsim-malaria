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
Scheduled at ti_exposed + Exponential(eip_days / dt_days) so the mean
transition time equals eip_days, matching the ODE sigma_m = 1/eip_days.

Mosquito → human transmission (EIR)
-------------------------------------
Per-node EIR values are written to NodeState (``eir_indoor``,
``eir_outdoor``) at the end of each step for the human module to read.
"""

import numpy as np
import starsim as ss

from .utils import interpolate_monthly

# Default monthly emergence profile: seasonal with a wet-season peak.
# Values are new adult female mosquitoes per human per day.
# With default life_expectancy=ss.days(10) (mu=0.1/day):
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
                life_expectancy=ss.days(14.0),
                eip_days=ss.days(10.0),
            ),
        )
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            emergence                = _DEFAULT_EMERGENCE.copy(),  # new females/human/day, by month
            bites_per_female_per_day = 0.3,                        # biting rate (bites/female/day)
            life_expectancy          = ss.days(10.0),              # mean adult lifespan
            eip_days                 = ss.days(12.0),              # extrinsic incubation period
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
        ns = self._get_node_state()

        # ------------------------------------------------------------------------
        # 1. Initialize mosquito arrays on the first step, once n_humans is known.
        # ------------------------------------------------------------------------
        if not self._initialized:
            self._init_mosquitoes(ns)
            self._initialized = True

        # ----------
        # 2. Deaths
        # ----------
        dying = self.alive & (self.ti_dead <= self.ti)
        if dying.any():
            self._kill_mosquitoes(dying)

        # --------------------------------------------------
        # 3. Births — vectorized over all nodes
        # --------------------------------------------------
        n_new_per_node, total_new = self._sample_births(ns.grid, ns.n_humans)

        if total_new > 0:
            self._place_births(n_new_per_node, total_new, ns.grid.n_nodes)

        # -----------------------------------
        # 4. Intervention mosquito effects
        # -----------------------------------
        self._apply_itn_killing(ns)

        # -----------------------
        # 5. Exposed → infectious
        # -----------------------
        newly_infectious = self.alive & self.exposed & (self.ti_infectious <= self.ti)
        if newly_infectious.any():
            self.exposed[newly_infectious]    = False
            self.infectious[newly_infectious] = True

        # ----------------------------------------------------------
        # 6. Human → mosquito transmission — vectorized over all nodes
        # ----------------------------------------------------------
        sus_idx = np.where(self.alive & self.susceptible)[0]
        if len(sus_idx):
            self._expose_susceptible_mosquitoes(sus_idx, ns)

        # -----------------------------------
        # 7. Results + write EIR to NodeState
        # -----------------------------------
        alive_mask = self.alive
        self._record_results(alive_mask, int(np.asarray(ns.n_humans).sum()))
        self._write_node_eir(ns, alive_mask)
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
        p       = self.pars
        dt_days = self.dt.days
        grid    = ns.grid

        # Use NodeState n_humans (written by ODE human module on this same step)
        # rather than sim.people.alive.sum(), which gives n_nodes*n_bins dummy
        # slots for ODE humans instead of the real population count.
        n_h_per_node = np.asarray(ns.n_humans, dtype=float)
        n_humans     = int(n_h_per_node.sum()) or int(self.sim.people.alive.sum())

        # Maximum emergence across all nodes.
        max_em = float(np.max(p.emergence))
        if grid.emergence is not None:
            max_em = max(max_em, float(np.max(grid.emergence)))

        max_equil_per_human = max_em * p.life_expectancy.days
        self._max_mos = max(100, int(max_equil_per_human * n_humans * 3))

        self.alive         = np.zeros(self._max_mos, dtype=bool)
        self.susceptible   = np.zeros(self._max_mos, dtype=bool)
        self.exposed       = np.zeros(self._max_mos, dtype=bool)
        self.infectious    = np.zeros(self._max_mos, dtype=bool)
        self.ti_infectious = np.full(self._max_mos,  np.inf)
        self.ti_dead       = np.full(self._max_mos,  np.inf)
        self.node_id       = np.zeros(self._max_mos, dtype=int)

        # Seed starting population distributed across nodes.
        t_frac   = self.t.year % 1
        mean_ls  = p.life_expectancy.days / dt_days
        slot     = 0
        for nid in range(grid.n_nodes):
            n_h_node       = int(round(n_h_per_node[nid]))
            em_rate        = self._emergence_rate_for_node(grid, nid, t_frac)
            n_start_node   = min(int(round(em_rate * p.life_expectancy.days * n_h_node)),
                                 self._max_mos - slot)
            end            = slot + n_start_node
            self.alive[slot:end]       = True
            self.susceptible[slot:end] = True
            self.node_id[slot:end]     = nid
            self.ti_dead[slot:end]     = np.random.exponential(mean_ls, n_start_node).astype(int)
            slot = end
        return

    def _apply_itn_killing(self, ns):
        """Extra mortality from insecticide contact for all alive mosquitoes."""
        alive_idx = np.where(self.alive)[0]
        if not len(alive_idx):
            return
        itn_kp = np.asarray(ns.itn_kill_prob)
        if not itn_kp.any():
            return
        nids  = self.node_id[alive_idx]
        f_arr = (np.asarray(ns.grid.f_indoor)
                 if ns.grid.f_indoor is not None
                 else np.full(ns.grid.n_nodes, self.pars.f_indoor))
        p_kill = np.clip(
            self.pars.bites_per_female_per_day * self.dt.days * f_arr[nids] * itn_kp[nids],
            0.0, 1.0,
        )
        killed_local = np.where(np.random.random(len(alive_idx)) < p_kill)[0]
        if not len(killed_local):
            return
        kill_mask = np.zeros(self._max_mos, dtype=bool)
        kill_mask[alive_idx[killed_local]] = True
        self._kill_mosquitoes(kill_mask)

    def _kill_mosquitoes(self, dying):
        """Mark dead mosquitoes as inactive, freeing their slots for new births."""
        self.alive[dying]         = False
        self.susceptible[dying]   = False
        self.exposed[dying]       = False
        self.infectious[dying]    = False
        self.ti_infectious[dying] = np.inf
        self.ti_dead[dying]       = np.inf
        return

    def _record_results(self, alive_mask, n_humans_total):
        """
        Record per-timestep mosquito population counts and aggregate EIR.

        Parameters
        ----------
        alive_mask     : bool ndarray — self.alive (pre-fetched by caller)
        n_humans_total : int  — total alive humans this timestep
        """
        p     = self.pars
        n_inf = int((alive_mask & self.infectious).sum())
        if n_humans_total > 0:
            eir_per_step = n_inf * p.bites_per_female_per_day * self.dt.days / n_humans_total
            annual_eir   = eir_per_step / self.dt.years
        else:
            annual_eir = 0.0
        self.results.n_susceptible[self.ti]      = int((alive_mask & self.susceptible).sum())
        self.results.n_exposed[self.ti]          = int((alive_mask & self.exposed).sum())
        self.results.n_infectious[self.ti]       = n_inf
        self.results.n_total[self.ti]            = int(alive_mask.sum())
        self.results.eir_annual[self.ti]         = annual_eir
        self.results.eir_indoor_annual[self.ti]  = annual_eir * p.f_indoor
        self.results.eir_outdoor_annual[self.ti] = annual_eir * (1.0 - p.f_indoor)
        return

    def _write_node_eir(self, ns, alive_mask):
        """
        Write per-node EIR to NodeState for the human module to read next step.

        Parameters
        ----------
        ns         : NodeState — eir_indoor/outdoor arrays to be written
        alive_mask : bool ndarray — self.alive (pre-fetched by caller)
        """
        p              = self.pars
        dt_days        = self.dt.days
        n_nodes        = ns.grid.n_nodes
        inf_node_ids   = self.node_id[alive_mask & self.infectious]
        n_inf_per_node = np.bincount(inf_node_ids, minlength=n_nodes).astype(float)
        n_h            = ns.n_humans
        live           = n_h > 0
        _f    = ns.grid.f_indoor if ns.grid.f_indoor is not None else p.f_indoor
        f_arr = np.full(n_nodes, _f) if np.isscalar(_f) else np.asarray(_f, dtype=float)
        itn_br         = np.asarray(ns.itn_biting_reduction)
        a              = p.bites_per_female_per_day
        eir_indoor_dt  = np.zeros(n_nodes)
        eir_outdoor_dt = np.zeros(n_nodes)
        eir_indoor_dt[live]  = n_inf_per_node[live] * a * f_arr[live] * (1.0 - itn_br[live]) * dt_days / n_h[live]
        eir_outdoor_dt[live] = n_inf_per_node[live] * a * (1.0 - f_arr[live])                 * dt_days / n_h[live]
        ns.eir_indoor[:]  = eir_indoor_dt
        ns.eir_outdoor[:] = eir_outdoor_dt
        return

    def _expose_susceptible_mosquitoes(self, sus_idx, ns):
        """
        Apply human → mosquito transmission for the current timestep.

        Each alive susceptible mosquito bites indoors or outdoors (Bernoulli
        draw using per-node f_indoor), then becomes exposed with probability
        p_bite * p_inf_indoor/outdoor read from NodeState.  Newly exposed
        mosquitoes have ti_infectious scheduled at ti + EIP.

        Parameters
        ----------
        sus_idx : ndarray of int — indices of alive susceptible mosquitoes
        ns      : NodeState — carries p_inf_indoor/outdoor and grid.f_indoor
        """
        p       = self.pars
        dt_days = self.dt.days
        nids    = self.node_id[sus_idx]
        f_arr   = (ns.grid.f_indoor[nids]
                   if ns.grid.f_indoor is not None
                   else p.f_indoor)
        p_bite        = p.bites_per_female_per_day * dt_days
        itn_br        = np.asarray(ns.itn_biting_reduction)[nids]
        p_bite_indoor = np.clip(p_bite * (1.0 - itn_br), 0.0, 1.0)
        p_bite_out    = min(p_bite, 1.0)
        bites_indoor  = np.random.random(len(sus_idx)) < f_arr
        p_inf_indoor  = ns.p_inf_indoor[nids]
        p_inf_outdoor = ns.p_inf_outdoor[nids]
        p_exposed     = np.where(bites_indoor,
                                 p_bite_indoor * p_inf_indoor,
                                 p_bite_out    * p_inf_outdoor)
        new_idx       = sus_idx[np.random.random(len(sus_idx)) < p_exposed]
        if len(new_idx):
            self.susceptible[new_idx]   = False
            self.exposed[new_idx]       = True
            self.ti_infectious[new_idx] = self.ti + np.random.exponential(
                p.eip_days.days / dt_days, len(new_idx)
            )
        return

    def _sample_births(self, grid, n_humans):
        """
        Draw the number of new susceptible mosquitoes per node for this timestep.

        Interpolates emergence rates for all nodes simultaneously (t_frac is
        the same for every node, so the month interval and alpha are computed
        once and broadcast across the profile matrix).  A single vectorized
        Poisson draw then produces per-node birth counts.

        Parameters
        ----------
        grid     : NodeGrid — spatial configuration (provides emergence profiles)
        n_humans : array-like of shape (n_nodes,) — current human count per node

        Returns
        -------
        n_new_per_node : ndarray of int, shape (n_nodes,)
        total_new      : int — sum of n_new_per_node
        """
        t_frac  = self.t.year % 1
        dt_days = self.dt.days
        if grid.emergence is not None:
            profiles = np.asarray(grid.emergence)                            # (n_nodes, 12)
        else:
            profiles = np.tile(np.asarray(self.pars.emergence), (grid.n_nodes, 1))
        x_months = np.arange(12) / 12.0
        x_wrap   = np.append(x_months, 1.0)                                 # (13,)
        y_wrap   = np.concatenate([profiles, profiles[:, :1]], axis=1)      # (n_nodes, 13)
        lo       = max(0, int(np.searchsorted(x_wrap, t_frac, side='right')) - 1)
        alpha    = (t_frac - x_wrap[lo]) / (x_wrap[lo + 1] - x_wrap[lo])
        rates    = y_wrap[:, lo] + alpha * (y_wrap[:, lo + 1] - y_wrap[:, lo])  # (n_nodes,)

        lambdas        = rates * n_humans * dt_days
        n_new_per_node = np.random.poisson(lambdas).astype(int)
        return n_new_per_node, int(n_new_per_node.sum())

    def _place_births(self, n_new_per_node, total_new, n_nodes):
        """
        Fill free mosquito-array slots with newly emerged susceptibles.

        Finds all free slots once, then assigns node IDs in bulk using
        np.repeat.  If the buffer is exhausted, nodes are filled in order
        until slots run out and a warning is printed.

        Parameters
        ----------
        n_new_per_node : ndarray of int, shape (n_nodes,) — births per node
        total_new      : int — sum of n_new_per_node (pre-computed by caller)
        n_nodes        : int — number of spatial nodes
        """
        dt_days    = self.dt.days
        free_slots = np.where(~self.alive)[0]
        if len(free_slots) < total_new:
            print(
                f'Warning (MosquitoesAdultFemaleAgentsOnly): mosquito buffer full at '
                f'timestep {self.ti}. {total_new - len(free_slots)} births dropped. '
                f'Consider increasing the buffer by raising the emergence values.'
            )
            # Cap nodes in order until free slots are exhausted.
            capped         = np.minimum(np.cumsum(n_new_per_node), len(free_slots))
            n_new_per_node = np.diff(capped, prepend=0)
            total_new      = int(n_new_per_node.sum())

        if total_new == 0:
            return

        idx             = free_slots[:total_new]
        node_ids_new    = np.repeat(np.arange(n_nodes), n_new_per_node)
        mean_life_steps = self.pars.life_expectancy.days / dt_days

        self.alive[idx]         = True
        self.susceptible[idx]   = True
        self.exposed[idx]       = False
        self.infectious[idx]    = False
        self.node_id[idx]       = node_ids_new
        self.ti_infectious[idx] = np.inf
        self.ti_dead[idx]       = self.ti + np.random.exponential(mean_life_steps, total_new).astype(int)
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
