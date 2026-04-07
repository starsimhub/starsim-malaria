"""
MalariaHumanBase: shared infrastructure for all MalariaHuman disease variants.

Consolidates the LASER components (HumanIntraHost, HumanSymptoms,
MosquitoToHumanTransmission, HumanToMosquitoTransmission) into a single
Starsim disease class inheriting from ss.SIS.

Extension guide:
  - Override _do_disease_transitions() for disease-specific state changes
  - Override set_prognoses(uids, sources) for entry-point behavior
  - Override _update_symptom_state() for symptom classification
  - Override step_die(uids) to reset extra states (call super())
"""

import numpy as np
import starsim as ss

_NORMAL_BODY_TEMP_F = 98.6

# Symptom state codes
SYMPTOM_NONE = 0
SYMPTOM_CLINICAL = 1
SYMPTOM_SEVERE = 2
SYMPTOM_DISEASE_DEATH = 3


class MalariaHumanBase(ss.SIS):
    """
    Base class for all MalariaHuman disease variants.

    Provides EIR-based M→H transmission, H→M infectiousness writing,
    biting heterogeneity, and observable states. Subclasses override
    _do_disease_transitions() to change disease dynamics.
    """

    def __init__(self):
        super().__init__()
        self.define_pars(
            # Override SIS defaults with malaria-specific values
            infection_duration_years=ss.normal(loc=120/365, scale=30/365),
            p_death=ss.bernoulli(p=0.0),
            init_prev=ss.bernoulli(p=0.01),
            imm_boost=0.0,  # disable ss.SIS immunity

            # Transmission parameters
            prob_human_infected_on_bite=1.0,     # Ross-MacDonald b
            prob_mosquito_infected_on_bite=1.0,  # Ross-MacDonald c

            # Biting heterogeneity (Gamma shape; <=0 disables)
            heterogeneity_shape=0.0,

            # Age-dependent biting profile (None = flat)
            age_biting_ages=None,
            age_biting_values=None,

            # Observable state defaults while infected
            default_infectiousness=1.0,
            default_hrp2_pg=0.0,
            default_fever_F=98.6,

            # Internal Bernoulli for per-agent infection draws
            p_inf_bite=ss.bernoulli(
                p=lambda self, sim, uids: self._p_inf_bite_callable(sim, uids)
            ),
        )

        self.define_states(
            ss.FloatArr('biting_heterogeneity', default=0.0, label='Biting heterogeneity'),
            ss.FloatArr('rel_biting_rate', default=0.0, label='Relative biting rate'),
            ss.FloatArr('infectiousness', default=0.0, label='Human-to-mosquito infectiousness'),
            ss.FloatArr('hrp2_pg', default=0.0, label='HRP2 concentration pg/µL'),
            ss.FloatArr('fever_F', default=_NORMAL_BODY_TEMP_F, label='Body temperature °F'),
            ss.IntArr('symptom_state', default=SYMPTOM_NONE, label='Symptom state'),
            ss.FloatArr('ti_dead', label='Scheduled disease-death timestep'),
            ss.IntArr('node_id', default=0, label='Spatial node index'),
        )

        self._ns_cache = None
        return

    # ------------------------------------------------------------------
    # step_state template
    # ------------------------------------------------------------------

    def step_state(self):
        """Template: pre-step → disease transitions → post-step."""
        self._malaria_pre_step_state()
        self._do_disease_transitions()
        self._malaria_post_step_state()
        return

    def _malaria_pre_step_state(self):
        """Initialize new agents and recompute biting rates."""
        alive = self.sim.people.alive
        uninit = (self.biting_heterogeneity == 0.0) & alive
        if uninit.any():
            uids = uninit.uids
            if float(self.pars.heterogeneity_shape) > 0:
                shape = float(self.pars.heterogeneity_shape)
                drawn = np.random.gamma(shape=shape, scale=1.0/shape, size=len(uids))
                self.biting_heterogeneity[uids] = np.maximum(drawn, 1e-10)
            else:
                self.biting_heterogeneity[uids] = 1.0

            # Assign to nodes via pop_fractions
            ns = self._get_node_state()
            fracs = ns.grid.pop_fractions
            self.node_id[uids] = np.random.choice(len(fracs), size=len(uids), p=fracs)

        self._update_rel_biting_rate()
        return

    def _do_disease_transitions(self):
        """Default SIS recovery. Override in subclasses."""
        super().step_state()
        return

    def _malaria_post_step_state(self):
        """Update observables, classify symptoms, handle disease deaths."""
        self._update_observables()
        self._update_symptom_state()

        # Kill agents whose symptom state reached DISEASE_DEATH
        infected_alive = self.infected & self.sim.people.alive
        symptom_deaths = (self.symptom_state == SYMPTOM_DISEASE_DEATH) & infected_alive
        if symptom_deaths.any():
            self.sim.people.request_death(symptom_deaths.uids)
        return

    # ------------------------------------------------------------------
    # step() — EIR-based M→H transmission
    # ------------------------------------------------------------------

    def step(self):
        """
        EIR-based M→H transmission, bypassing Starsim's network machinery.

        Reads per-node EIR from NodeState and applies M→H transmission to
        all susceptibles via a Bernoulli filter with per-agent probability.
        """
        ns = self._get_node_state()
        new_uids = self.pars.p_inf_bite.filter(self.susceptible)
        if len(new_uids):
            self.set_prognoses(new_uids, sources=None)
        self._write_node_state(ns)
        return

    def set_prognoses(self, uids, sources=None):
        """Move newly inoculated agents to infected state (SIS baseline)."""
        ti = self.ti
        dt_year = self.sim.t.dt_year
        p = self.pars

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.immunity[uids] += float(p.imm_boost)

        dur_inf = p.dur_inf.rvs(uids) / dt_year  # years → steps
        self.ti_recovered[uids] = ti + dur_inf
        return

    # ------------------------------------------------------------------
    # step_die — reset custom states
    # ------------------------------------------------------------------

    def step_die(self, uids):
        """Reset all custom per-agent states when agents die."""
        super().step_die(uids)
        self.symptom_state[uids] = SYMPTOM_NONE
        self.biting_heterogeneity[uids] = 0.0
        self.rel_biting_rate[uids] = 0.0
        self.infectiousness[uids] = 0.0
        self.hrp2_pg[uids] = 0.0
        self.fever_F[uids] = 0.0
        self.ti_dead[uids] = np.nan
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _p_inf_bite_callable(self, sim, uids):
        """Per-agent M→H infection probability for p_inf_bite Bernoulli (LASER M2H)."""
        ns = self._get_node_state()
        node_ids = np.asarray(self.node_id[uids], dtype=int)
        eir = np.asarray(ns.eir_indoor)[node_ids] + np.asarray(ns.eir_outdoor)[node_ids]
        scale = float(self.pars.prob_human_infected_on_bite)
        w = np.asarray(self.rel_biting_rate[uids], dtype=float)
        mu = np.clip(eir * w * scale, 0.0, 50.0)
        return 1.0 - np.exp(-mu)

    def _update_rel_biting_rate(self):
        """Recompute rel_biting_rate for all alive agents, normalized to mean 1."""
        alive_uids = self.sim.people.alive.uids
        if len(alive_uids) == 0:
            return

        raw = np.array(self.biting_heterogeneity[alive_uids], dtype=float)

        p = self.pars
        if p.age_biting_ages is not None and p.age_biting_values is not None:
            ages = np.asarray(p.age_biting_ages, dtype=float)
            values = np.asarray(p.age_biting_values, dtype=float)
            agent_ages = np.asarray(self.sim.people.age[alive_uids], dtype=float)
            age_factor = np.interp(agent_ages, ages, values,
                                   left=float(values[0]), right=float(values[-1]))
            raw = raw * age_factor

        mean_raw = raw.mean()
        if mean_raw > 0:
            raw = raw / mean_raw

        self.rel_biting_rate[alive_uids] = raw
        return

    def _update_observables(self):
        """Update infectiousness, hrp2_pg, fever_F based on disease states."""
        infected_alive = self.infected & self.sim.people.alive
        susceptible_alive = self.susceptible & self.sim.people.alive

        p = self.pars
        self.infectiousness[infected_alive] = float(p.default_infectiousness)
        self.infectiousness[susceptible_alive] = 0.0

        self.hrp2_pg[infected_alive] = float(p.default_hrp2_pg)
        self.hrp2_pg[susceptible_alive] = 0.0

        self.fever_F[infected_alive] = float(p.default_fever_F)
        self.fever_F[susceptible_alive] = _NORMAL_BODY_TEMP_F
        return

    def _update_symptom_state(self):
        """Placeholder — override in subclasses for clinical classification."""
        susceptible_alive = self.susceptible & self.sim.people.alive
        self.symptom_state[susceptible_alive] = SYMPTOM_NONE
        return

    def _write_node_state(self, ns):
        """Write per-node human state to NodeState for the mosquito connector."""
        n_nodes = ns.grid.n_nodes
        alive_uids = self.sim.people.alive.uids
        c = float(self.pars.prob_mosquito_infected_on_bite)

        node_ids = np.asarray(self.node_id[alive_uids], dtype=int)
        w = np.asarray(self.rel_biting_rate[alive_uids], dtype=float)
        inf = np.asarray(self.infectiousness[alive_uids], dtype=float)

        n_humans = np.bincount(node_ids, minlength=n_nodes)
        w_inf_sum = np.bincount(node_ids, weights=w * inf, minlength=n_nodes)

        safe_n = np.where(n_humans > 0, n_humans, 1)
        p_inf = c * w_inf_sum / safe_n
        p_inf[n_humans == 0] = 0.0

        ns.n_humans[:] = n_humans
        ns.p_inf_indoor[:] = p_inf
        ns.p_inf_outdoor[:] = p_inf
        return

    def _get_node_state(self):
        """Return the NodeState connector (cached after first lookup)."""
        if self._ns_cache is None:
            self._ns_cache = next(
                (m for m in self.sim.modules if getattr(m, '_is_node_state', False)),
                None,
            )
            if self._ns_cache is None:
                raise RuntimeError(
                    f'{type(self).__name__}._get_node_state(): no NodeState '
                    'module found. Use Sim() or add NodeState manually.'
                )
        return self._ns_cache
