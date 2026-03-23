"""
MalariaHumanBase: shared infrastructure for all MalariaHuman disease variants.

This module defines MalariaHumanBase, the base class that provides the common
biting platform and observable states shared by all MalariaHuman variants:

  MalariaHumanSIS   — simple SIS model (human_sis.py)
  MalariaHumanSEIS  — SEIS model with incubation period (human_seis.py)

Extension guide
---------------
To create a new variant, inherit from MalariaHumanBase and override:

  _do_disease_transitions()
      Called each step inside the step_state() template to progress disease
      states. Default: ss.SIS recovery (infected → susceptible). Override to
      change the dynamics — SEIS adds exposed→infected before calling super();
      a within-host model replaces the method entirely with parasite dynamics.

  set_prognoses(uids, sources)
      Called by step() for each newly exposed agent. Default ss.SIS behavior
      routes agents directly to infected and schedules recovery. Override to
      send agents to a different starting state (e.g. exposed for SEIS).

  _update_symptom_state()
      Placeholder called each step for infected agents. Override to compute
      symptom_state from infectiousness, hrp2_pg, and fever_F.

  step_die(uids)
      Called when agents die. Override to reset any extra states defined by
      the subclass, then call super().step_die(uids).

  step()
      EIR-based M→H transmission shared by all current variants. Override for
      fundamentally different mechanisms (e.g. explicit per-bite tracking).
"""

import numpy as np
import starsim as ss

_NORMAL_BODY_TEMP_F = 98.6   # normal human body temperature (°F)

# Symptom state codes stored in the per-agent symptom_state IntArr.
# Transition logic belongs in _update_symptom_state(); see that method for
# where to add clinical classification in future versions.
SYMPTOM_NONE          = 0   # asymptomatic / not currently infected
SYMPTOM_CLINICAL      = 1   # clinical malaria (fever, malaise)
SYMPTOM_SEVERE        = 2   # severe malaria
SYMPTOM_DISEASE_DEATH = 3   # fatal episode; agent will be killed this timestep


class MalariaHumanBase(ss.SIS):
    """
    Base class for all MalariaHuman disease variants.

    Provides shared malaria biting infrastructure and observable states.
    Subclasses override _do_disease_transitions() to implement different
    epidemiological models (SIS, SEIS, within-host, etc.) without repeating
    the shared infrastructure.

    step_state() template
    ---------------------
    Every timestep:
      1. _malaria_pre_step_state()  — init new agents, update rel_biting_rate
      2. _do_disease_transitions()  — disease-specific state changes (override this)
      3. _malaria_post_step_state() — symptom classification, disease deaths,
                                      sync observable states

    Transmission entry point
    ------------------------
    step() is overridden to implement EIR-based M→H transmission, bypassing
    Starsim's network machinery entirely. All human exposure comes through
    mosquito biting; contact networks are ignored.

    See the module docstring for the complete extension guide.

    Parameters
    ----------
    infection_duration_years : ss.normal
        Duration of infection in years.
        Default: Normal(mean=200 days, sd=30 days).

    prob_human_infected_on_bite : float
        Probability of human infection per infectious mosquito bite
        (Ross-MacDonald *b*). Default: 1.0.

    prob_mosquito_infected_on_bite : float
        Probability of mosquito infection per bite on an infectious human
        (Ross-MacDonald *c*). Default: 1.0.

    heterogeneity_shape : float
        Gamma shape parameter for individual biting heterogeneity.
        <= 0 → disabled (all agents get biting_heterogeneity = 1).
        Positive: Gamma(shape, scale=1/shape), mean=1, variance=1/shape.
        Smaller shape gives more over-dispersion. Default: 0.0.

    age_biting_ages : list of float or None
        Breakpoint ages (years) for the age-dependent biting profile.
        Paired with age_biting_values. None → flat profile.

    age_biting_values : list of float or None
        Relative biting rates at each breakpoint age; linearly interpolated,
        clamped at extremes, and re-normalized to mean 1 each step.
        None → flat profile.

    default_infectiousness : float
        infectiousness while infected (0–1; 1 = maximum). 0 when susceptible.
        Default: 1.0.

    default_hrp2_pg : float
        hrp2_pg while infected (pg/µL). 0 when susceptible. Default: 0.0.

    default_fever_F : float
        fever_F while infected (°F). 98.6°F (normal) when susceptible.
        Default: 98.6.

    init_prev : ss.bernoulli
        Fraction of agents infected at simulation start. Default: 1%.

    p_death : ss.bernoulli
        Probability of death per episode via scheduled ti_dead. Default: 0.
        Disease mortality is also driven by symptom_state reaching
        SYMPTOM_DISEASE_DEATH (see _update_symptom_state).
    """

    def __init__(self):
        super().__init__()
        self.define_pars(
            # -------------------------------------------------------------------
            # Starsim distributions give you the ability to reduce stochastic
            # variability - greatly reduce butterfly effects.  See DanK work.
            # -------------------------------------------------------------------

            # Override SIS defaults with malaria-specific values.
            infection_duration_years       = ss.normal(loc=200/365, scale=30/365),
            p_death                        = ss.bernoulli(p=0.0),
            init_prev                      = ss.bernoulli(p=0.01),
            # Neutralize the ss.SIS immunity system — malaria immunity is not
            # modelled at this level. imm_boost=0 means no immunity accumulates,
            # so update_immunity() is a no-op every step.
            imm_boost                      = 0.0,

            # Malaria-specific transmission parameters.
            prob_human_infected_on_bite    = 1.0,   # Ross-MacDonald *b*
            prob_mosquito_infected_on_bite = 1.0,   # Ross-MacDonald *c*

            # Biting heterogeneity: Gamma shape parameter.
            # <= 0 → disabled (biting_heterogeneity = 1 for all agents).
            # > 0  → Gamma(shape, scale=1/shape), mean=1, variance=1/shape.
            heterogeneity_shape = 0.0,

            # Age-dependent biting profile (None = flat).
            # Example: age_biting_ages=[0, 5, 15], age_biting_values=[0.5, 0.8, 1.0]
            age_biting_ages   = None,
            age_biting_values = None,

            # Per-agent observable state values while infected.
            # infectiousness: 0 when susceptible; 0–1 when infected (1 = maximum).
            default_infectiousness = 1.0,
            # hrp2_pg: 0 when susceptible; concentration (pg/µL) when infected.
            default_hrp2_pg        = 0.0,
            # fever_F: 98.6°F (normal) when susceptible; temperature when infected.
            default_fever_F        = 98.6,

            # Internal Bernoulli for per-timestep infection draws.
            # Dynamic callable gives each susceptible agent an individual
            # probability = EIR_dt * b * rel_biting_rate[agent].
            # _eir_total_dt is set in step() just before filter() is called.
            p_inf_bite = ss.bernoulli(
                p=lambda self, sim, uids: self._p_inf_bite_callable(sim, uids)
            ),
        )
        # Subclasses call update_pars(pars, **kwargs) after adding their own pars.

        # Per-agent states.
        # biting_heterogeneity : Gamma draw at agent creation; 0.0 = uninitialized sentinel.
        # rel_biting_rate      : heterogeneity × age factor, normalized to mean 1.
        # infectiousness       : 0 when susceptible; default_infectiousness when infected.
        # hrp2_pg              : 0 when susceptible; default_hrp2_pg when infected.
        # fever_F              : 98.6 when susceptible; default_fever_F when infected.
        # symptom_state        : SYMPTOM_* integer code; 0 = SYMPTOM_NONE.
        # ti_dead              : scheduled disease-death timestep (nan = not scheduled).
        #
        # Note: FloatArr defaults to nan and IntArr to -999999 when no default= is given.
        # Explicit defaults are required here to get sensible initial values.
        self.define_states(
            ss.FloatArr('biting_heterogeneity', default=0.0,                 label='Biting heterogeneity (Gamma-drawn, fixed per agent)'),
            ss.FloatArr('rel_biting_rate',      default=0.0,                 label='Relative biting rate (normalized, age × heterogeneity)'),
            ss.FloatArr('infectiousness',       default=0.0,                 label='Human-to-mosquito infectiousness (0 = susceptible)'),
            ss.FloatArr('hrp2_pg',              default=0.0,                 label='HRP2 concentration pg/µL (0 = susceptible)'),
            ss.FloatArr('fever_F',              default=_NORMAL_BODY_TEMP_F, label='Body temperature °F (98.6 = susceptible)'),
            ss.IntArr('symptom_state',          default=SYMPTOM_NONE,        label='Symptom state (SYMPTOM_* codes; 0 = none)'),
            ss.FloatArr('ti_dead',                                           label='Scheduled disease-death timestep (nan = none)'),
            ss.IntArr('node_id',                default=0,                   label='Spatial node index (0 when no NodeState)'),
        )

        # Holds the current-step total EIR; read by the p_inf_bite callable.
        self._eir_total_dt = 0.0
        # NodeState — resolved once on first call to _get_node_state().
        self._ns_cache = None
        return

    # ------------------------------------------------------------------
    # step_state template
    # ------------------------------------------------------------------

    def step_state(self):
        """
        Template: update biting states → disease transitions → sync observables.

        Subclasses should override _do_disease_transitions() to change disease
        dynamics rather than overriding this method.
        """
        self._malaria_pre_step_state()
        self._do_disease_transitions()
        self._malaria_post_step_state()
        return

    def _malaria_pre_step_state(self):
        """
        Initialize new agents and recompute normalized biting rates.

        Called at the start of step_state(), before disease transitions.
        New agents are detected by biting_heterogeneity == 0 (default FloatArr
        value), covering both the initial population and born agents.
        rel_biting_rate is recomputed every step because agents age continuously.
        When a NodeState is present, new agents are also assigned to a node
        drawn from the grid's pop_fractions (births use the same initial
        fractions so node populations remain stable over time).
        """
        alive  = self.sim.people.alive
        uninit = (self.biting_heterogeneity == 0.0) & alive
        if uninit.any():
            uids = uninit.uids
            # ----------------------------------------------------------------------------
            # laser-ngm has the biting_heterogeneity, but it did not implement setting it.
            # ----------------------------------------------------------------------------
            if float(self.pars.heterogeneity_shape) > 0:
                shape = float(self.pars.heterogeneity_shape)
                drawn = np.random.gamma(shape=shape, scale=1.0 / shape, size=len(uids))
                self.biting_heterogeneity[uids] = np.maximum(drawn, 1e-10)
            else:
                self.biting_heterogeneity[uids] = 1.0
            # ----------------------------------------------------------------------------
            # ----------------------------------------------------------------------------

            ns    = self._get_node_state()
            fracs = ns.grid.pop_fractions
            self.node_id[uids] = np.random.choice(len(fracs), size=len(uids), p=fracs)

        self._update_rel_biting_rate()
        return

    def set_prognoses(self, uids, sources=None):
        """
        Move newly inoculated agents into the infectious state (SIS baseline).

        Overrides ss.SIS.set_prognoses to convert dur_inf from years to
        timestep counts. ss.SIS stores ti_recovered as an integer step index;
        without this conversion agents recover after 1 step regardless of dt.

        Durations sampled from distribution (in years) are divided by dt_year
        to produce the correct step offset from the current ti.

        This is also the place to determine the infectiousness, hrp2_pg, and
        fever_F of newly infected agents.
        """
        # ---------------------------------------
        # laser-ngm's HumanIntraHost
        # ---------------------------------------
        ti      = self.ti
        dt_year = self.sim.t.dt_year
        p       = self.pars

        self.susceptible[uids] = False
        self.infected[uids]    = True
        self.ti_infected[uids] = ti
        self.immunity[uids]   += float(p.imm_boost)   # no-op when imm_boost=0

        dur_inf = p.dur_inf.rvs(uids) / dt_year       # years → steps
        self.ti_recovered[uids] = ti + dur_inf
        return

    def _do_disease_transitions(self):
        """
        Perform disease-state transitions for the current timestep.

        Default: SIS recovery — infected agents whose ti_recovered has elapsed
        transition back to susceptible (via ss.SIS.step_state).

        Override in subclasses to change the transition model:
          - SEIS: add exposed→infectious before calling super()
          - WithinHost: replace entirely with parasite-density dynamics
        """
        super().step_state()
        # ---------------------------------------------------------------
        # SIS.step_state() does the following:
        # ---------------------------------------------------------------
        # recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        # self.infected[recovered] = False
        # self.susceptible[recovered] = True
        # self.update_immunity()
        # ---------------------------------------------------------------

        return

    def _malaria_post_step_state(self):
        """
        Classify symptoms and request disease deaths.

        Called at the end of step_state(), after disease transitions, so that
        freshly infected agents appear in infected_alive.
        """

        # Update observables (infectiousness, hrp2_pg, fever_F) based on the new disease states.
        self._update_observables()

        # Classify symptoms for infected agents.
        self._update_symptom_state()

        # Kill agents whose symptom state reached DISEASE_DEATH.
        # request_death is processed at the same point as natural deaths.
        infected_alive    = self.infected & self.sim.people.alive
        symptom_deaths = (self.symptom_state == SYMPTOM_DISEASE_DEATH) & infected_alive
        if symptom_deaths.any():
            self.sim.people.request_death(symptom_deaths.uids)
            # -------------------------------------------
            # People.request_death() does the following:
            # -------------------------------------------
            # self.ti_dead[uids] = self.sim.ti
            # -------------------------------------------

        return

    def step_die(self, uids):
        """
        Reset all custom per-agent states when agents die (any cause).

        Subclasses that add extra states must override this and call super():

            def step_die(self, uids):
                super().step_die(uids)
                self.my_extra_state[uids] = 0
        """
        super().step_die(uids)
        self.symptom_state[uids]        = SYMPTOM_NONE
        self.biting_heterogeneity[uids] = 0.0
        self.rel_biting_rate[uids]      = 0.0
        self.infectiousness[uids]       = 0.0
        self.hrp2_pg[uids]              = 0.0
        self.fever_F[uids]              = 0.0
        self.ti_dead[uids]              = np.nan
        return

    # ------------------------------------------------------------------
    # Transmission — mosquito → human
    # ------------------------------------------------------------------

    def step(self):
        """
        EIR-based M→H transmission, bypassing Starsim's network machinery.

        Reads per-node EIR from NodeState (written by the mosquito connector
        in the previous timestep) and applies M→H transmission separately for
        susceptibles in each node.  NodeState is always present — a 1-node
        grid is created by make_sim() when no explicit spatial structure is
        configured.

        set_prognoses() is called for newly exposed agents. Override
        set_prognoses() in subclasses to route agents to a different starting
        state (e.g. the exposed state in SEIS).
        """
        ns = self._get_node_state()
        all_new_uids = []
        for nid in range(ns.grid.n_nodes):
            eir_total_dt = ns.eir_indoor[nid] + ns.eir_outdoor[nid]
            if eir_total_dt <= 0:
                continue
            sus_in_node = self.susceptible & (self.node_id == nid)
            if not sus_in_node.any():
                continue
            self._eir_total_dt = eir_total_dt
            # ----------------------------------------------------------------
            # p_inf_bite is the starsim special Bernoulli distribution where
            # _p_inf_bite_callable is called for each person and each person
            # is using the equivalent of their own random number generator.
            # ----------------------------------------------------------------
            new_uids = self.pars.p_inf_bite.filter(sus_in_node)

            # ---------------------------------
            # laser-ngm's m2h_has_new_infection
            # ---------------------------------
            all_new_uids.extend(new_uids)

        if all_new_uids:
            self.set_prognoses(np.array(all_new_uids), sources=None)

        self._write_node_state(ns)
        return

    # ------------------------------------------------------------------
    # Transmission — human → mosquito interface
    # ------------------------------------------------------------------
    # laser-ngm's HumanToMosquitoTransmission, but it doesn't have
    # the prob_mosquito_infected_on_bite multiplier.
    # ------------------------------------------------------------------

    def get_p_inf_mosquito_indoor(self, sim):
        """
        Probability that a susceptible mosquito biting indoors becomes exposed.

        Computed as the biting-rate-weighted mean infectiousness of all agents
        multiplied by prob_mosquito_infected_on_bite. infectiousness is 0 for
        susceptible agents, so only infectious agents contribute.

        Returns the same value as get_p_inf_mosquito_outdoor() until
        interventions that selectively protect indoor exposure are added.
        """
        return self._weighted_infectiousness() * self.pars.prob_mosquito_infected_on_bite

    def get_p_inf_mosquito_outdoor(self, sim):
        """
        Probability that a susceptible mosquito biting outdoors becomes exposed.

        Identical to get_p_inf_mosquito_indoor() in the base model. Will differ
        once indoor-specific interventions (ITNs, IRS) are added.
        """
        return self._weighted_infectiousness() * self.pars.prob_mosquito_infected_on_bite

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _p_inf_bite_callable(self, sim, uids):
        """
        Per-agent M→H infection probability for the p_inf_bite Bernoulli.

        Called by Starsim during p_inf_bite.filter(susceptible). Returns:

            p[i] = EIR_total_dt * prob_human_infected_on_bite * rel_biting_rate[i]

        _eir_total_dt is set in step() immediately before filter() is called.
        """
        # ----------------------------------------------------------------
        # laser-ngm has this too, but rel_biting_rate was not implemented.
        # laser-ngm has m2h_p_infection_scale which is sort of like this
        # ----------------------------------------------------------------
        p = (self._eir_total_dt
             * self.pars.prob_human_infected_on_bite
             * self.rel_biting_rate[uids])
        return np.clip(p, 0.0, 1.0)

    def _weighted_infectiousness(self):
        """
        Return biting-rate-weighted mean infectiousness across all alive agents.

        A mosquito's bite target is proportional to rel_biting_rate[i]. The
        expected probability of exposure per bite is:

            sum(rel_biting_rate[i] * infectiousness[i]) / n_alive

        Because rel_biting_rate is normalized to mean 1 over alive agents,
        n_alive = sum(rel_biting_rate[alive]).

        Subclasses with intermediate infectiousness values work automatically.
        """
        # ------------------------------------------------
        # Found in laser-ngm - HumanToMosquitoTransmission
        # ------------------------------------------------
        alive = self.sim.people.alive
        n_alive = alive.sum()
        if n_alive == 0:
            return 0.0
        numerator = float(
            (self.rel_biting_rate[alive] * self.infectiousness[alive]).sum()
        )
        return numerator / float(n_alive)

    def _update_rel_biting_rate(self):
        """
        Recompute rel_biting_rate for all alive agents and normalize to mean 1.

        rel_biting_rate[i] = biting_heterogeneity[i] × age_factor(age[i]),
        scaled so that mean(rel_biting_rate[alive]) = 1.
        """
        # ----------------------------------------------------------------
        # laser-ngm did not implement this
        # ----------------------------------------------------------------
        alive_uids = self.sim.people.alive.uids
        if len(alive_uids) == 0:
            return

        raw = np.array(self.biting_heterogeneity[alive_uids], dtype=float)

        p = self.pars
        if p.age_biting_ages is not None and p.age_biting_values is not None:
            ages       = np.asarray(p.age_biting_ages,   dtype=float)
            values     = np.asarray(p.age_biting_values, dtype=float)
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
        """
        Upate things that are 'observable' on the human side,
        like infectiousness, hrp2_pg, and fever_F.  Infectiousness
        is observable to mosquitoes; hrp2_pg, fever_F, and symptom_state are
        observable to the human immune system and to clinical surveillance.
        """
        infected_alive    = self.infected    & self.sim.people.alive
        susceptible_alive = self.susceptible & self.sim.people.alive

        # Determine observables so that _update_symptom_state() can
        # classify symptoms for newly infected agents.
        p = self.pars

        self.infectiousness[infected_alive]    = float(p.default_infectiousness)
        self.infectiousness[susceptible_alive] = 0.0

        self.hrp2_pg[infected_alive]    = float(p.default_hrp2_pg)
        self.hrp2_pg[susceptible_alive] = 0.0

        self.fever_F[infected_alive]    = float(p.default_fever_F)
        self.fever_F[susceptible_alive] = _NORMAL_BODY_TEMP_F

        return

    def _update_symptom_state(self):
        """
        Classify symptom severity for currently infected agents.

        Placeholder — Override in subclasses to implement
        clinical and severity classification based on parasite density, immune
        status, or age. Set self.symptom_state[uids] to the appropriate code:

            self.symptom_state[some_uids] = SYMPTOM_CLINICAL
            self.symptom_state[other_uids] = SYMPTOM_SEVERE
            self.symptom_state[fatal_uids] = SYMPTOM_DISEASE_DEATH

        Susceptible agents are already reset to SYMPTOM_NONE before this
        method is called.
        """
        # ---------------------------------------
        # laser-ngm's HumanSymptoms
        # ---------------------------------------
        # Reset symptom state for agents that just recovered or were never infected.
        susceptible_alive = self.susceptible & self.sim.people.alive
        self.symptom_state[susceptible_alive] = SYMPTOM_NONE

        return

    def _write_node_state(self, ns):
        """
        Write per-node human state to NodeState for the mosquito connector.

        Computes n_humans and biting-rate-weighted mean infectiousness for
        each node.  Indoor and outdoor p_inf are set equal (no indoor-specific
        interventions yet).
        """
        # ---------------------------------------
        # laser-ngm's HumanToMosquitoTransmission
        # ---------------------------------------
        alive = self.sim.people.alive
        c     = float(self.pars.prob_mosquito_infected_on_bite)
        for nid in range(ns.grid.n_nodes):
            alive_in_node = alive & (self.node_id == nid)
            n             = int(alive_in_node.sum())
            ns.n_humans[nid] = n
            if n == 0:
                ns.p_inf_indoor[nid]  = 0.0
                ns.p_inf_outdoor[nid] = 0.0
                continue
            uids = alive_in_node.uids
            w    = np.array(self.rel_biting_rate[uids], dtype=float)
            inf  = np.array(self.infectiousness[uids],  dtype=float)
            p    = float((w * inf).sum()) / n * c
            ns.p_inf_indoor[nid]  = p
            ns.p_inf_outdoor[nid] = p
        return

    def _get_node_state(self):
        """
        Return the NodeState connector, raising RuntimeError if not found.

        Searches sim.modules on the first call and caches the result.
        NodeState is always expected — make_sim() creates a 1-node
        NodeGrid/NodeState when no explicit spatial structure is configured.
        """
        if self._ns_cache is None:
            self._ns_cache = next(
                (m for m in self.sim.modules if getattr(m, '_is_node_state', False)),
                None,
            )
            if self._ns_cache is None:
                raise RuntimeError(
                    f'{type(self).__name__}._get_node_state(): no NodeState '
                    'module found in sim.modules.  Use make_sim() or add '
                    'NodeState as the first connector manually.'
                )
        return self._ns_cache
