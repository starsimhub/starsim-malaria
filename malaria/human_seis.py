"""
MalariaHumanSEIS: SEIS malaria disease module for human agents.

Extends MalariaHumanBase with an explicit exposed (incubation) state:

    susceptible → exposed → infected → susceptible

Newly inoculated agents enter the exposed compartment and remain there for
a Normally-distributed exposure duration (the extrinsic incubation period
on the human side) before becoming infectious. This gives the model a more
realistic delay between inoculation and infectiousness.

Terminology note
----------------
"Infected" in the epidemiological sense (carrying the parasite) happens at the
moment of the infectious bite. However in Starsim the BoolState named ``infected``
represents *infectious* — able to transmit to mosquitoes. The ``exposed`` state
represents the incubation window: the human is biologically infected but not yet
able to transmit.

Epidemiological state transitions
----------------------------------
  susceptible → exposed      : on each infectious mosquito bite (step / set_prognoses)
  exposed     → infectious   : when ti_infected elapses (_do_disease_transitions)
                               (stored as ``infected`` BoolState = able to transmit)
  infectious  → susceptible  : when ti_recovered elapses (ss.SIS.step_state via super)

Observable states during incubation
-------------------------------------
Exposed agents are neither susceptible nor infectious, so _malaria_post_step_state
does not modify their observable states. They retain:
  infectiousness  = 0        (not yet infectious to mosquitoes)
  hrp2_pg         = 0        (no detectable antigen)
  fever_F         = 98.6     (normal body temperature)
  symptom_state   = SYMPTOM_NONE

These values are biologically appropriate for the pre-patent incubation period.

For the shared biting infrastructure and full parameter reference, see
MalariaHumanBase in human_base.py.
"""

import numpy as np
import starsim as ss

from .human_base import MalariaHumanBase


class MalariaHumanSEIS(MalariaHumanBase):
    """
    SEIS malaria disease module: susceptible → exposed → infected → susceptible.

    Adds an explicit incubation (exposed) state to MalariaHumanBase. Newly
    bitten agents enter exposed rather than infected directly; they become
    infectious after a Normally-distributed exposure duration.

    Parameters
    ----------
    dur_exp : ss.normal
        Duration of the exposed (pre-infectious) period in years.
        Default: Normal(mean=12 days, sd=2 days).

    All other parameters are inherited from MalariaHumanBase.

    Examples
    --------
    Default incubation period::

        human = MalariaHumanSEIS()

    Shorter, less variable incubation::

        human = MalariaHumanSEIS(dur_exp=ss.normal(loc=10/365, scale=1/365))
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            # Exposed (incubation) period.
            # Default: ~12 days mean, ~2 days sd, matching P. falciparum EIP.
            # Values are in years; set_prognoses converts to timesteps via dt_year.
            dur_exp = ss.normal(loc=12/365, scale=2/365),
        )
        self.update_pars(pars, **kwargs)
        # ss.SIS.set_prognoses draws duration from self.pars.dur_inf.
        # Point it at infection_duration_years so user overrides are respected.
        self.pars.dur_inf = self.pars.infection_duration_years

        # Additional states for the exposed compartment.
        # exposed    : agents currently incubating (not yet infectious).
        # ti_exposed : timestep when the agent entered exposed.
        # ti_dead is inherited from MalariaHumanBase.
        self.define_states(
            ss.BoolState('exposed',   label='Exposed (incubating, pre-infectious)'),
            ss.FloatArr('ti_exposed', label='Time step when agent became exposed'),
        )
        return

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def set_prognoses(self, uids, sources=None):
        """
        Route newly inoculated agents into the exposed state.

        Unlike SIS (where agents go directly to infected), SEIS agents first
        spend dur_exp timesteps in exposed before becoming infectious.

        Timeline (from current timestep ti):
          ti_exposed   = ti                          (entered exposed now)
          ti_infected  = ti + dur_exp / dt_year      (will become infectious after incubation)
          ti_recovered = ti + (dur_exp + dur_inf) / dt_year  (will recover after full duration)

        Durations (dur_exp, dur_inf) are sampled in years and divided by dt_year
        to convert them to integer timestep counts, matching how Starsim stores ti.
        """
        ti      = self.ti
        dt_year = self.sim.t.dt_year
        p       = self.pars

        # Remove from susceptible and enter exposed.
        self.susceptible[uids] = False
        self.exposed[uids]     = True
        self.ti_exposed[uids]  = ti

        # Draw individual durations in years, then convert to timestep counts.
        dur_exp = p.dur_exp.rvs(uids) / dt_year
        dur_inf = p.dur_inf.rvs(uids) / dt_year

        # Schedule the exposed → infectious transition.
        self.ti_infected[uids] = ti + dur_exp

        # All agents recover — disease death is not yet implemented.
        self.ti_recovered[uids] = ti + dur_exp + dur_inf

        return

    def _do_disease_transitions(self):
        """
        Progress exposed → infectious, then run SIS recovery (infectious → susceptible).

        Agents who have completed incubation (ti_infected has arrived) leave the
        exposed compartment and become infectious. The parent ss.SIS.step_state
        then handles the infectious → susceptible recovery for the full infectious
        population.

        Note on naming: in this model "infected" (the BoolState) means *infectious*
        — able to transmit to mosquitoes. An agent is biologically infected from the
        moment of inoculation; it is *infectious* only after the incubation period
        represented by the exposed compartment.
        """
        ti = self.ti

        # Exposed (incubating) → infectious: agents who have completed incubation
        # and are now able to transmit to mosquitoes.
        becoming_infectious = self.exposed & (self.ti_infected <= ti)
        if becoming_infectious.any():
            uids = becoming_infectious.uids
            self.exposed[uids]  = False
            self.infected[uids] = True

        # SIS recovery: infectious → susceptible (via ss.SIS.step_state).
        super()._do_disease_transitions()
        return

    def step_die(self, uids):
        """Reset exposed state for dying agents, then call base cleanup."""
        super().step_die(uids)
        self.exposed[uids]    = False
        self.ti_exposed[uids] = np.nan
        return
