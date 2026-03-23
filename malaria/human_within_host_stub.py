"""
MalariaHumanWithinHost: within-host parasite dynamics malaria module.

Extends MalariaHumanBase by replacing the simple SIS timed recovery with
explicit per-agent parasite and immune dynamics. Recovery is no longer
scheduled at infection time; instead agents clear infection when their
parasite density falls below a threshold.

Placeholder methods that need real biology:
  _update_parasite_density    — asexual parasite growth / drug clearance
  _update_gametocyte_density  — sexual-stage production from asexual parasites
  _update_immune_response     — acquisition and decay of immunity
  _check_recovery             — clearance threshold → susceptible transition

NOTE: This file is a design stub. It is not imported by __init__.py and is
not part of the active simulation framework. Integrate by:
  1. Filling in the placeholder methods with real within-host biology.
  2. Importing MalariaHumanWithinHost from this file in __init__.py.
  3. Adding 'within_host' as an option in make_sim().
"""

import numpy as np
import starsim as ss

from .human_base import MalariaHumanBase, SYMPTOM_NONE, SYMPTOM_CLINICAL, SYMPTOM_SEVERE


# Thresholds (placeholders — replace with literature values)
_PARASITE_DENSITY_SYMPTOMATIC  = 1_000.0   # parasites/µL → clinical symptoms
_PARASITE_DENSITY_SEVERE       = 50_000.0  # parasites/µL → severe malaria
_PARASITE_DENSITY_CLEARANCE    =     10.0  # parasites/µL → infection cleared


class MalariaHumanWithinHost(MalariaHumanBase):
    """
    Within-host malaria model: parasite and immune dynamics drive outcomes.

    Per-agent states added
    ----------------------
    parasite_density   : asexual blood-stage parasites per µL
    gametocyte_density : sexual-stage parasites per µL (drives mosquito infectiousness)
    immune_response    : unitless 0–1 immunity level (0 = naive, 1 = fully immune)

    Key behavioural differences from MalariaHumanSIS
    --------------------------------------------------
    - Recovery is not scheduled at infection time; it occurs when
      parasite_density drops below _PARASITE_DENSITY_CLEARANCE.
    - infectiousness is derived from gametocyte_density each step rather
      than set to a fixed default_infectiousness parameter.
    - _update_symptom_state classifies severity from parasite_density.

    Parameters
    ----------
    inoculation_parasites : float
        Initial parasite density (parasites/µL) at the moment of infection.
        Default: 10.0 (low inoculum; replace with a distribution).

    All other parameters are inherited from MalariaHumanBase.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            inoculation_parasites = 10.0,   # parasites/µL at time of infection
            # Add growth rate, clearance rate, gametocyte conversion fraction,
            # immune acquisition rate, immune decay rate, drug effect, etc.
        )
        self.update_pars(pars, **kwargs)
        # dur_inf is unused (recovery is density-driven), but ss.SIS internals
        # may still reference it; keep it consistent with infection_duration_years.
        self.pars.dur_inf = self.pars.infection_duration_years

        self.define_states(
            ss.FloatArr('parasite_density',   label='Asexual parasite density (parasites/µL)'),
            ss.FloatArr('gametocyte_density', label='Gametocyte density (parasites/µL)'),
            ss.FloatArr('immune_response',    label='Immunity level (0 = naive, 1 = immune)'),
        )
        return

    # ------------------------------------------------------------------
    # Infection entry point
    # ------------------------------------------------------------------

    def set_prognoses(self, uids, sources=None):
        """
        Seed within-host state for newly inoculated agents.

        Unlike SIS/SEIS, we do not schedule ti_recovered here — recovery is
        determined dynamically by _check_recovery() each step.
        """
        self.susceptible[uids]      = False
        self.infected[uids]         = True
        self.parasite_density[uids] = float(self.pars.inoculation_parasites)
        # gametocyte_density and immune_response are left at their current
        # values (0 for naïve agents; >0 if previously exposed).
        return

    # ------------------------------------------------------------------
    # Disease transitions (Template Method override)
    # ------------------------------------------------------------------

    def _do_disease_transitions(self):
        """
        Run one step of within-host dynamics for all infected agents.

        Does NOT call super()._do_disease_transitions() because recovery is
        density-driven, not schedule-driven. The infected→susceptible
        transition is handled here by _check_recovery().
        """
        alive         = self.sim.people.alive
        infected_uids = ss.uids(self.infected & alive)

        if len(infected_uids) == 0:
            return

        self._update_parasite_density(infected_uids)
        self._update_gametocyte_density(infected_uids)
        self._update_immune_response(infected_uids)
        self._check_recovery(infected_uids)
        return

    # ------------------------------------------------------------------
    # Observable state sync (override to use gametocyte_density)
    # ------------------------------------------------------------------

    def _malaria_post_step_state(self):
        """
        Same as base, but derives infectiousness from gametocyte density
        instead of the fixed default_infectiousness parameter.
        """
        super()._malaria_post_step_state()

        # Replace the flat infectiousness set by the base with a density-derived
        # value. _gametocyte_to_infectiousness is a placeholder.
        alive          = self.sim.people.alive
        infected_alive = self.infected & alive
        if infected_alive.any():
            uids = ss.uids(infected_alive)
            self.infectiousness[uids] = self._gametocyte_to_infectiousness(uids)
        return

    # ------------------------------------------------------------------
    # Symptom classification (override)
    # ------------------------------------------------------------------
    def _update_observables(self):
        """
        Determine infeciousness from gametocyte density, calcualte hrp2_pg,
        and determine fever from parasite density and immune response.
        _update_system_state() should determine symptom_state from these
        observables.
        """
        super()._update_observables()
        return

    def _update_symptom_state(self):
        """Classify symptoms from parasite density (placeholder thresholds)."""
        alive         = self.sim.people.alive
        infected_uids = ss.uids(self.infected & alive)
        if len(infected_uids) == 0:
            return

        density = self.parasite_density[infected_uids]

        # Default: asymptomatic
        self.symptom_state[infected_uids] = SYMPTOM_NONE

        clinical_mask = density >= _PARASITE_DENSITY_SYMPTOMATIC
        self.symptom_state[infected_uids[clinical_mask]] = SYMPTOM_CLINICAL

        severe_mask = density >= _PARASITE_DENSITY_SEVERE
        self.symptom_state[infected_uids[severe_mask]] = SYMPTOM_SEVERE

        # SYMPTOM_DISEASE_DEATH left for a dedicated case-fatality model;
        # set here when parasite density exceeds a lethal threshold.
        return

    # ------------------------------------------------------------------
    # Death cleanup
    # ------------------------------------------------------------------

    def step_die(self, uids):
        super().step_die(uids)
        self.parasite_density[uids]   = 0.0
        self.gametocyte_density[uids] = 0.0
        # Immune response intentionally NOT reset — immunity persists after death
        # only matters if the agent is somehow reused, which Starsim does not do.
        self.immune_response[uids]    = 0.0
        return

    # ------------------------------------------------------------------
    # Within-host placeholder methods — replace with real biology
    # ------------------------------------------------------------------

    def _update_parasite_density(self, uids):
        """
        Update asexual parasite density for infected agents.

        Placeholder — currently a no-op. Implement with e.g.:
          - Simple exponential growth then immune-mediated clearance
          - Lotka-Volterra parasite–immune ODE discretised at sim.dt
          - Empirical growth/decay curves fitted to patient data
        """
        pass

    def _update_gametocyte_density(self, uids):
        """
        Update gametocyte density from asexual parasite density.

        Placeholder — currently a no-op. Typical approach:
          gametocyte_density[i] += conversion_fraction * parasite_density[i]
          gametocyte_density[i] *= (1 - gametocyte_clearance_rate * dt)
        """
        pass

    def _update_immune_response(self, uids):
        """
        Update immunity based on cumulative parasite exposure.

        Placeholder — currently a no-op. Typical approach:
          immune_response[i] += acquisition_rate * parasite_density[i] * dt
          immune_response[i] -= decay_rate * immune_response[i] * dt
          immune_response[i]  = clip(immune_response[i], 0, 1)
        """
        pass

    def _check_recovery(self, uids):
        """
        Transition agents to susceptible when parasite density clears.

        Called after density update so the clearance check uses the
        end-of-step density.
        """
        cleared = self.parasite_density[uids] < _PARASITE_DENSITY_CLEARANCE
        if cleared.any():
            cleared_uids = uids[cleared]
            self.infected[cleared_uids]         = False
            self.susceptible[cleared_uids]      = True
            self.parasite_density[cleared_uids] = 0.0
        return

    def _gametocyte_to_infectiousness(self, uids):
        """
        Convert gametocyte density to a 0–1 infectiousness value.

        Placeholder — returns zeros until real biology is implemented.
        Typical approach: saturating Hill function,
          f(G) = G^n / (G^n + K^n),  n~1, K~20 gametocytes/µL
        """
        return np.zeros(len(uids), dtype=float)
