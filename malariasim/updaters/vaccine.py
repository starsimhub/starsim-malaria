"""
vaccine.py — MalariaVaccine product and MalariaVaccineUpdater connector.

MalariaVaccine
--------------
A ``ss.Product`` subclass that applies an efficacy-waning vaccine to a set of
agents.  The waning profile is specified by a ``WaningEffect`` passed at
construction time.  One ``MalariaVaccine`` instance is shared across any number
of distribution events.

``MalariaVaccine.__init__`` calls ``Updater.get('waning').register(...)`` to
declare the ``'vaccine'`` prefix, so it must be instantiated inside the
``build_campaign()`` callback that ``ms.Sim`` activates.

MalariaVaccineUpdater
---------------------
A Starsim connector that runs once per timestep and applies the current vaccine
efficacy to each vaccinated agent's ``prob_acquire`` multiplier.

The human disease module is expected to reset ``prob_acquire`` to 1.0 at the start
of each step (see ``MalariaHumanBase.update_immunity()``).
``MalariaVaccineUpdater`` then reapplies the vaccine protection so it is in
place for the transmission calculation that follows.

Execution order
---------------
1. Human disease step     — resets prob_acquire = 1.0, reads EIR, runs transitions.
2. WaningEffectUpdater    — updates vaccine_efficacy for all vaccinated agents.
3. MalariaVaccineUpdater  — writes prob_acquire *= (1 − vaccine_efficacy) for all
                            vaccinated agents.
4. Mosquito connector     — reads p_inf from human module.
"""

import numpy as np
import starsim as ss

from malariasim.updater import Updater, Target
from malariasim.updaters.waning import WaningEffect, WaningEffectUpdater
from malariasim.human_base import BiteTarget


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class MalariaVaccine(ss.Product):
    """
    Malaria vaccine product with time-varying efficacy waning.

    Parameters
    ----------
    waning : WaningEffect, optional
        Waning profile describing how vaccine efficacy evolves after dosing.
        Defaults to ``WaningEffect()`` — permanent full efficacy.

    Examples
    --------
    RTS,S-like profile — 80 % efficacy, 6-month box, then exponential decay::

        vaccine = MalariaVaccine(
            waning=ms.WaningEffect(
                initial_efficacy    = 0.80,
                box_duration        = ss.days(180),
                decay_time_constant = ss.days(730),
            )
        )

    Constant 70 % efficacy, drops instantly after 1 year::

        vaccine = MalariaVaccine(
            waning=ms.WaningEffect(
                initial_efficacy    = 0.70,
                box_duration        = ss.days(365),
                decay_time_constant = ss.days(0),
            )
        )
    """

    PREFIX = 'vaccine'

    def __init__(self, waning=None, name=None):
        super().__init__(name=name)
        self.waning = waning if waning is not None else WaningEffect()
        Updater.get(WaningEffectUpdater._updater_key).register(self.PREFIX, Target.DISEASE)

    # ------------------------------------------------------------------
    # Starsim lifecycle
    # ------------------------------------------------------------------

    def init_pre(self, sim):
        """Resolve and store handles to the disease module and waning updater."""
        super().init_pre(sim)
        self._disease        = self._find_disease()
        self._waning_updater = self._find_waning_updater()

    def _find_disease(self):
        """Return the first disease module that supports intervention effects."""
        for mod in self.sim.diseases.values():
            if hasattr(mod, '_supports_intervention_effects'):
                return mod
        raise RuntimeError(
            'MalariaVaccine: no disease module with '
            "'_supports_intervention_effects' found. "
            'Ensure MalariaHumanBase (or a subclass) is registered as the '
            'human disease module.'
        )

    def _find_waning_updater(self):
        """Find the WaningEffectUpdater registered in sim.connectors.

        Uses sim.connectors rather than Updater._possible_updaters because
        starsim may copy module objects when building the sim — the live
        connector (the one whose init_pre() ran) is the only one with
        correctly resolved entry.module handles.
        """
        for mod in self.sim.connectors.values():
            if isinstance(mod, WaningEffectUpdater):
                return mod
        raise RuntimeError(
            'MalariaVaccine: no WaningEffectUpdater found in sim.connectors. '
            'Ensure ms.Sim received a build_campaign that includes a MalariaVaccine.'
        )

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def administer(self, uids, weights=None):
        """
        Vaccinate the agents at ``uids``.

        Delegates state-array writes to ``WaningEffectUpdater.apply()``, then
        immediately reduces ``prob_acquire`` for the current timestep using the
        returned initial efficacy.  On subsequent timesteps
        ``MalariaVaccineUpdater`` reapplies the protection after the disease
        module resets ``prob_acquire``.

        Parameters
        ----------
        uids : array-like
            UIDs of agents to vaccinate.
        weights : ndarray or None
            Per-uid coverage weights in [0, 1].  ABM callers may pass None
            (defaults to all-ones).  ODE callers pass fractional coverage
            so ``pre_administer`` can update ``_vaccinated_fraction``.
        """
        if weights is None:
            weights = np.ones(len(uids), dtype=float)
        t_eff    = self._disease.pre_administer(uids, weights, self.sim.t.year)
        efficacy = self._waning_updater.apply(self.PREFIX, uids, self.waning, t_eff)
        vax_weights = self._disease.get_vaccinated_weights(uids)
        self._disease.set_acquisition_probability(uids, efficacy, vax_weights, target=BiteTarget.ALL)


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class MalariaVaccineUpdater(Updater):
    """
    Connector that applies current vaccine efficacy to prob_acquire each timestep.

    Asks ``WaningEffectUpdater`` for the active mask and current efficacy by
    prefix, so it never accesses the underlying state-array names directly.
    """

    _updater_key = 'vaccine'

    # ------------------------------------------------------------------
    # Starsim lifecycle
    # ------------------------------------------------------------------

    def init_pre(self, sim):
        """Resolve handles to the disease module and waning updater."""
        super().init_pre(sim)
        self._disease = None
        for mod in self.sim.diseases.values():
            if hasattr(mod, '_supports_intervention_effects'):
                self._disease = mod
                break
        self._waning_updater = next(
            (m for m in self.sim.connectors.values() if isinstance(m, WaningEffectUpdater)),
            None,
        )

    # ------------------------------------------------------------------
    # Per-step logic
    # ------------------------------------------------------------------

    def step(self):
        """Apply current vaccine efficacy to prob_acquire each timestep."""
        if self._disease is None:
            return
        if MalariaVaccine.PREFIX not in self._waning_updater._entries:
            return
        ever_dosed = self._waning_updater.active_mask(MalariaVaccine.PREFIX)
        efficacy   = self._waning_updater.current_efficacy(MalariaVaccine.PREFIX)
        self._disease.sync_vaccinated_fraction(ever_dosed.uids, efficacy)
        active_uids = (ever_dosed & (efficacy > 0)).uids
        if len(active_uids):
            vax_weights = self._disease.get_vaccinated_weights(active_uids)
            self._disease.set_acquisition_probability(
                active_uids, efficacy[active_uids], vax_weights, target=BiteTarget.ALL
            )
