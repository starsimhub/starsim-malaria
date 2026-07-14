"""
bednet.py — Bednet product and BednetUpdater connector.

Bednet
------
A ``ss.Product`` with four ``WaningEffect`` components: repelling, blocking,
killing, and usage.  Usage scales all three efficacy effects.

Human-side effect (indoor bites only):
    _prob_acquire_indoor *= (1 - usage*repelling) * (1 - usage*blocking)

Mosquito-side effects written to NodeState each step:
    itn_biting_reduction[node]  — fraction of bite attempts blocked
                                  = node-mean(usage * (1-(1-eff_rep)(1-eff_block)))
    itn_kill_prob[node]         — kill probability per bite attempt (dimensionless)
                                  = node-mean(usage * (1-repelling) * blocking * killing)
                                  (killing conditioned on not repelled and on blocking)

Both NodeState fields are independent of mosquito-module parameters. The
mosquito module multiplies itn_kill_prob by its own a*dt to get the per-step
kill fraction.

BednetUpdater
-------------
Connector that runs after WaningEffectUpdater (efficacy values are current)
and before the mosquito connector (so NodeState ITN fields are ready).

Execution order:
    WaningEffectUpdater → BednetUpdater → NodeState → MosquitoConnector
"""

import numpy as np
import starsim as ss

from malariasim.updater import Updater, Target
from malariasim.updaters.waning import WaningEffect, WaningEffectUpdater
from malariasim.human_base import BiteTarget

# ---------------------------------------------------------------------------
# Prefix constants — used by both Bednet and BednetUpdater
# ---------------------------------------------------------------------------

PREFIX_REPELLING = 'itn_repelling'
PREFIX_BLOCKING  = 'itn_blocking'
PREFIX_KILLING   = 'itn_killing'
PREFIX_USAGE     = 'itn_usage'
ALL_PREFIXES     = (PREFIX_REPELLING, PREFIX_BLOCKING, PREFIX_KILLING, PREFIX_USAGE)


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class Bednet(ss.Product):
    """
    Insecticide-treated bed net product with four independently waning effects.

    Parameters
    ----------
    repelling : WaningEffect, optional
        Probability a mosquito is repelled from entering the house.
        Default: permanent full efficacy.
    blocking : WaningEffect, optional
        Probability an entering mosquito is blocked from reaching the person.
        Default: permanent full efficacy.
    killing : WaningEffect, optional
        Probability a blocked mosquito is killed by insecticide contact.
        Killing is conditioned on not being repelled and on blocking — a repelled mosquito cannot be killed.
        Default: permanent full efficacy.
    usage : WaningEffect, optional
        Whether the net is in use. Scales all three efficacy effects.
        Default: permanent full efficacy.

    Examples
    --------
    LLIN with realistic waning::

        bednet = Bednet(
            repelling = WaningEffect(initial_efficacy=0.5,  box_duration=ss.days(0),   decay_time_constant=ss.days(730)),
            blocking  = WaningEffect(initial_efficacy=0.9,  box_duration=ss.days(365), decay_time_constant=ss.days(365)),
            killing   = WaningEffect(initial_efficacy=0.6,  box_duration=ss.days(180), decay_time_constant=ss.days(365)),
            usage     = WaningEffect(initial_efficacy=0.8,  box_duration=ss.days(365), decay_time_constant=ss.days(365)),
        )
    """

    def __init__(self, repelling=None, blocking=None, killing=None, usage=None, name=None):
        super().__init__(name=name)
        self.repelling = repelling if repelling is not None else WaningEffect()
        self.blocking  = blocking  if blocking  is not None else WaningEffect()
        self.killing   = killing   if killing   is not None else WaningEffect()
        self.usage     = usage     if usage     is not None else WaningEffect()
        wu = Updater.get(WaningEffectUpdater._updater_key)
        for prefix in ALL_PREFIXES:
            wu.register(prefix, Target.DISEASE)

    def init_pre(self, sim):
        super().init_pre(sim)
        self._disease = next(
            (m for m in self.sim.diseases.values()
             if hasattr(m, '_supports_intervention_effects')),
            None,
        )
        if self._disease is None:
            raise RuntimeError(
                'Bednet: no disease module with _supports_intervention_effects found. '
                'Ensure MalariaHumanBase (or a subclass) is registered.'
            )
        self._waning_updater = next(
            (m for m in self.sim.connectors.values() if isinstance(m, WaningEffectUpdater)),
            None,
        )
        if self._waning_updater is None:
            raise RuntimeError('Bednet: no WaningEffectUpdater found in sim.connectors.')
        self._bednet_updater = next(
            (m for m in self.sim.connectors.values() if isinstance(m, BednetUpdater)),
            None,
        )

    def administer(self, uids, weights=None):
        """
        Distribute a bednet to agents at ``uids``.

        Writes the four waning-effect state arrays and immediately applies the
        bednet effect for the current timestep (BednetUpdater already ran this
        step before the distributor fires).
        """
        if weights is None:
            weights = np.ones(len(uids), dtype=float)
        self._disease.store_itn_fraction(uids, weights)
        t_eff = np.full(len(uids), self.sim.t.year, dtype=float)
        self._waning_updater.apply(PREFIX_REPELLING, uids, self.repelling, t_eff)
        self._waning_updater.apply(PREFIX_BLOCKING,  uids, self.blocking,  t_eff)
        self._waning_updater.apply(PREFIX_KILLING,   uids, self.killing,   t_eff)
        self._waning_updater.apply(PREFIX_USAGE,     uids, self.usage,     t_eff)
        if self._bednet_updater is not None:
            self._bednet_updater.apply_immediate(uids)


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class BednetUpdater(Updater):
    """
    Connector that applies bednet efficacy each timestep.

    Runs after WaningEffectUpdater (so efficacy values are current) and
    before the mosquito connector (so NodeState ITN fields are ready).
    """

    _updater_key = 'bednet'

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force)
        self._disease = next(
            (m for m in self.sim.diseases.values()
             if hasattr(m, '_supports_intervention_effects')),
            None,
        )
        self._node_state = next(
            (m for m in self.sim.modules if getattr(m, '_is_node_state', False)),
            None,
        )
        self._waning_updater = next(
            (m for m in self.sim.connectors.values() if isinstance(m, WaningEffectUpdater)),
            None,
        )

    def step(self):
        """Apply bednet effects to all currently-dosed agents."""
        if self._disease is None or self._node_state is None or self._waning_updater is None:
            return
        if PREFIX_USAGE not in self._waning_updater._entries:
            return
        active = self._waning_updater.active_mask(PREFIX_USAGE)
        if not len(active.uids):
            return
        self.apply_immediate(active.uids)

    def apply_immediate(self, uids):
        """
        Write _prob_acquire_indoor and NodeState ITN fields for agents at ``uids``.

        Called both from step() each timestep and from Bednet.administer() on
        the distribution timestep (BednetUpdater already ran before the
        distributor fires that step).

        NodeState values are pure human-coverage aggregates — no mosquito
        parameters are needed here.  The mosquito module applies its own
        biting rate when consuming itn_kill_prob.
        """
        uids = ss.uids(uids) if not isinstance(uids, ss.uids) else uids
        if not len(uids):
            return

        wu = self._waning_updater
        eff_rep   = np.asarray(wu.current_efficacy(PREFIX_REPELLING)[uids])
        eff_block = np.asarray(wu.current_efficacy(PREFIX_BLOCKING)[uids])
        eff_kill  = np.asarray(wu.current_efficacy(PREFIX_KILLING)[uids])
        eff_usage = np.asarray(wu.current_efficacy(PREFIX_USAGE)[uids])

        u_rep   = eff_usage * eff_rep
        u_block = eff_usage * eff_block
        u_kill  = eff_usage * (1.0 - eff_rep) * eff_block * eff_kill  # killing conditioned on not repelled and on blocking

        # Combined fraction of indoor bites blocked by repelling or blocking
        combined_block = 1.0 - (1.0 - u_rep) * (1.0 - u_block)

        # Human-side: bednet reduces indoor acquisition probability only.
        # get_itn_weights returns coverage fractions (1.0 per ABM agent; _itn_fraction per ODE compartment).
        itn_weights = self._disease.get_itn_weights(uids)
        self._disease.set_acquisition_probability(uids, combined_block, itn_weights, target=BiteTarget.INDOOR)

        # Mosquito-side: aggregate to per-node NodeState fields via disease method.
        # ODE override population-weights by compartment size; ABM counts one per agent.
        self._disease.aggregate_itn_to_node_state(uids, combined_block, u_kill, self._node_state)
