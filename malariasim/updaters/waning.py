"""
waning.py — WaningEffect data class and WaningEffectUpdater connector.

WaningEffect
------------
A pure data class holding the three parameters that describe a box +
exponential-decay waning profile.  Users create this object and pass it to a
Product (e.g. ``MalariaVaccine``).  The Product is responsible for
registering the waning prefix with ``WaningEffectUpdater`` via
``Updater.get('waning').register(prefix, target)``.

WaningEffectUpdater
-------------------
A Starsim connector that owns all waning state arrays and exposes three
operations to other objects:

``register(prefix, target)``
    Called by a Product during ``build_campaign()`` to declare that it needs
    waning state arrays for the given prefix.

``apply(prefix, uids, waning_effect, t_yrs)``
    Called by a Product's ``administer()`` to write state arrays for newly-
    affected agents.  Returns the initial efficacy so the caller can apply
    any immediate effect (e.g. reducing ``prob_acquire``).

``current_efficacy(prefix)`` / ``active_mask(prefix)``
    Called by companion Updaters (e.g. ``MalariaVaccineUpdater``) each step to
    read the current efficacy and identify affected agents without knowing the
    underlying array names.

Per-person state arrays (four per prefix, on the disease module or NodeState)
-----------------------------------------------------------------------------
    {prefix}_efficacy0   — initial efficacy stored at time of application
    {prefix}_efficacy    — current efficacy, updated each step
    {prefix}_box_end     — sim time (years) when box phase ends
    {prefix}_decay_end   — sim time (years) when efficacy falls below 1 %
                           of initial.  Sentinel value -1 means unaffected.

Sentinel guard
--------------
``decay_end > 0`` is False for unaffected agents (sentinel = -1), so the
per-step loop skips them with a single comparison and no extra branching.
A second application simply overwrites the four values — the Updater picks
it up automatically on the next step.
"""

import numpy as np
import starsim as ss

from malariasim.updater import Updater, Target

# Efficacy is considered expired when it falls to this fraction of
# initial_efficacy.  Pre-computing decay_end at application time avoids
# floating-point work in the per-step loop.
EXPIRY_THRESHOLD    = 0.01                           # 1 % of initial efficacy
_LOG_EXPIRY_INVERSE = np.log(1.0 / EXPIRY_THRESHOLD) # ln(100) ≈ 4.605


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class WaningEffect:
    """
    Box + exponential-decay waning profile.

    Pure data class.  Create this object and pass it to a Product; the
    Product registers the waning prefix with ``WaningEffectUpdater`` and
    calls ``WaningEffectUpdater.apply()`` at dose time.

    Parameters
    ----------
    initial_efficacy : float, optional
        Efficacy in [0, 1] immediately after the intervention is applied.
        Defaults to 1.0 (full efficacy).
    box_duration : ss.days, optional
        Duration at full efficacy before exponential decay begins.  Use
        ``ss.days(0)`` for immediate decay from the time of application.
        Defaults to ``ss.days(np.inf)`` (permanent — never decays).
    decay_time_constant : ss.days, optional
        Exponential decay time constant τ.  Larger values mean slower decay.
        Use ``ss.days(0)`` for an instant drop to zero at the end of the box.
        Defaults to ``ss.days(np.inf)`` (permanent).  Must be infinite when
        ``box_duration`` is infinite, and finite when ``box_duration`` is
        finite.

    Examples
    --------
    Permanent full-efficacy effect (default)::

        WaningEffect()

    RTS,S-like vaccine profile::

        WaningEffect(
            initial_efficacy    = 0.80,
            box_duration        = ss.days(180),
            decay_time_constant = ss.days(730),
        )

    Immediate decay from application time::

        WaningEffect(
            initial_efficacy    = 0.60,
            box_duration        = ss.days(0),
            decay_time_constant = ss.days(365),
        )

    Constant efficacy for 100 days then instant drop to zero::

        WaningEffect(
            initial_efficacy    = 0.80,
            box_duration        = ss.days(100),
            decay_time_constant = ss.days(0),
        )
    """

    def __init__(self, initial_efficacy=1.0, box_duration=ss.days(np.inf),
                 decay_time_constant=ss.days(np.inf)):
        if not (0.0 <= initial_efficacy <= 1.0):
            raise ValueError(f'initial_efficacy must be in [0, 1], got {initial_efficacy}')
        if box_duration.days < 0:
            raise ValueError(f'box_duration must be >= 0, got {box_duration}')
        if decay_time_constant.days < 0:
            raise ValueError(
                f'decay_time_constant must be >= 0, got {decay_time_constant}'
            )

        box_infinite = np.isinf(box_duration.days)
        tau_infinite = np.isinf(decay_time_constant.days)

        if box_infinite and not tau_infinite:
            raise ValueError(
                'decay_time_constant must also be infinite when box_duration is infinite.'
            )
        if not box_infinite and tau_infinite:
            raise ValueError(
                'decay_time_constant must be finite when box_duration is finite.'
            )

        self.initial_efficacy    = float(initial_efficacy)
        self.box_duration        = box_duration         # ss.days
        self.decay_time_constant = decay_time_constant  # ss.days (τ)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def box_duration_yrs(self):
        """Box duration in simulation years."""
        return self.box_duration.years

    @property
    def tau_yrs(self):
        """Decay time constant τ in simulation years."""
        return self.decay_time_constant.years

    @property
    def decay_end_offset_yrs(self):
        """
        Years from box_end until efficacy falls below EXPIRY_THRESHOLD.

        Derived from:
            E0 * exp(-t_decay / τ) = E0 * EXPIRY_THRESHOLD
            t_decay = τ * ln(1 / EXPIRY_THRESHOLD)
        """
        return self.tau_yrs * _LOG_EXPIRY_INVERSE

    def make_effect_state(self, t_applied_yrs):
        """
        Compute per-person state values to store at the time of application.

        Called by ``WaningEffectUpdater.apply()`` — not by user code.

        Parameters
        ----------
        t_applied_yrs : float
            Current simulation time in years.

        Returns
        -------
        efficacy0 : float
        box_end   : float — sim time (years) when box ends
        decay_end : float — sim time (years) when efficacy reaches EXPIRY_THRESHOLD
        """
        box_end   = t_applied_yrs + self.box_duration_yrs
        decay_end = box_end + self.decay_end_offset_yrs
        return self.initial_efficacy, box_end, decay_end


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class _WaningEntry:
    """Per-prefix waning configuration resolved during init_pre()."""
    __slots__ = ('prefix', 'target', 'module')

    def __init__(self, prefix, target):
        self.prefix = prefix
        self.target = target
        self.module = None   # resolved in init_pre()


class WaningEffectUpdater(Updater):
    """
    Connector that owns all waning state arrays and updates them each step.

    Managed by ms.Sim — not instantiated directly by users.
    """

    _updater_key = 'waning'

    # ------------------------------------------------------------------
    # Registration (called during build_campaign())
    # ------------------------------------------------------------------

    def register(self, prefix, target=Target.DISEASE):
        """
        Declare that ``prefix`` needs waning state arrays on ``target``.

        Only the first registration per prefix is stored — a second call
        with the same prefix is silently ignored, enforcing the one-active-
        instance-per-type rule.

        Parameters
        ----------
        prefix : str
            Name prefix for the four per-person state arrays
            (e.g. ``'vaccine'``, ``'itn_blocking'``).
        target : Target
            Where state arrays live: ``Target.DISEASE`` (default) or
            ``Target.NODE_STATE``.
        """
        if prefix in self._entries:
            existing = self._entries[prefix].target
            if existing != target:
                raise ValueError(
                    f"WaningEffectUpdater: prefix {prefix!r} is already registered "
                    f"with target {existing}, cannot re-register with {target}."
                )
            return   # same prefix + same target: intentional sharing, no-op
        self._entries[prefix] = _WaningEntry(prefix, target)

    # ------------------------------------------------------------------
    # Starsim lifecycle
    # ------------------------------------------------------------------

    def init_pre(self, sim, force=False):
        """
        Resolve target modules and inject waning state arrays.

        Runs during Starsim's ``init_modules_pre()`` phase, which processes
        connectors **before** diseases.  Calling ``_inject_waning_states()``
        here means the per-person arrays are already attributes of the disease
        module by the time the disease's own ``init_pre()`` calls
        ``sim.people.add_module(self)``, so they are linked to People
        correctly.
        """
        super().init_pre(sim, force)
        for entry in self._entries.values():
            entry.module = self._resolve_target(entry.prefix, entry.target)
        self._inject_waning_states()

    def _resolve_target(self, prefix, target):
        """Find the module that should hold arrays for this prefix."""
        if target == Target.DISEASE:
            for mod in self.sim.diseases.values():
                if hasattr(mod, '_supports_intervention_effects'):
                    return mod
            raise RuntimeError(
                f"WaningEffectUpdater: no disease module with "
                f"'_supports_intervention_effects' found for prefix {prefix!r}. "
                f"Ensure MalariaHumanBase (or a subclass) is the human module."
            )
        if target == Target.NODE_STATE:
            for mod in self.sim.connectors.values():
                if hasattr(mod, '_is_node_state'):
                    return mod
            raise RuntimeError(
                f"WaningEffectUpdater: no NodeState connector found for "
                f"prefix {prefix!r}."
            )
        raise ValueError(f'Unknown Target: {target}')

    def _inject_waning_states(self):
        """
        Inject four per-person state arrays into each registered target module.

        Called from ``init_pre()`` after target modules are resolved and
        before they connect their states to People.  Sentinel value -1 means
        the agent has never received this effect.
        """
        for prefix, entry in self._entries.items():
            mod = entry.module
            mod.define_states(
                ss.FloatArr(f'{prefix}_efficacy0', default=0.0),
                ss.FloatArr(f'{prefix}_efficacy',  default=0.0),
                ss.FloatArr(f'{prefix}_box_end',   default=-1.0),
                ss.FloatArr(f'{prefix}_decay_end', default=-1.0),
            )

    # ------------------------------------------------------------------
    # Operations called by Products and companion Updaters
    # ------------------------------------------------------------------

    def apply(self, prefix, uids, waning_effect, t_yrs):
        """
        Write waning state arrays for newly-affected agents.

        Called by a Product's ``administer()`` method.  All waning
        implementation details (array names, sentinel values, expiry
        arithmetic) are hidden here.

        Parameters
        ----------
        prefix       : str                   — registered prefix (e.g. ``'vaccine'``)
        uids         : array-like            — UIDs of agents receiving the effect
        waning_effect : WaningEffect         — waning profile data
        t_yrs        : float or ndarray      — effective dose time in years.  ABM callers
                                               pass a scalar (``sim.t.year``); ODE callers
                                               pass a per-uid array returned by
                                               ``disease.pre_administer()``.  numpy
                                               broadcasting handles both cases.

        Returns
        -------
        float
            Initial efficacy, so the caller can apply an immediate effect
            (e.g. reducing ``prob_acquire``) without knowing the array names.
        """
        uids = ss.uids(uids) if not isinstance(uids, ss.uids) else uids
        efficacy0, box_end, decay_end = waning_effect.make_effect_state(t_yrs)
        mod = self._entries[prefix].module
        getattr(mod, f'{prefix}_efficacy0')[uids] = efficacy0
        getattr(mod, f'{prefix}_efficacy')[uids]  = efficacy0
        getattr(mod, f'{prefix}_box_end')[uids]   = box_end
        getattr(mod, f'{prefix}_decay_end')[uids] = decay_end
        return efficacy0

    def current_efficacy(self, prefix):
        """
        Return the current efficacy FloatArr for ``prefix``.

        Called by companion Updaters each step.  Callers index by UID via
        ``efficacy[uids]`` or combine with ``active_mask()`` via ``&``.
        """
        entry = self._entries[prefix]
        return getattr(entry.module, f'{prefix}_efficacy')

    def active_mask(self, prefix):
        """
        Return a BoolArr of agents with an active effect for ``prefix``.

        An agent is active when its ``decay_end > 0`` (sentinel -1 means
        never affected).  Use ``.uids`` on the result to get actual UIDs.
        """
        entry = self._entries[prefix]
        return getattr(entry.module, f'{prefix}_decay_end') > 0

    # ------------------------------------------------------------------
    # Per-step logic
    # ------------------------------------------------------------------

    def step(self):
        """Update waning efficacy for all affected agents (vectorised)."""
        t = self.sim.t.year

        for prefix, entry in self._entries.items():
            mod       = entry.module
            efficacy0 = getattr(mod, f'{prefix}_efficacy0')
            efficacy  = getattr(mod, f'{prefix}_efficacy')
            box_end   = getattr(mod, f'{prefix}_box_end')
            decay_end = getattr(mod, f'{prefix}_decay_end')
            self._apply_waning(efficacy0, efficacy, box_end, decay_end, t)

    # ------------------------------------------------------------------
    # Pure-numpy inner loop (static for testability)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_waning(efficacy0, efficacy, box_end, decay_end, t):
        """
        Update the efficacy FloatArr for one prefix at time ``t``.

        Parameters
        ----------
        efficacy0 : FloatArr  — initial efficacy stored at time of application
        efficacy  : FloatArr  — current efficacy state array (written via UIDs)
        box_end   : FloatArr  — sim time (years) when box ends
        decay_end : FloatArr  — sim time (years) of expiry  (sentinel: -1)
        t         : float     — current sim time in years

        Behaviour
        ---------
        Unaffected agents (decay_end <= 0): skipped by sentinel guard.
        In-box   (t <= box_end)           : efficacy unchanged.
        Decaying (box_end < t < decay_end): exponential decay.
        Expired  (t >= decay_end)         : efficacy set to 0.

        tau is reconstructed from the stored arrays as
        ``(decay_end - box_end) / ln(1 / EXPIRY_THRESHOLD)``.
        For instant-drop profiles (decay_end == box_end) the decaying mask
        is always False, so the division is never reached.
        """
        affected = decay_end > 0
        decaying = affected & (box_end < t) & (decay_end > t)
        expired  = affected & (decay_end <= t)

        if len(decaying.uids):
            e0    = efficacy0[decaying.uids]
            t_box = box_end[decaying.uids]
            t_dec = decay_end[decaying.uids]
            tau   = (t_dec - t_box) / _LOG_EXPIRY_INVERSE
            efficacy[decaying.uids] = e0 * np.exp(-(t - t_box) / tau)

        if len(expired.uids):
            efficacy[expired.uids] = 0.0
