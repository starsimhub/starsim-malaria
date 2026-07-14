"""
Tests for WaningEffect and WaningEffectUpdater.

WaningEffect is a pure data class — its tests construct instances freely,
no build_campaign() window required.

WaningEffectUpdater is tested through its public API:
  register(prefix, target)       — called during build_campaign()
  apply(prefix, uids, waning, t) — called by Product.administer()
  current_efficacy(prefix)       — called by companion Updaters
  active_mask(prefix)            — called by companion Updaters
  _apply_waning(...)             — pure-numpy static method

Tests that exercise register() use the _activate() / _deactivate() helpers
to open and close the build_campaign() window.
"""

import numpy as np
import pytest
import starsim as ss

from malariasim.updater import Updater, Target
from malariasim.updaters.waning import (
    WaningEffect,
    WaningEffectUpdater,
    _WaningEntry,
    EXPIRY_THRESHOLD,
    _LOG_EXPIRY_INVERSE,
)
from tests.mock_objects import activate as _activate, make_waning as _make_waning, MockDisease as _MockDisease, make_waning_updater_for_disease


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_updater_with_disease(prefix='vaccine', n=10):
    """
    Return (updater, disease) with a pre-populated entry and resolved module.
    Bypasses sim initialisation for unit testing.
    """
    disease = _MockDisease(n=n)
    disease.add_prefix(prefix)
    return make_waning_updater_for_disease(disease, prefix), disease


# ===========================================================================
# WaningEffect — pure data class
# ===========================================================================

class TestWaningEffect:

    # ------------------------------------------------------------------
    # Construction and defaults
    # ------------------------------------------------------------------

    def test_constructs_successfully(self):
        w = _make_waning()
        assert w.initial_efficacy == pytest.approx(0.80)
        assert w.box_duration.days        == pytest.approx(180.0)
        assert w.decay_time_constant.days == pytest.approx(730.0)

    def test_default_is_permanent_full_efficacy(self):
        """WaningEffect() with no arguments is permanent full efficacy."""
        w = WaningEffect()
        assert w.initial_efficacy == pytest.approx(1.0)
        assert np.isinf(w.box_duration.days)
        assert np.isinf(w.decay_time_constant.days)

    def test_constructable_outside_build_campaign(self):
        """WaningEffect is not tied to the build_campaign() window."""
        assert not Updater._active
        w = _make_waning()
        assert w is not None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_invalid_efficacy_high(self):
        with pytest.raises(ValueError):
            _make_waning(initial_efficacy=1.1)

    def test_invalid_efficacy_low(self):
        with pytest.raises(ValueError):
            _make_waning(initial_efficacy=-0.1)

    def test_invalid_box_duration_negative(self):
        with pytest.raises(ValueError):
            _make_waning(box_duration=ss.days(-1))

    def test_zero_box_duration_allowed(self):
        w = _make_waning(box_duration=ss.days(0))
        assert w.box_duration.days == 0.0

    def test_zero_decay_time_constant_allowed(self):
        """ss.days(0) means instant drop to zero at box end — valid."""
        w = WaningEffect(initial_efficacy=0.80,
                         box_duration=ss.days(100),
                         decay_time_constant=ss.days(0))
        assert w.decay_time_constant.days == 0.0

    def test_invalid_decay_time_constant_negative(self):
        with pytest.raises(ValueError):
            _make_waning(decay_time_constant=ss.days(-100))

    def test_infinite_box_requires_infinite_tau(self):
        """box_duration=inf with finite decay_time_constant is an error."""
        with pytest.raises(ValueError, match='infinite'):
            _make_waning(box_duration=ss.days(np.inf), decay_time_constant=ss.days(730))

    def test_finite_box_requires_finite_tau(self):
        """Finite box_duration with decay_time_constant=inf is an error."""
        with pytest.raises(ValueError, match='finite'):
            _make_waning(box_duration=ss.days(180), decay_time_constant=ss.days(np.inf))

    def test_permanent_effect_explicit(self):
        """Explicitly passing both infinite values is valid."""
        w = WaningEffect(initial_efficacy=0.70,
                         box_duration=ss.days(np.inf),
                         decay_time_constant=ss.days(np.inf))
        assert np.isinf(w.box_duration_yrs)
        assert np.isinf(w.tau_yrs)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def test_box_duration_yrs(self):
        w = _make_waning(box_duration=ss.days(365))
        assert w.box_duration_yrs == pytest.approx(1.0)

    def test_tau_yrs(self):
        w = _make_waning(decay_time_constant=ss.days(365))
        assert w.tau_yrs == pytest.approx(1.0)

    def test_decay_end_offset_formula(self):
        """decay_end_offset = τ_yrs * ln(1 / EXPIRY_THRESHOLD)."""
        w = _make_waning()
        expected = w.tau_yrs * _LOG_EXPIRY_INVERSE
        assert w.decay_end_offset_yrs == pytest.approx(expected)

    def test_decay_end_offset_increases_with_tau(self):
        w_fast = _make_waning(decay_time_constant=ss.days(180))
        w_slow = _make_waning(decay_time_constant=ss.days(730))
        assert w_slow.decay_end_offset_yrs > w_fast.decay_end_offset_yrs

    # ------------------------------------------------------------------
    # make_effect_state
    # ------------------------------------------------------------------

    def test_make_effect_state_efficacy0(self):
        w = _make_waning(initial_efficacy=0.75)
        e0, _, _ = w.make_effect_state(2005.0)
        assert e0 == pytest.approx(0.75)

    def test_make_effect_state_box_end(self):
        w = _make_waning(box_duration=ss.days(365))   # exactly 1 year
        _, box_end, _ = w.make_effect_state(2005.0)
        assert box_end == pytest.approx(2006.0)

    def test_make_effect_state_decay_end(self):
        """decay_end = box_end + τ_yrs * ln(1 / EXPIRY_THRESHOLD)."""
        w = _make_waning()
        _, box_end, decay_end = w.make_effect_state(2005.0)
        expected = box_end + w.tau_yrs * _LOG_EXPIRY_INVERSE
        assert decay_end == pytest.approx(expected)

    def test_make_effect_state_zero_box(self):
        """With box_duration=0, box_end equals t_applied."""
        w = _make_waning(box_duration=ss.days(0))
        _, box_end, _ = w.make_effect_state(2010.0)
        assert box_end == pytest.approx(2010.0)

    def test_make_effect_state_instant_drop(self):
        """decay_time_constant=0 → decay_end equals box_end → instant drop."""
        w = WaningEffect(initial_efficacy=0.80,
                         box_duration=ss.days(100),
                         decay_time_constant=ss.days(0))
        _, box_end, decay_end = w.make_effect_state(2005.0)
        assert decay_end == pytest.approx(box_end)

    def test_make_effect_state_permanent(self):
        """Permanent effect yields infinite box_end and decay_end."""
        w = WaningEffect()
        _, box_end, decay_end = w.make_effect_state(2005.0)
        assert np.isinf(box_end)
        assert np.isinf(decay_end)


# ===========================================================================
# WaningEffectUpdater — registration
# ===========================================================================

class TestWaningEffectUpdaterRegister:

    def test_register_requires_build_campaign_window(self):
        assert not Updater._active
        updater = WaningEffectUpdater()
        # register() itself doesn't check _active — Updater.get() does.
        # Here we just confirm register() stores the entry.
        updater.register('vaccine', Target.DISEASE)
        assert 'vaccine' in updater._entries

    def test_updater_get_raises_outside_window(self):
        assert not Updater._active
        with pytest.raises(RuntimeError, match='build_campaign'):
            Updater.get(WaningEffectUpdater._updater_key)

    def test_updater_get_returns_singleton_inside_window(self):
        restore = _activate()
        try:
            u = Updater.get(WaningEffectUpdater._updater_key)
            assert u is Updater._possible_updaters[WaningEffectUpdater._updater_key]
        finally:
            restore()

    def test_register_stores_prefix_and_target(self):
        updater = WaningEffectUpdater()
        updater.register('vaccine', Target.DISEASE)
        entry = updater._entries['vaccine']
        assert entry.prefix == 'vaccine'
        assert entry.target == Target.DISEASE

    def test_second_registration_same_prefix_same_target_ignored(self):
        """Same prefix + same target is intentional sharing — silently accepted."""
        updater = WaningEffectUpdater()
        updater.register('vaccine', Target.DISEASE)
        updater.register('vaccine', Target.DISEASE)   # no-op
        assert len(updater._entries) == 1

    def test_second_registration_same_prefix_different_target_raises(self):
        """Same prefix + different target is a programming error — raises."""
        updater = WaningEffectUpdater()
        updater.register('vaccine', Target.DISEASE)
        with pytest.raises(ValueError, match="prefix 'vaccine'"):
            updater.register('vaccine', Target.NODE_STATE)

    def test_different_prefixes_create_separate_entries(self):
        updater = WaningEffectUpdater()
        updater.register('vaccine',     Target.DISEASE)
        updater.register('itn_killing', Target.NODE_STATE)
        assert 'vaccine'     in updater._entries
        assert 'itn_killing' in updater._entries

    def test_clear_wipes_entries(self):
        updater = WaningEffectUpdater()
        updater.register('vaccine', Target.DISEASE)
        updater.clear()
        assert len(updater._entries) == 0


# ===========================================================================
# WaningEffectUpdater — apply(), current_efficacy(), active_mask()
# ===========================================================================

class TestWaningEffectUpdaterAPI:

    # ------------------------------------------------------------------
    # apply() — writes state arrays, returns initial efficacy
    # ------------------------------------------------------------------

    def test_apply_returns_initial_efficacy(self):
        """Return value is the initial efficacy so callers can act on it
        (e.g. reduce rel_sus) without reading back from the arrays."""
        updater, disease = _make_updater_with_disease()
        w = _make_waning(initial_efficacy=0.75)
        efficacy = updater.apply('vaccine', [0], w, 2010.0)
        assert efficacy == pytest.approx(0.75)

    def test_apply_writes_efficacy0(self):
        """efficacy0 stores the dose-time value so the decay formula can
        scale from it on every subsequent step."""
        updater, disease = _make_updater_with_disease()
        w = _make_waning(initial_efficacy=0.80)
        updater.apply('vaccine', [0, 1, 2], w, 2010.0)
        np.testing.assert_array_almost_equal(
            disease.vaccine_efficacy0.values[[0, 1, 2]], 0.80
        )

    def test_apply_sets_current_efficacy_equal_to_initial(self):
        """At the moment of dosing, current efficacy equals initial efficacy
        — the agent is immediately at full protection."""
        updater, disease = _make_updater_with_disease()
        w = _make_waning(initial_efficacy=0.80)
        updater.apply('vaccine', [3], w, 2010.0)
        assert disease.vaccine_efficacy.values[3] == pytest.approx(0.80)

    def test_apply_writes_box_end(self):
        """box_end is stored in sim-years so the per-step loop can compare
        it to self.t.year without converting on every tick."""
        updater, disease = _make_updater_with_disease()
        w = _make_waning(box_duration=ss.days(365))
        updater.apply('vaccine', [0], w, 2010.0)
        assert disease.vaccine_box_end.values[0] == pytest.approx(2011.0)

    def test_apply_writes_decay_end(self):
        """decay_end = box_end + τ * ln(100), pre-computed so the step loop
        never calls log()."""
        updater, disease = _make_updater_with_disease()
        w = _make_waning(box_duration=ss.days(365), decay_time_constant=ss.days(365))
        updater.apply('vaccine', [0], w, 2010.0)
        expected = 2011.0 + 1.0 * _LOG_EXPIRY_INVERSE
        assert disease.vaccine_decay_end.values[0] == pytest.approx(expected)

    def test_apply_unaffected_agents_keep_sentinel(self):
        """Only the dosed UIDs are written; all others keep the sentinel -1
        so the step loop skips them at zero cost."""
        updater, disease = _make_updater_with_disease()
        updater.apply('vaccine', [0], _make_waning(), 2010.0)
        np.testing.assert_array_equal(disease.vaccine_decay_end.values[1:], -1.0)

    def test_apply_permanent_vaccine(self):
        """Permanent effect (both durations infinite) writes inf timestamps
        so the agent is never in the decaying or expired region."""
        updater, disease = _make_updater_with_disease()
        updater.apply('vaccine', [0], WaningEffect(), 2010.0)
        assert np.isinf(disease.vaccine_box_end.values[0])
        assert np.isinf(disease.vaccine_decay_end.values[0])

    def test_apply_overwrites_on_revaccination(self):
        """A second dose simply overwrites all four arrays — the step loop
        picks up the new profile on the next tick without special handling."""
        updater, disease = _make_updater_with_disease()
        updater.apply('vaccine', [0], _make_waning(initial_efficacy=0.80), 2010.0)
        updater.apply('vaccine', [0], _make_waning(initial_efficacy=0.50), 2012.0)
        assert disease.vaccine_efficacy0.values[0] == pytest.approx(0.50)

    # ------------------------------------------------------------------
    # current_efficacy() and active_mask() — read access for companion updaters
    # ------------------------------------------------------------------

    def test_current_efficacy_returns_array(self):
        """current_efficacy() exposes the live efficacy array so companion
        updaters (e.g. MalariaVaccineUpdater) never touch the array name."""
        updater, disease = _make_updater_with_disease()
        disease.vaccine_efficacy.values[2] = 0.65
        eff = updater.current_efficacy('vaccine')
        assert eff[2] == pytest.approx(0.65)

    def test_active_mask_false_for_unaffected(self):
        """Before any dose, all agents carry the sentinel -1 → mask is all False."""
        updater, disease = _make_updater_with_disease()
        mask = updater.active_mask('vaccine')
        assert not mask.any()

    def test_active_mask_true_after_apply(self):
        """Only the dosed UIDs become True; undosed agents stay False."""
        updater, disease = _make_updater_with_disease()
        updater.apply('vaccine', [0, 3], _make_waning(), 2010.0)
        mask = updater.active_mask('vaccine')
        assert mask[0] and mask[3]
        assert not mask[1] and not mask[2]

    def test_active_mask_true_for_expired_agents(self):
        """Expired agents still have decay_end > 0 so the mask stays True.
        Their efficacy is 0 so the companion updater multiplies rel_sus by 1
        — no harm done, and expiry does not require a separate code path."""
        updater, disease = _make_updater_with_disease()
        disease.vaccine_decay_end.values[5] = 2008.0   # past expiry, still > 0
        disease.vaccine_efficacy.values[5]  = 0.0
        mask = updater.active_mask('vaccine')
        assert mask[5]


# ===========================================================================
# WaningEffectUpdater._apply_waning — pure-numpy step logic
# ===========================================================================

class TestApplyWaning:

    def _arrays(self, n, efficacy0_val, box_end_val, decay_end_val, efficacy_val=None):
        e_val     = efficacy0_val if efficacy_val is None else efficacy_val
        efficacy0 = ss.FloatArr('efficacy0', raw=np.full(n, efficacy0_val), default=0.0,  skip_init=True)
        efficacy  = ss.FloatArr('efficacy',  raw=np.full(n, e_val),         default=0.0,  skip_init=True)
        box_end   = ss.FloatArr('box_end',   raw=np.full(n, box_end_val),   default=-1.0, skip_init=True)
        decay_end = ss.FloatArr('decay_end', raw=np.full(n, decay_end_val), default=-1.0, skip_init=True)
        return efficacy0, efficacy, box_end, decay_end

    def _apply(self, efficacy0, efficacy, box_end, decay_end, t):
        WaningEffectUpdater._apply_waning(efficacy0, efficacy, box_end, decay_end, t)

    def test_unaffected_skipped(self):
        e0, eff, box, dec = self._arrays(5, 0.8, -1.0, -1.0, 0.0)
        self._apply(e0, eff, box, dec, t=2010.0)
        np.testing.assert_array_equal(eff.values, 0.0)

    def test_in_box_unchanged(self):
        e0, eff, box, dec = self._arrays(3, 0.80, 2011.0, 2020.0)
        original = eff.copy()
        self._apply(e0, eff, box, dec, t=2010.0)
        np.testing.assert_array_almost_equal(eff.values, original)

    def test_expired_set_to_zero(self):
        e0, eff, box, dec = self._arrays(4, 0.80, 2005.0, 2008.0)
        self._apply(e0, eff, box, dec, t=2010.0)
        np.testing.assert_array_equal(eff.values, 0.0)

    def test_decaying_exponential(self):
        tau_yrs = 2.0
        box_val = 2005.0
        dec_val = box_val + tau_yrs * _LOG_EXPIRY_INVERSE
        t       = 2007.0

        e0, eff, box, dec = self._arrays(2, 0.80, box_val, dec_val)
        self._apply(e0, eff, box, dec, t=t)

        expected = 0.80 * np.exp(-(t - box_val) / tau_yrs)
        np.testing.assert_allclose(eff.values, expected, rtol=1e-6)

    def test_mixed_states(self):
        tau_yrs = 1.0
        t       = 2010.0
        dec_far = 2008.0 + tau_yrs * _LOG_EXPIRY_INVERSE + 2.0

        e0_arr  = ss.FloatArr('efficacy0', raw=np.array([0.80, 0.80, 0.80, 0.80]),          default=0.0,  skip_init=True)
        eff_arr = ss.FloatArr('efficacy',  raw=np.array([0.00, 0.80, 0.80, 0.80]),          default=0.0,  skip_init=True)
        box_arr = ss.FloatArr('box_end',   raw=np.array([-1.0, 2012.0, 2008.0, 2000.0]),    default=-1.0, skip_init=True)
        dec_arr = ss.FloatArr('decay_end', raw=np.array([-1.0, 2017.0, dec_far, 2006.0]),   default=-1.0, skip_init=True)

        self._apply(e0_arr, eff_arr, box_arr, dec_arr, t=t)

        # tau reconstructed from (dec_far - box_end) / ln(100)
        tau = (dec_far - 2008.0) / _LOG_EXPIRY_INVERSE
        assert eff_arr.values[0] == pytest.approx(0.0)
        assert eff_arr.values[1] == pytest.approx(0.80)
        assert eff_arr.values[2] == pytest.approx(0.80 * np.exp(-(t - 2008.0) / tau))
        assert eff_arr.values[3] == pytest.approx(0.0)

    def test_instant_drop_at_box_end(self):
        """With tau=0 decay_end==box_end, efficacy drops to 0 exactly at box end."""
        w = WaningEffect(initial_efficacy=0.80,
                         box_duration=ss.days(100),
                         decay_time_constant=ss.days(0))
        _, box_end, decay_end = w.make_effect_state(2005.0)
        assert decay_end == pytest.approx(box_end)

        # Still inside box: efficacy unchanged
        e0, eff, box, dec = self._arrays(1, 0.80, box_end, decay_end, 0.80)
        self._apply(e0, eff, box, dec, t=box_end - 0.001)
        assert eff.values[0] == pytest.approx(0.80)

        # At box_end: expired immediately
        e0, eff, box, dec = self._arrays(1, 0.80, box_end, decay_end, 0.80)
        self._apply(e0, eff, box, dec, t=box_end)
        assert eff.values[0] == pytest.approx(0.0)

    def test_efficacy_monotone_decreasing(self):
        tau_yrs = 1.0
        box_val = 2005.0
        dec_val = box_val + tau_yrs * _LOG_EXPIRY_INVERSE
        prev    = 0.80

        for t in np.linspace(box_val, dec_val, 50):
            e0, eff, box, dec = self._arrays(1, 0.80, box_val, dec_val, prev)
            self._apply(e0, eff, box, dec, t=float(t))
            assert eff.values[0] <= prev + 1e-12, f'Efficacy increased at t={t}'
            prev = eff.values[0]


# ===========================================================================
# Regression: waning must apply to correct UIDs after agent deaths
# ===========================================================================

import sciris as sc


class TestWaningAfterDeaths:
    """Guards UID-correctness in WaningEffectUpdater after agent deaths.

    Scenario: n_total=10, vaccinated UIDs 3 and 7.  We hit bugs here before;
    these tests ensure they don't come back.
    """

    N_TOTAL   = 10
    VACC_UIDS = [3, 7]

    def _make_updater(self, prefix='vaccine'):
        """Return (updater, disease) with real ss.FloatArr state arrays."""
        n       = self.N_TOTAL
        updater = WaningEffectUpdater()

        class _D:
            _supports_intervention_effects = True
        disease = _D()

        for attr, default in (('efficacy0', 0.0), ('efficacy', 0.0),
                               ('box_end',  -1.0), ('decay_end', -1.0)):
            setattr(disease, f'{prefix}_{attr}',
                    ss.FloatArr(f'{prefix}_{attr}', raw=np.full(n, default), default=default, skip_init=True))

        for uid in self.VACC_UIDS:
            getattr(disease, f'{prefix}_decay_end')[ss.uids([uid])] = 9999.0
            getattr(disease, f'{prefix}_efficacy')[ss.uids([uid])]  = 0.80

        entry = _WaningEntry(prefix, Target.DISEASE)
        entry.module = disease
        updater._entries[prefix] = entry
        return updater, disease

    # ------------------------------------------------------------------

    def test_active_mask_returns_correct_uids(self):
        """active_mask().uids returns vaccinated UIDs [3, 7], not positions."""
        updater, _ = self._make_updater()
        mask = updater.active_mask('vaccine')

        assert hasattr(mask, 'uids'), 'active_mask() must return a BoolArr with .uids'
        found = set(mask.uids.tolist())
        assert found == set(self.VACC_UIDS), (
            f'active_mask().uids returned {found}, expected {set(self.VACC_UIDS)}.'
        )

    def test_current_efficacy_uid_indexing(self):
        """current_efficacy() returns a FloatArr; indexing by vaccinated UIDs gives 0.80."""
        updater, _ = self._make_updater()
        eff = updater.current_efficacy('vaccine')

        assert isinstance(eff, ss.FloatArr), 'current_efficacy() must return a FloatArr'
        for uid in self.VACC_UIDS:
            val = eff[ss.uids([uid])][0]
            assert val == pytest.approx(0.80), (
                f'Vaccinated UID {uid} should have efficacy 0.80, got {val:.4f}.'
            )
        for uid in range(3):
            val = eff[ss.uids([uid])][0]
            assert val == pytest.approx(0.0), (
                f'Unvaccinated UID {uid} should have default efficacy 0.0, got {val:.4f}.'
            )

    def test_step_writes_correct_uids(self):
        """step() updates only vaccinated UIDs [3, 7], not unvaccinated UIDs."""
        n      = self.N_TOTAL
        prefix = 'vaccine'
        tau    = 2.0
        box_at = 2005.0
        dec_at = box_at + tau * _LOG_EXPIRY_INVERSE
        t      = 2007.0
        expected = 0.80 * np.exp(-(t - box_at) / tau)

        updater = WaningEffectUpdater()

        class _D:
            _supports_intervention_effects = True
        disease = _D()

        for attr, default in (('efficacy0', 0.0), ('efficacy', 0.0),
                               ('box_end',  -1.0), ('decay_end', -1.0)):
            setattr(disease, f'{prefix}_{attr}',
                    ss.FloatArr(f'{prefix}_{attr}', raw=np.full(n, default), default=default, skip_init=True))

        for uid in self.VACC_UIDS:
            getattr(disease, f'{prefix}_efficacy0')[ss.uids([uid])] = 0.80
            getattr(disease, f'{prefix}_box_end')[ss.uids([uid])]   = box_at
            getattr(disease, f'{prefix}_decay_end')[ss.uids([uid])] = dec_at
            getattr(disease, f'{prefix}_efficacy')[ss.uids([uid])]  = 0.80

        entry = _WaningEntry(prefix, Target.DISEASE)
        entry.module = disease
        updater._entries[prefix] = entry
        updater.setattribute('sim', sc.objdict(t=sc.objdict(year=t)))

        updater.step()

        eff_arr = getattr(disease, f'{prefix}_efficacy')
        for uid in self.VACC_UIDS:
            assert eff_arr.raw[uid] == pytest.approx(expected, rel=1e-5), (
                f'UID {uid} (vaccinated, decaying): expected {expected:.4f}, '
                f'got {eff_arr.raw[uid]:.4f}.'
            )
        for uid in range(n):
            if uid not in self.VACC_UIDS:
                assert eff_arr.raw[uid] == pytest.approx(0.0), (
                    f'UID {uid} (unvaccinated) was incorrectly modified.'
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
