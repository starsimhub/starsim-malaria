"""
Tests for MalariaHumanODE intervention interface.

All tests that exercise per-compartment state use a small real ms.Sim so that
ss.FloatArr states are properly initialized by Starsim's init lifecycle.

UID convention (n_nodes=2, n_bins=3):
    uid = nid * n_bins + bin
    node 0: uids 0, 1, 2
    node 1: uids 3, 4, 5
"""

import numpy as np
import pytest
import starsim as ss

import malariasim as ms
from malariasim.human_ode import MalariaHumanODE


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_ode(n_nodes=2, n_bins=3):
    """Return an initialized MalariaHumanODE from a small real sim."""
    age_bin_edges = list(range(n_bins + 1))          # e.g. [0,1,2,3] for 3 bins
    nodes         = ms.NodeGrid(populations=[100] * n_nodes)
    sim           = ms.Sim(
        human         = 'ode',
        mosquito      = 'eir_only',
        age_bin_edges = age_bin_edges,
        nodes         = nodes,
        start         = 2000,
        stop          = 2001,
    )
    sim.init()
    return list(sim.diseases.values())[0]


def _uid(nid, b, n_bins):
    """Return the UID for (node, bin) given n_bins."""
    return nid * n_bins + b


def _get(state, uid):
    """Read a scalar float from a ss.FloatArr at a single uid."""
    return float(state.raw[uid])


def _set(state, uid, value):
    """Write a scalar value into a ss.FloatArr at a single uid."""
    state[ss.uids([uid])] = value


# ===========================================================================
# ODE state structure
# ===========================================================================

class TestODEStates:
    """Verify that ss.FloatArr states are created with correct size and defaults."""

    def test_prob_acquire_size(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        assert len(ode._prob_acquire_indoor.raw) == 6

    def test_vaccinated_fraction_size(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        assert len(ode._vaccinated_fraction.raw) == 6

    def test_dose_time_size(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        assert len(ode._dose_time.raw) == 6

    def test_prob_acquire_default_one(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        np.testing.assert_array_equal(ode._prob_acquire_indoor.raw, 1.0)

    def test_vaccinated_fraction_default_zero(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        np.testing.assert_array_equal(ode._vaccinated_fraction.raw, 0.0)

    def test_single_node_single_bin(self):
        ode = _make_ode(n_nodes=1, n_bins=1)
        assert len(ode._prob_acquire_indoor.raw) == 1
        assert len(ode._vaccinated_fraction.raw) == 1
        assert len(ode._dose_time.raw) == 1

    def test_no_initialize_intervention_states(self):
        """Method removed — states are sized by Starsim init instead."""
        ode = _make_ode()
        assert not hasattr(ode, 'initialize_intervention_states')

    def test_no_define_intervention_states(self):
        """Method removed — states are declared in define_states instead."""
        ode = _make_ode()
        assert not hasattr(ode, 'define_intervention_states')

    def test_apply_coverage_exists(self):
        ode = _make_ode()
        assert hasattr(ode, 'apply_coverage')


# ===========================================================================
# reset_acquisition_probability
# ===========================================================================

class TestResetAcquisitionProbability:

    def test_resets_all_to_one(self):
        ode = _make_ode()
        ode._prob_acquire_indoor[ode.sim.people.auids] = 0.5
        ode.reset_acquisition_probability()
        np.testing.assert_array_equal(ode._prob_acquire_indoor.raw, 1.0)

    def test_resets_across_all_nodes_and_bins(self):
        ode = _make_ode(n_nodes=3, n_bins=4)
        ode._prob_acquire_indoor[ode.sim.people.auids] = 0.0
        ode.reset_acquisition_probability()
        np.testing.assert_array_equal(ode._prob_acquire_indoor.raw, 1.0)

    def test_already_one_unchanged(self):
        ode = _make_ode()
        ode.reset_acquisition_probability()
        np.testing.assert_array_equal(ode._prob_acquire_indoor.raw, 1.0)


# ===========================================================================
# set_acquisition_probability
# ===========================================================================

class TestSetAcquisitionProbability:

    def test_uid_maps_to_correct_node_and_bin(self):
        """uid=1 (node 0, bin 1 with n_bins=3): _prob_acquire = 1 - efficacy."""
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid = _uid(0, 1, 3)
        ode.set_acquisition_probability(
            uids    = np.array([uid]),
            efficacy = np.array([0.80]),
            weights  = np.array([1.0]),
        )
        assert _get(ode._prob_acquire_indoor, uid) == pytest.approx(0.20)
        assert _get(ode._prob_acquire_indoor, _uid(0, 0, 3)) == pytest.approx(1.0)  # untouched
        assert _get(ode._prob_acquire_indoor, _uid(1, 0, 3)) == pytest.approx(1.0)  # untouched

    def test_node_1_uid_mapping(self):
        """uid=3 (node 1, bin 0 with n_bins=3)."""
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid = _uid(1, 0, 3)
        ode.set_acquisition_probability(
            uids    = np.array([uid]),
            efficacy = np.array([0.50]),
            weights  = np.array([1.0]),
        )
        assert _get(ode._prob_acquire_indoor, uid) == pytest.approx(0.50)
        assert _get(ode._prob_acquire_indoor, _uid(0, 0, 3)) == pytest.approx(1.0)

    def test_partial_weight_reduces_less(self):
        """_prob_acquire = 1 - weights * efficacy; weight 0.5 halves effect."""
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        ode.set_acquisition_probability(
            uids    = np.array([uid]),
            efficacy = np.array([1.0]),
            weights  = np.array([0.5]),
        )
        assert _get(ode._prob_acquire_indoor, uid) == pytest.approx(0.5)

    def test_zero_efficacy_no_change(self):
        ode = _make_ode()
        uids = np.array([0, 1, 2])
        ode.set_acquisition_probability(
            uids    = uids,
            efficacy = np.zeros(3),
            weights  = np.ones(3),
        )
        for uid in uids:
            assert _get(ode._prob_acquire_indoor, uid) == pytest.approx(1.0)

    def test_multiple_uids_across_nodes(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid0 = _uid(0, 0, 3)   # 0
        uid1 = _uid(1, 0, 3)   # 3
        ode.set_acquisition_probability(
            uids    = np.array([uid0, uid1]),
            efficacy = np.array([0.80, 0.60]),
            weights  = np.array([1.0,  1.0]),
        )
        assert _get(ode._prob_acquire_indoor, uid0) == pytest.approx(0.20)
        assert _get(ode._prob_acquire_indoor, uid1) == pytest.approx(0.40)
        assert _get(ode._prob_acquire_indoor, _uid(0, 1, 3)) == pytest.approx(1.0)  # untouched


# ===========================================================================
# pre_administer
# ===========================================================================

class TestPreAdminister:

    def test_returns_t_now_when_fraction_is_zero(self):
        """First dose into an unvaccinated compartment: t_eff == t_now."""
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        t_eff = ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2005.0)
        assert t_eff[0] == pytest.approx(2005.0)

    def test_updates_vaccinated_fraction_first_dose(self):
        """50% coverage into empty compartment → fraction becomes 0.5."""
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2005.0)
        assert _get(ode._vaccinated_fraction, uid) == pytest.approx(0.5)

    def test_updates_dose_time_first_dose(self):
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2005.0)
        assert _get(ode._dose_time, uid) == pytest.approx(2005.0)

    def test_weighted_mean_dose_time_second_dose(self):
        """Second dose at t=2006 into 50%-vaccinated compartment:
        f_old=0.5, c_new=0.5*(1-0.5)=0.25 → f_new=0.75
        t_eff = (0.5*2005 + 0.25*2006) / 0.75"""
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2005.0)
        t_eff = ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2006.0)
        f_new    = 0.5 + 0.5 * 0.5       # 0.75
        expected = (0.5 * 2005.0 + 0.25 * 2006.0) / f_new
        assert t_eff[0] == pytest.approx(expected)

    def test_vaccinated_fraction_accumulates(self):
        """Two 50% doses: 0 → 0.5 → 0.75."""
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2005.0)
        ode.pre_administer(np.array([uid]), np.array([0.5]), t_now=2006.0)
        assert _get(ode._vaccinated_fraction, uid) == pytest.approx(0.75)

    def test_full_coverage_sets_fraction_to_one(self):
        ode = _make_ode()
        uid = _uid(0, 0, 3)
        ode.pre_administer(np.array([uid]), np.array([1.0]), t_now=2005.0)
        assert _get(ode._vaccinated_fraction, uid) == pytest.approx(1.0)

    def test_multiple_uids_independent(self):
        """Different uids update independently."""
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid0 = _uid(0, 0, 3)   # 0
        uid1 = _uid(1, 0, 3)   # 3
        ode.pre_administer(
            uids    = np.array([uid0, uid1]),
            weights = np.array([0.4,  0.6]),
            t_now   = 2005.0,
        )
        assert _get(ode._vaccinated_fraction, uid0) == pytest.approx(0.4)
        assert _get(ode._vaccinated_fraction, uid1) == pytest.approx(0.6)
        assert _get(ode._dose_time,           uid0) == pytest.approx(2005.0)
        assert _get(ode._dose_time,           uid1) == pytest.approx(2005.0)

    def test_untouched_bins_unchanged(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid0 = _uid(0, 0, 3)
        uid_other = _uid(0, 1, 3)
        uid_node1 = _uid(1, 0, 3)
        ode.pre_administer(np.array([uid0]), np.array([0.5]), t_now=2005.0)
        assert _get(ode._vaccinated_fraction, uid_other) == pytest.approx(0.0)
        assert _get(ode._vaccinated_fraction, uid_node1) == pytest.approx(0.0)


# ===========================================================================
# sync_vaccinated_fraction
# ===========================================================================

class TestSyncVaccinatedFraction:

    def _make_ode_with_state(self):
        """Return ODE with _vaccinated_fraction and _dose_time pre-populated."""
        ode  = _make_ode(n_nodes=2, n_bins=3)
        auids = ode.sim.people.auids
        ode._vaccinated_fraction[auids] = 0.5
        ode._dose_time[auids]           = 2005.0
        return ode

    def test_zeros_expired_fraction(self):
        """Compartments whose efficacy == 0 get _vaccinated_fraction reset."""
        ode = self._make_ode_with_state()
        # uid 0 = node 0, bin 0; expires
        efficacy = np.array([0.0, 0.8, 0.8, 0.8, 0.8, 0.8])
        ode.sync_vaccinated_fraction(np.arange(6), efficacy)
        assert _get(ode._vaccinated_fraction, 0) == pytest.approx(0.0)

    def test_zeros_expired_dose_time(self):
        ode = self._make_ode_with_state()
        efficacy = np.array([0.0, 0.8, 0.8, 0.8, 0.8, 0.8])
        ode.sync_vaccinated_fraction(np.arange(6), efficacy)
        assert _get(ode._dose_time, 0) == pytest.approx(0.0)

    def test_active_bins_untouched(self):
        ode = self._make_ode_with_state()
        efficacy = np.array([0.0, 0.8, 0.8, 0.8, 0.8, 0.8])
        ode.sync_vaccinated_fraction(np.arange(6), efficacy)
        assert _get(ode._vaccinated_fraction, 1) == pytest.approx(0.5)
        assert _get(ode._vaccinated_fraction, 3) == pytest.approx(0.5)

    def test_all_expired_resets_all(self):
        ode = self._make_ode_with_state()
        efficacy = np.zeros(6)
        ode.sync_vaccinated_fraction(np.arange(6), efficacy)
        np.testing.assert_array_equal(ode._vaccinated_fraction.raw, 0.0)
        np.testing.assert_array_equal(ode._dose_time.raw,           0.0)

    def test_none_expired_no_change(self):
        ode = self._make_ode_with_state()
        efficacy = np.full(6, 0.8)
        ode.sync_vaccinated_fraction(np.arange(6), efficacy)
        np.testing.assert_array_equal(ode._vaccinated_fraction.raw, 0.5)

    def test_empty_uids_no_change(self):
        ode = self._make_ode_with_state()
        ode.sync_vaccinated_fraction(np.array([], dtype=int), np.array([]))
        np.testing.assert_array_equal(ode._vaccinated_fraction.raw, 0.5)


# ===========================================================================
# get_vaccinated_weights
# ===========================================================================

class TestGetVaccinatedWeights:

    def test_returns_fraction_for_uid(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid = _uid(0, 0, 3)
        _set(ode._vaccinated_fraction, uid, 0.4)
        weights = ode.get_vaccinated_weights(np.array([uid]))
        assert weights[0] == pytest.approx(0.4)

    def test_uid_to_node_bin_mapping(self):
        """uid 4 = node 1, bin 1 (n_bins=3)."""
        ode = _make_ode(n_nodes=2, n_bins=3)
        uid = _uid(1, 1, 3)   # = 4
        _set(ode._vaccinated_fraction, uid, 0.7)
        weights = ode.get_vaccinated_weights(np.array([uid]))
        assert weights[0] == pytest.approx(0.7)

    def test_multiple_uids(self):
        ode = _make_ode(n_nodes=2, n_bins=3)
        _set(ode._vaccinated_fraction, _uid(0, 0, 3), 0.3)   # uid 0
        _set(ode._vaccinated_fraction, _uid(0, 2, 3), 0.6)   # uid 2
        _set(ode._vaccinated_fraction, _uid(1, 0, 3), 0.9)   # uid 3
        weights = ode.get_vaccinated_weights(np.array([0, 2, 3]))
        np.testing.assert_allclose(weights, [0.3, 0.6, 0.9])

    def test_unvaccinated_returns_zero(self):
        ode = _make_ode()
        weights = ode.get_vaccinated_weights(np.array([0, 1, 2]))
        np.testing.assert_array_equal(weights, 0.0)


# ===========================================================================
# _age_vaccinated_state (static)
# ===========================================================================

_age_vaccinated_state = MalariaHumanODE._age_vaccinated_state


class TestAgeVaccinatedState:
    """
    Tests for MalariaHumanODE._age_vaccinated_state.

    Uses simple 3-bin populations so expected values can be computed by hand.
    Arrays are 2-D (1, n_bins) matching what step() passes: one node, three
    age bins.  aging_dt and death_dt remain 1-D (n_bins,) and broadcast.
    Bin widths and death rates are passed as dt-scaled rates (already multiplied
    by dt), matching how step() calls this method.
    """

    def test_no_aging_no_death_fraction_unchanged(self):
        """With zero aging and death rates, vaccinated fraction is conserved."""
        Sh   = np.array([[100.0, 200.0, 150.0]])
        Ih   = np.array([[ 10.0,  20.0,  15.0]])
        frac = np.array([[  0.5,   0.5,   0.5]])
        t    = np.array([[2005.0, 2005.0, 2005.0]])
        new_frac, new_t = _age_vaccinated_state(Sh, Ih, frac, t,
                                    aging_dt=np.zeros(3), death_dt=np.zeros(3))
        np.testing.assert_allclose(new_frac, 0.5)

    def test_no_aging_no_death_dose_time_unchanged(self):
        """Bins 1+ keep their dose time; bin 0 is always zeroed (births unvaccinated)."""
        Sh   = np.array([[100.0, 200.0, 150.0]])
        Ih   = np.array([[ 10.0,  20.0,  15.0]])
        frac = np.array([[  0.5,   0.5,   0.5]])
        t    = np.array([[2003.0, 2004.0, 2005.0]])
        _, new_t = _age_vaccinated_state(Sh, Ih, frac, t,
                             aging_dt=np.zeros(3), death_dt=np.zeros(3))
        np.testing.assert_allclose(new_t, [[0.0, 2004.0, 2005.0]])

    def test_aging_moves_vaccinated_into_next_bin(self):
        """Vaccinated people in bin 0 age into bin 1; bin 0 fraction falls."""
        Sh   = np.array([[100.0, 100.0, 100.0]])
        Ih   = np.zeros((1, 3))
        frac = np.array([[1.0, 0.0, 0.0]])   # only bin 0 vaccinated
        t    = np.array([[2005.0, 0.0, 0.0]])
        # aging_dt = 0.1 means 10% of each bin ages out per step
        aging_dt  = np.array([0.1, 0.1, 0.1])
        death_dt  = np.zeros(3)
        new_frac, _ = _age_vaccinated_state(Sh, Ih, frac, t, aging_dt, death_dt)
        # V=100, V_out=10 → V_new[0]=90, N_new=100 → frac[0]=0.9
        assert new_frac[0, 0] == pytest.approx(0.9)
        # bin 1 receives 10 vaccinated into 100 people → frac[1]=0.1
        assert new_frac[0, 1] == pytest.approx(0.1)
        # bin 2 receives from bin 1 which was 0 → frac[2]=0
        assert new_frac[0, 2] == pytest.approx(0.0)

    def test_dose_time_flows_into_next_bin(self):
        """Dose time of vaccinated cohort in bin 0 flows into bin 1 on aging."""
        Sh   = np.array([[100.0, 100.0, 100.0]])
        Ih   = np.zeros((1, 3))
        frac = np.array([[1.0, 0.0, 0.0]])
        t    = np.array([[2005.0, 0.0, 0.0]])
        aging_dt = np.array([0.1, 0.1, 0.1])
        death_dt = np.zeros(3)
        _, new_t = _age_vaccinated_state(Sh, Ih, frac, t, aging_dt, death_dt)
        # bin 1 receives 10 vaccinated people dosed at 2005 → dose_time = 2005
        assert new_t[0, 1] == pytest.approx(2005.0)

    def test_bin0_dose_time_zeroed_births_unvaccinated(self):
        """Bin 0 always gets T_new[0]=0 since newborns are unvaccinated."""
        Sh   = np.array([[100.0, 100.0, 100.0]])
        Ih   = np.zeros((1, 3))
        frac = np.array([[0.5, 0.5, 0.5]])
        t    = np.array([[2005.0, 2005.0, 2005.0]])
        aging_dt = np.array([0.1, 0.1, 0.1])
        death_dt = np.zeros(3)
        _, new_t = _age_vaccinated_state(Sh, Ih, frac, t, aging_dt, death_dt)
        assert new_t[0, 0] == pytest.approx(0.0)

    def test_zero_population_bin_fraction_is_zero(self):
        """When a bin is empty, fraction should be 0 not nan."""
        Sh   = np.array([[0.0, 100.0, 100.0]])
        Ih   = np.zeros((1, 3))
        frac = np.array([[0.5, 0.5, 0.0]])
        t    = np.array([[2005.0, 2005.0, 0.0]])
        aging_dt = np.zeros(3)
        death_dt = np.zeros(3)
        new_frac, _ = _age_vaccinated_state(Sh, Ih, frac, t, aging_dt, death_dt)
        assert new_frac[0, 0] == pytest.approx(0.0)
        assert not np.isnan(new_frac[0, 0])

    def test_death_reduces_vaccinated_count(self):
        """Death removes vaccinated people; fraction falls because V_out is applied
        to the post-update N (an O(dt) approximation matching the Euler scheme)."""
        Sh   = np.array([[100.0, 0.0, 0.0]])
        Ih   = np.zeros((1, 3))
        frac = np.array([[1.0, 0.0, 0.0]])
        t    = np.array([[2005.0, 0.0, 0.0]])
        aging_dt = np.zeros(3)
        death_dt = np.array([0.1, 0.0, 0.0])
        new_frac, _ = _age_vaccinated_state(Sh, Ih, frac, t, aging_dt, death_dt)
        # V=1.0*100=100, V_out=100*0.1=10, V_new=90, N_new=100 → frac=0.9
        assert new_frac[0, 0] == pytest.approx(0.9)


# ===========================================================================
# MalariaHumanODE.validate_eligibility
# ===========================================================================

class TestODEValidateEligibility:
    """validate_eligibility raises ValueError when age bounds don't land on a
    bin edge, and passes silently when they do (or when bounds are absent)."""

    def _make_eligibility(self, min_age=None, max_age=None):
        from malariasim.distributors.eligibility import HumanEligibility
        return HumanEligibility(
            min_age=ss.years(min_age) if min_age is not None else None,
            max_age=ss.years(max_age) if max_age is not None else None,
        )

    def test_aligned_bounds_no_error(self):
        """min_age and max_age both on bin edges → no error."""
        d = _make_ode(n_nodes=1, n_bins=3)   # edges [0, 1, 2, 3]
        d.validate_eligibility(self._make_eligibility(min_age=1, max_age=2))

    def test_misaligned_min_age_raises(self):
        """min_age that falls between bin edges → ValueError naming the value."""
        d = _make_ode(n_nodes=1, n_bins=3)   # edges [0, 1, 2, 3]
        with pytest.raises(ValueError, match='min_age'):
            d.validate_eligibility(self._make_eligibility(min_age=0.5, max_age=2))

    def test_misaligned_max_age_raises(self):
        """max_age that falls between bin edges → ValueError naming the value."""
        d = _make_ode(n_nodes=1, n_bins=3)   # edges [0, 1, 2, 3]
        with pytest.raises(ValueError, match='max_age'):
            d.validate_eligibility(self._make_eligibility(min_age=1, max_age=2.5))

    def test_none_min_age_no_error(self):
        """Omitting min_age is valid — only the supplied bounds are checked."""
        d = _make_ode(n_nodes=1, n_bins=3)
        d.validate_eligibility(self._make_eligibility(max_age=2))

    def test_none_max_age_no_error(self):
        """Omitting max_age is valid — only the supplied bounds are checked."""
        d = _make_ode(n_nodes=1, n_bins=3)
        d.validate_eligibility(self._make_eligibility(min_age=1))

    def test_no_bin_edges_no_error(self):
        """Single-bin ODE has no _age_bin_edges → validation is skipped."""
        nodes = ms.NodeGrid(populations=[100])
        sim   = ms.Sim(human='ode', mosquito='eir_only', nodes=nodes,
                       start=2000, stop=2001)
        sim.init()
        d = list(sim.diseases.values())[0]
        assert d._age_bin_edges is None
        d.validate_eligibility(self._make_eligibility(min_age=5, max_age=10))

    def test_error_message_contains_bad_value(self):
        """The ValueError names the offending value so the user knows what to fix."""
        d = _make_ode(n_nodes=1, n_bins=3)
        with pytest.raises(ValueError, match='1.7'):
            d.validate_eligibility(self._make_eligibility(min_age=1.7, max_age=2))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
