"""
Tests for AntiMalarialDrug intervention.

ABM tests exercise MalariaHumanBase.set_drug_clearance using a real ms.Sim.
ODE tests exercise MalariaHumanODE.set_drug_clearance and the drug clearance
block inside step().
"""

import numpy as np
import pytest
import starsim as ss

import malariasim as ms
from malariasim.human_ode import MalariaHumanODE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_abm_sim(n=200, n_infected=50, seed=1):
    """Return an initialized ABM SIS sim with n_infected agents pre-seeded."""
    nodes = ms.NodeGrid(populations=[n])
    sim = ms.Sim(
        human='sis',
        mosquito='eir_only',
        nodes=nodes,
        start=2000,
        stop=2005,
        rand_seed=seed,
    )
    sim.init()
    disease = list(sim.diseases.values())[0]
    sus = disease.susceptible.uids[:n_infected]
    disease.set_prognoses(sus)
    return sim


def _make_ode_sim(n_nodes=1, n_bins=1):
    """Return an initialized MalariaHumanODE from a small real sim."""
    age_bin_edges = list(range(n_bins + 1))
    nodes = ms.NodeGrid(populations=[100] * n_nodes)
    sim = ms.Sim(
        human='ode',
        mosquito='eir_only',
        age_bin_edges=age_bin_edges,
        nodes=nodes,
        start=2000,
        stop=2001,
    )
    sim.init()
    return list(sim.diseases.values())[0]


def _uid(nid, b, n_bins):
    return nid * n_bins + b


# ===========================================================================
# ABM — MalariaHumanBase.set_drug_clearance
# ===========================================================================

class TestABMSetDrugClearance:

    def _disease(self, sim):
        return list(sim.diseases.values())[0]

    def test_susceptible_agents_are_noop(self):
        """set_drug_clearance must not change ti_recovered for susceptible agents."""
        sim = _make_abm_sim()
        disease = self._disease(sim)
        sus = disease.susceptible.uids
        if not len(sus):
            pytest.skip('no susceptible agents after init')
        before = np.asarray(disease.ti_recovered[sus]).copy()
        disease.set_drug_clearance(sus, 1.5 / 365, np.ones(len(sus)))
        np.testing.assert_array_equal(np.asarray(disease.ti_recovered[sus]), before)

    def test_infected_ti_recovered_set_sooner(self):
        """Drug overwrites ti_recovered to a time at or before natural recovery."""
        sim = _make_abm_sim()
        disease = self._disease(sim)
        inf = disease.infected.uids
        if not len(inf):
            pytest.skip('no infected agents after init')
        natural = np.asarray(disease.ti_recovered[inf]).copy()
        disease.set_drug_clearance(inf, 1.5 / 365, np.ones(len(inf)))
        drug = np.asarray(disease.ti_recovered[inf])
        assert np.all(drug <= natural), 'drug must not delay natural recovery'

    def test_ti_recovered_close_to_now(self):
        """Drawn ti_recovered is within a few steps of current sim.ti."""
        sim = _make_abm_sim()
        disease = self._disease(sim)
        inf = disease.infected.uids
        if not len(inf):
            pytest.skip('no infected agents after init')
        mean_dur_years = 3.0 / 365
        dt_year = sim.t.dt_year
        disease.set_drug_clearance(inf, mean_dur_years, np.ones(len(inf)))
        drug = np.asarray(disease.ti_recovered[inf])
        # At 20× the mean, virtually all exponential draws would have resolved
        max_steps = sim.ti + 20 * mean_dur_years / dt_year
        assert np.all(drug <= max_steps)

    def test_mixed_uids_only_infected_updated(self):
        """Passing both susceptible and infected UIDs only updates infected ones."""
        sim = _make_abm_sim()
        disease = self._disease(sim)
        sus = disease.susceptible.uids
        inf = disease.infected.uids
        if not len(sus) or not len(inf):
            pytest.skip('need both susceptible and infected agents')
        before_sus = np.asarray(disease.ti_recovered[sus]).copy()
        mixed = ss.uids(np.concatenate([np.asarray(sus), np.asarray(inf)]))
        disease.set_drug_clearance(mixed, 1.5 / 365, np.ones(len(mixed)))
        np.testing.assert_array_equal(np.asarray(disease.ti_recovered[sus]), before_sus)


# ===========================================================================
# ODE — MalariaHumanODE.set_drug_clearance state writes
# ===========================================================================

class TestODESetDrugClearance:

    def test_coverage_written(self):
        ode = _make_ode_sim()
        ode.set_drug_clearance(ss.uids([0]), 2.0 / 365, np.array([0.5]))
        assert float(ode._drug_coverage.raw[0]) == pytest.approx(0.5)

    def test_clear_end_t_written(self):
        ode = _make_ode_sim()
        mean_dur = 3.0 / 365
        t_before = ode.sim.t.year
        ode.set_drug_clearance(ss.uids([0]), mean_dur, np.array([1.0]))
        assert float(ode._drug_clear_end_t.raw[0]) == pytest.approx(t_before + mean_dur)

    def test_clearance_rate_written(self):
        ode = _make_ode_sim()
        mean_dur = 2.0 / 365
        ode.set_drug_clearance(ss.uids([0]), mean_dur, np.array([1.0]))
        assert float(ode._drug_clearance_rate.raw[0]) == pytest.approx(1.0 / mean_dur)

    def test_defaults_are_sentinel(self):
        """Before any drug is given all drug arrays hold their default values."""
        ode = _make_ode_sim(n_nodes=2, n_bins=3)
        np.testing.assert_array_equal(ode._drug_coverage.raw, 0.0)
        np.testing.assert_array_equal(ode._drug_clear_end_t.raw, -1.0)
        np.testing.assert_array_equal(ode._drug_clearance_rate.raw, 0.0)

    def test_untreated_uids_unchanged(self):
        """Only the targeted UID is written; neighbouring compartments stay at defaults."""
        ode = _make_ode_sim(n_nodes=2, n_bins=3)
        uid = _uid(0, 0, 3)
        ode.set_drug_clearance(ss.uids([uid]), 1.5 / 365, np.array([0.8]))
        others = [u for u in range(6) if u != uid]
        np.testing.assert_array_equal(ode._drug_coverage.raw[others], 0.0)
        np.testing.assert_array_equal(ode._drug_clear_end_t.raw[others], -1.0)
        np.testing.assert_array_equal(ode._drug_clearance_rate.raw[others], 0.0)


# ===========================================================================
# ODE — drug clearance logic in step()
# ===========================================================================

class TestODEDrugClearanceStep:
    """
    Tests call ode.step() directly.  NodeState EIR defaults to 0, so the only
    fluxes are recovery, drug clearance, deaths, and aging — enough to isolate
    drug behaviour.
    """

    def _set_ih(self, ode, uid, value):
        ode._Ih[ss.uids([uid])] = value

    def _get_ih(self, ode, uid):
        return float(ode._Ih.raw[uid])

    def _get_sh(self, ode, uid):
        return float(ode._Sh.raw[uid])

    def test_ih_decreases_with_active_drug(self):
        """A dosed compartment loses more Ih than an undosed one over one step."""
        ode_drug   = _make_ode_sim()
        ode_nodrug = _make_ode_sim()
        uid = 0
        self._set_ih(ode_drug,   uid, 50.0)
        self._set_ih(ode_nodrug, uid, 50.0)
        ode_drug.set_drug_clearance(ss.uids([uid]), 1.5 / 365, np.array([1.0]))
        ode_drug.step()
        ode_nodrug.step()
        assert self._get_ih(ode_drug, uid) < self._get_ih(ode_nodrug, uid)

    def test_sh_increases_with_active_drug(self):
        """Cleared agents move from Ih to Sh, so Sh rises more with drug than without."""
        ode_drug   = _make_ode_sim()
        ode_nodrug = _make_ode_sim()
        uid = 0
        self._set_ih(ode_drug,   uid, 50.0)
        self._set_ih(ode_nodrug, uid, 50.0)
        sh_before = self._get_sh(ode_drug, uid)
        ode_drug.set_drug_clearance(ss.uids([uid]), 1.5 / 365, np.array([1.0]))
        ode_drug.step()
        ode_nodrug.step()
        sh_drug   = self._get_sh(ode_drug,   uid)
        sh_nodrug = self._get_sh(ode_nodrug, uid)
        assert sh_drug > sh_nodrug

    def test_drug_resets_to_sentinel_after_expiry(self):
        """After _drug_clear_end_t passes, all three drug states reset to defaults."""
        ode = _make_ode_sim()
        uid = 0
        # Set end time behind current sim time so it expires on the first step
        ode._drug_coverage[ss.uids([uid])]       = 1.0
        ode._drug_clear_end_t[ss.uids([uid])]    = ode.sim.t.year - 0.001
        ode._drug_clearance_rate[ss.uids([uid])] = 365.0 / 1.5
        ode.step()
        assert float(ode._drug_coverage.raw[uid])       == pytest.approx(0.0)
        assert float(ode._drug_clear_end_t.raw[uid])    == pytest.approx(-1.0)
        assert float(ode._drug_clearance_rate.raw[uid]) == pytest.approx(0.0)

    def test_can_redose_after_expiry(self):
        """After expiry, set_drug_clearance can be called again with new state."""
        ode = _make_ode_sim()
        uid = 0
        # Expire first dose
        ode._drug_coverage[ss.uids([uid])]       = 1.0
        ode._drug_clear_end_t[ss.uids([uid])]    = ode.sim.t.year - 0.001
        ode._drug_clearance_rate[ss.uids([uid])] = 365.0 / 1.5
        ode.step()
        # Second dose
        mean_dur = 2.0 / 365
        ode.set_drug_clearance(ss.uids([uid]), mean_dur, np.array([0.6]))
        assert float(ode._drug_coverage.raw[uid])       == pytest.approx(0.6)
        assert float(ode._drug_clearance_rate.raw[uid]) == pytest.approx(1.0 / mean_dur)

    def test_half_coverage_clears_half_as_much(self):
        """Half coverage clears roughly half the Ih of full coverage in one step."""
        ode_full = _make_ode_sim()
        ode_half = _make_ode_sim()
        uid = 0
        self._set_ih(ode_full, uid, 100.0)
        self._set_ih(ode_half, uid, 100.0)
        ode_full.set_drug_clearance(ss.uids([uid]), 1.5 / 365, np.array([1.0]))
        ode_half.set_drug_clearance(ss.uids([uid]), 1.5 / 365, np.array([0.5]))
        ode_full.step()
        ode_half.step()
        full_cleared = 100.0 - self._get_ih(ode_full, uid)
        half_cleared = 100.0 - self._get_ih(ode_half, uid)
        assert full_cleared > 0
        assert half_cleared > 0
        assert full_cleared == pytest.approx(2 * half_cleared, rel=0.1)

    def test_no_clearance_without_drug(self):
        """With no drug active, drug_clear arrays play no role — Ih unchanged by drug."""
        ode = _make_ode_sim()
        uid = 0
        self._set_ih(ode, uid, 50.0)
        before = self._get_ih(ode, uid)
        ode.step()
        # Ih may decrease due to natural recovery, but _drug_coverage stays 0
        assert float(ode._drug_coverage.raw[uid])       == pytest.approx(0.0)
        assert float(ode._drug_clear_end_t.raw[uid])    == pytest.approx(-1.0)
        assert float(ode._drug_clearance_rate.raw[uid]) == pytest.approx(0.0)
