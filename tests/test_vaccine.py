"""
Tests for MalariaVaccine and MalariaVaccineUpdater.

MalariaVaccine.__init__ calls Updater.get('waning'), so it must be
instantiated inside the build_campaign() window.  The _activate() helper
opens and closes that window around test code.

MalariaVaccineUpdater.step() is tested with a mock disease and a
pre-configured WaningEffectUpdater, so no full Starsim sim is needed.
"""

import numpy as np
import pytest
import starsim as ss

from malariasim.updater import Updater, Target
from malariasim.updaters.waning import (
    WaningEffect,
    WaningEffectUpdater,
    _WaningEntry,
    _LOG_EXPIRY_INVERSE,
)
from malariasim.updaters.vaccine import MalariaVaccine, MalariaVaccineUpdater
from tests.mock_objects import (
    activate as _activate,
    make_waning as _make_waning,
    MockDisease as _MockDisease,
    make_waning_updater_for_disease as _make_waning_updater_for_disease,
    MockSim,
)


# ===========================================================================
# MalariaVaccine — construction and registration
# ===========================================================================

class TestMalariaVaccineConstruction:

    def test_error_outside_build_campaign(self):
        """MalariaVaccine calls Updater.get() in __init__, which guards against
        construction outside the build_campaign() window."""
        assert not Updater._active
        with pytest.raises(RuntimeError, match='build_campaign'):
            MalariaVaccine()

    def test_constructs_with_default_waning(self):
        """With no waning argument the vaccine defaults to permanent full
        efficacy — a safe, always-on protection useful for quick experiments."""
        restore = _activate()
        try:
            v = MalariaVaccine()
            assert isinstance(v.waning, WaningEffect)
            assert v.waning.initial_efficacy == pytest.approx(1.0)
            assert np.isinf(v.waning.box_duration.days)
        finally:
            restore()

    def test_constructs_with_custom_waning(self):
        """The user-supplied WaningEffect is stored by reference so the same
        object is used at administer() time without copying."""
        restore = _activate()
        try:
            w = _make_waning()
            v = MalariaVaccine(waning=w)
            assert v.waning is w
        finally:
            restore()

    def test_registers_vaccine_prefix_with_waning_updater(self):
        """Construction calls WaningEffectUpdater.register() so that the
        updater knows to create the vaccine_* state arrays during init_pre()."""
        waning_updater = Updater._possible_updaters[WaningEffectUpdater._updater_key]
        waning_updater.clear()
        restore = _activate()
        try:
            MalariaVaccine(waning=_make_waning())
            assert MalariaVaccine.PREFIX in waning_updater._entries
        finally:
            restore()

    def test_prefix_is_vaccine(self):
        """The class-level PREFIX constant is the shared key used by both
        MalariaVaccine and MalariaVaccineUpdater to identify their arrays."""
        assert MalariaVaccine.PREFIX == 'vaccine'

    def test_two_instances_share_one_waning_entry(self):
        """Multiple MalariaVaccine instances (e.g. for different campaigns)
        share a single set of state arrays — there is no per-instance storage."""
        waning_updater = Updater._possible_updaters[WaningEffectUpdater._updater_key]
        waning_updater.clear()
        restore = _activate()
        try:
            MalariaVaccine(waning=_make_waning())
            MalariaVaccine(waning=_make_waning(initial_efficacy=0.50))
            assert len(waning_updater._entries) == 1
        finally:
            restore()

    def test_clear_wipes_registration(self):
        """ms.Sim calls clear() before each build_campaign() so a fresh sim
        does not inherit registrations from a previous run."""
        waning_updater = Updater._possible_updaters[WaningEffectUpdater._updater_key]
        restore = _activate()
        try:
            MalariaVaccine(waning=_make_waning())
        finally:
            restore()
        waning_updater.clear()
        assert MalariaVaccine.PREFIX not in waning_updater._entries


# ===========================================================================
# MalariaVaccine — administer()
# ===========================================================================

class TestMalariaVaccineAdminister:
    """
    Tests for administer().

    We bypass init_pre() by wiring _disease and _waning_updater directly on
    the vaccine instance, and use setattribute() to set self.t (locked by
    ss.Module).
    """

    def _make_vaccine(self, waning=None):
        restore = _activate()
        try:
            v = MalariaVaccine(waning=waning or _make_waning())
        finally:
            restore()
        disease = _MockDisease(n=10)
        disease.add_prefix(MalariaVaccine.PREFIX)
        v._disease        = disease
        v._waning_updater = _make_waning_updater_for_disease(disease)
        v.setattribute('sim', MockSim(year=2010))
        return v, disease

    def test_reduces_prob_acquire_immediately(self):
        """administer() applies vaccine protection in the same timestep it is
        called, so agents dosed mid-step are already protected this tick."""
        v, d = self._make_vaccine()
        d._prob_acquire_indoor.values[:] = 1.0
        v.administer([0, 1])
        np.testing.assert_allclose(d._prob_acquire_indoor.values[[0, 1]], 1.0 * (1.0 - 0.80))

    def test_prob_acquire_unvaccinated_unchanged(self):
        """Only the specified UIDs receive the dose; all other agents are
        untouched even though they share the same prob_acquire array."""
        v, d = self._make_vaccine()
        d._prob_acquire_indoor.values[:] = 1.0
        v.administer([0, 1])
        np.testing.assert_array_equal(d._prob_acquire_indoor.values[2:], 1.0)

    def test_waning_arrays_written_via_updater(self):
        """administer() delegates all state-array writes to
        WaningEffectUpdater.apply() — MalariaVaccine never touches the
        vaccine_* arrays directly."""
        v, d = self._make_vaccine(waning=_make_waning(initial_efficacy=0.75))
        v.administer([0])
        assert d.vaccine_efficacy0.values[0]  == pytest.approx(0.75)
        assert d.vaccine_efficacy.values[0]   == pytest.approx(0.75)

    def test_box_end_written_correctly(self):
        """Confirms administer() passes the current sim time to
        WaningEffectUpdater.apply() so box_end is anchored to the dose time,
        not to some fixed reference year."""
        w = _make_waning(box_duration=ss.days(365))
        v, d = self._make_vaccine(waning=w)
        v.administer([0])
        assert d.vaccine_box_end.values[0] == pytest.approx(2011.0)

    def test_unvaccinated_keep_sentinel(self):
        """Agents not in the uid list retain the sentinel -1 in decay_end so
        the step loop skips them at zero cost."""
        v, d = self._make_vaccine()
        v.administer([0])
        np.testing.assert_array_equal(d.vaccine_decay_end.values[1:], -1.0)

    def test_permanent_vaccine(self):
        """A permanent WaningEffect writes infinite timestamps so the agent
        never enters the decaying or expired region."""
        v, d = self._make_vaccine(waning=WaningEffect())
        v.administer([0])
        assert np.isinf(d.vaccine_box_end.values[0])
        assert np.isinf(d.vaccine_decay_end.values[0])

    def test_full_efficacy_zeroes_prob_acquire(self):
        """initial_efficacy=1.0 means the agent cannot acquire infection this
        step — prob_acquire becomes 0."""
        v, d = self._make_vaccine(waning=WaningEffect(initial_efficacy=1.0,
                                                       box_duration=ss.days(365),
                                                       decay_time_constant=ss.days(365)))
        d._prob_acquire_indoor.values[:] = 1.0
        v.administer([0])
        assert d._prob_acquire_indoor.values[0] == pytest.approx(0.0)

    def test_zero_efficacy_no_prob_acquire_change(self):
        """initial_efficacy=0.0 is a valid no-op vaccine — prob_acquire is
        multiplied by 1.0 and left unchanged."""
        v, d = self._make_vaccine(waning=WaningEffect(initial_efficacy=0.0,
                                                       box_duration=ss.days(365),
                                                       decay_time_constant=ss.days(365)))
        d._prob_acquire_indoor.values[:] = 0.7
        v.administer([0])
        assert d._prob_acquire_indoor.values[0] == pytest.approx(0.7)


# ===========================================================================
# MalariaVaccineUpdater — step()
# ===========================================================================

class TestMalariaVaccineUpdaterStep:

    def _updater(self, disease):
        u = MalariaVaccineUpdater()
        u._disease        = disease
        u._waning_updater = _make_waning_updater_for_disease(disease)
        return u

    def test_no_op_when_disease_is_none(self):
        """If init_pre() found no disease module, step() must not raise — it
        simply has nothing to update."""
        u = MalariaVaccineUpdater()
        u._disease = None
        u._waning_updater = WaningEffectUpdater()
        u.step()

    def test_no_vaccinated_agents_no_change(self):
        """When all decay_end values are -1 (sentinel), active_mask() is all
        False and prob_acquire is not touched."""
        d = _MockDisease(n=5, prefix=MalariaVaccine.PREFIX)
        d._prob_acquire_indoor.values[:] = 1.0
        u = self._updater(d)
        u.step()
        np.testing.assert_array_equal(d._prob_acquire_indoor.values, 1.0)

    def test_vaccinated_agents_prob_acquire_reduced(self):
        """Only agents with an active effect (decay_end > 0) have their
        prob_acquire reduced; interleaved unvaccinated agents are left at 1.0."""
        d = _MockDisease(n=6, prefix=MalariaVaccine.PREFIX)
        d._prob_acquire_indoor.values[:] = 1.0
        for i in [0, 2, 4]:
            d.vaccine_efficacy.values[i]  = 0.80
            d.vaccine_decay_end.values[i] = 2020.0
        u = self._updater(d)
        u.step()
        np.testing.assert_allclose(d._prob_acquire_indoor.values[[0, 2, 4]], 1.0 * (1.0 - 0.80))
        np.testing.assert_array_equal(d._prob_acquire_indoor.values[[1, 3, 5]], 1.0)

    def test_expired_zero_efficacy_no_prob_acquire_change(self):
        """Expired agents have efficacy=0 (set by WaningEffectUpdater) so
        multiplying prob_acquire by (1-0) leaves it unchanged — no special expiry
        branch needed in the vaccine updater."""
        d = _MockDisease(n=3, prefix=MalariaVaccine.PREFIX)
        d._prob_acquire_indoor.values[:] = 0.7
        d.vaccine_efficacy.values[:] = 0.0
        d.vaccine_decay_end.values[:] = 2020.0
        u = self._updater(d)
        u.step()
        np.testing.assert_allclose(d._prob_acquire_indoor.values, 0.7)

    def test_partial_efficacy(self):
        """prob_acquire reduction is exactly (1 - vaccine_efficacy), verified
        across a range of efficacy values to catch off-by-one sign errors."""
        d = _MockDisease(n=4, prefix=MalariaVaccine.PREFIX)
        efficacies = [0.20, 0.50, 0.75, 1.0]
        for i, e in enumerate(efficacies):
            d.vaccine_efficacy.values[i]  = e
            d.vaccine_decay_end.values[i] = 2020.0
        d._prob_acquire_indoor.values[:] = 1.0
        u = self._updater(d)
        u.step()
        for i, e in enumerate(efficacies):
            assert d._prob_acquire_indoor.values[i] == pytest.approx(1.0 - e)

    def test_step_is_multiplicative(self):
        """The disease module may reset prob_acquire to a value other than 1.0
        (e.g. from another modifier).  The vaccine multiplies into whatever
        is already there rather than overwriting it."""

        d = _MockDisease(n=2, prefix=MalariaVaccine.PREFIX)
        d._prob_acquire_indoor.values[:] = 0.60
        d.vaccine_efficacy.values[:] = 0.80
        d.vaccine_decay_end.values[:] = 2020.0

        assert d.vaccine_efficacy.values[0] == 0.8
        assert d.vaccine_efficacy.values[1] == 0.8
        assert d._prob_acquire_indoor.values[0] == 0.6
        assert d._prob_acquire_indoor.values[1] == 0.6
        
        u = self._updater(d)
        u.step()
        np.testing.assert_allclose(d._prob_acquire_indoor.values, 0.60 * (1.0 - 0.80))


# ===========================================================================
# Updater auto-registration
# ===========================================================================

class TestAutoRegistration:

    def test_vaccine_updater_in_possible_updaters(self):
        """MalariaVaccineUpdater registers itself via __init_subclass__ the
        moment vaccine.py is imported — no explicit registration call needed."""
        assert MalariaVaccineUpdater._updater_key in Updater._possible_updaters

    def test_waning_updater_in_possible_updaters(self):
        """WaningEffectUpdater is registered by the same mechanism; confirmed
        here because MalariaVaccine depends on it being present."""
        assert WaningEffectUpdater._updater_key in Updater._possible_updaters

    def test_active_false_by_default(self):
        """The build_campaign() window must be closed at module load time so
        that importing malariasim does not accidentally open a registration
        window."""
        assert not Updater._active

    def test_activate_deactivate(self):
        """activate() and deactivate() toggle the window that guards
        Updater.get() — ms.Sim calls these around build_campaign()."""
        Updater.activate()
        assert Updater._active
        Updater.deactivate()
        assert not Updater._active


# ===========================================================================
# MalariaVaccineUpdater — integration tests with real starsim FloatArr
#
# These tests verify the hypothesis that MalariaVaccineUpdater.step() correctly
# writes prob_acquire on the real starsim FloatArr, not just on the mock _Arr.
#
# The mock _Arr in TestMalariaVaccineUpdaterStep delegates __setitem__ to plain
# numpy — it cannot expose bugs in starsim FloatArr's bool-mask write path.
#
# Two cases match the observed failure mode in run_malariasim.py:
#   Case A — all agents vaccinated (node_ids covers all nodes):
#             prob_acquire should be 0.0 for every alive agent at end of sim.
#   Case B — 80 % of agents vaccinated (node_ids=[0, 1], node 2 excluded):
#             vaccinated agents → prob_acquire = 0.0
#             unvaccinated agents (node 2) → prob_acquire = 1.0
# ===========================================================================

import malariasim as ms


def _vaccine_sim(node_ids_for_yr1, populations=(500, 300, 200),
                 start=2000, stop=2002, rand_seed=0):
    """
    Build and run a minimal three-node sim with a single vaccine campaign.

    The vaccine fires at start+1 with 100 % coverage and permanent (non-expiring)
    efficacy, targeting the nodes given by *node_ids_for_yr1* (None = all nodes).
    Returns the completed sim so tests can inspect state arrays.
    """
    nodes = ms.NodeGrid(populations=list(populations))

    def build_campaign(sim, eir_only=False):
        vaccine = ms.MalariaVaccine(
            name='test_vax',
            waning=ms.WaningEffect(
                initial_efficacy    = 1.0,
                box_duration        = ss.days(365 * 20),   # won't expire during test
                decay_time_constant = ss.days(365 * 20),
            ),
        )
        elig = (ms.HumanEligibility(node_ids=node_ids_for_yr1)
                if node_ids_for_yr1 is not None else None)
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'vacc_yr1',
                products    = vaccine,
                start       = start + 1,
                eligibility = elig,
                coverage    = 1.0,
            ),
        ])

    sim = ms.Sim(
        human          = 'sis',
        mosquito       = 'eir_only',
        nodes          = nodes,
        start          = start,
        stop           = stop,
        dt             = ss.days(1),
        rand_seed      = rand_seed,
        verbose        = 0,
        build_campaign = build_campaign,
    )
    sim.run()
    return sim


class TestMalariaVaccineUpdaterIntegration:
    """
    Integration tests for MalariaVaccineUpdater using real starsim FloatArr.

    After a full sim run, the LAST thing MalariaVaccineUpdater did was set
    prob_acquire for vaccinated agents.  step_state has not run again, so the
    values should still reflect the updater's write.

    If the bool-mask write on FloatArr is broken (copy issue), prob_acquire
    will equal 1.0 for vaccinated agents in case B even though they should
    be 0.0.
    """

    def _alive_node_prob(self, sim):
        """Return (node_ids, prob_acquire) arrays aligned to alive agents."""
        disease  = sim.diseases[0]
        alive    = np.asarray(sim.people.auids, dtype=int)
        nodes    = np.asarray(disease.node_id[sim.people.auids], dtype=int)
        pa       = disease._prob_acquire_indoor.values[alive]
        return nodes, pa

    def test_case_a_all_vaccinated_prob_acquire_is_zero(self):
        """Case A: vaccine targets all nodes (node_ids=None → all agents).

        After the sim completes, every alive agent was vaccinated and
        MalariaVaccineUpdater should have set prob_acquire = 0.0 for all of them.
        A value of 1.0 here means the bool-mask write on FloatArr failed.
        """
        sim = _vaccine_sim(node_ids_for_yr1=None)
        _, pa = self._alive_node_prob(sim)

        assert pa.max() == pytest.approx(0.0, abs=1e-6), (
            f"Case A: expected prob_acquire=0.0 for all vaccinated agents, "
            f"but max={pa.max():.6f} (write did not persist)"
        )

    def test_case_b_vaccinated_nodes_prob_acquire_is_zero(self):
        """Case B: vaccine targets nodes 0 and 1 only (node 2 unvaccinated).

        Agents in nodes 0 and 1 were vaccinated; their prob_acquire should
        be 0.0.  If the bool-mask write fails for a partial mask, these will
        remain at 1.0 instead.
        """
        sim = _vaccine_sim(node_ids_for_yr1=[0, 1])
        node_ids, pa = self._alive_node_prob(sim)

        vaccinated_pa = pa[node_ids != 2]
        assert vaccinated_pa.max() == pytest.approx(0.0, abs=1e-6), (
            f"Case B: vaccinated agents (nodes 0,1) should have prob_acquire=0.0, "
            f"but max={vaccinated_pa.max():.6f} (partial bool-mask write failed)"
        )

    def test_case_b_unvaccinated_node_prob_acquire_is_one(self):
        """Case B: agents in node 2 (unvaccinated) must keep prob_acquire=1.0.

        This confirms the node filter is working: node 2 agents were not
        given the vaccine so step_state's reset to 1.0 is the final value.
        """
        sim = _vaccine_sim(node_ids_for_yr1=[0, 1])
        node_ids, pa = self._alive_node_prob(sim)

        unvaccinated_pa = pa[node_ids == 2]
        assert len(unvaccinated_pa) > 0, "No agents found in node 2"
        np.testing.assert_allclose(unvaccinated_pa, 1.0, atol=1e-6,
            err_msg=(
                f"Case B: unvaccinated agents (node 2) should have prob_acquire=1.0, "
                f"but min={unvaccinated_pa.min():.6f}"
            )
        )

    def test_case_a_prevalence_lower_than_baseline(self):
        """Case A: all agents vaccinated.

        Cumulative new infections after the vaccine fires should be much
        lower than in a no-vaccine baseline run.  Uses a permanent (non-
        expiring) vaccine so the protection is in place for the whole
        post-vaccination period.
        """
        nodes = ms.NodeGrid(populations=[500, 300, 200])
        start, stop = 2000, 2005

        def _run(node_ids_for_yr1):
            def build_campaign(sim, eir_only=False):
                vaccine = ms.MalariaVaccine(
                    name='vax',
                    waning=ms.WaningEffect(),   # permanent full efficacy
                )
                elig = (ms.HumanEligibility(node_ids=node_ids_for_yr1)
                        if node_ids_for_yr1 is not None else None)
                return ms.Campaign(sim, [
                    ms.InterventionDistributorScheduled(
                        name='vacc_yr1', products=vaccine, start=start+1,
                        eligibility=elig, coverage=1.0,
                    ),
                ])
            return ms.Sim(
                human='sis', mosquito='eir_only', nodes=nodes,
                start=start, stop=stop, dt=ss.days(1),
                rand_seed=0, verbose=0,
                build_campaign=build_campaign,
            )

        sim_baseline = ms.Sim(
            human='sis', mosquito='eir_only', nodes=nodes,
            start=start, stop=stop, dt=ss.days(1),
            rand_seed=0, verbose=0,
        )
        sim_a = _run(node_ids_for_yr1=None)  # all nodes

        sim_baseline.run()
        sim_a.run()

        # Count new infections from the step AFTER vaccination to end of sim.
        # ti_vax is the timestep index nearest to start+1.
        import sciris as sc
        ti_vax = sc.findnearest(sim_a.t.yearvec, start + 1)
        cum_baseline = sim_baseline.diseases[0].results.new_infections[ti_vax:].sum()
        cum_a        = sim_a.diseases[0].results.new_infections[ti_vax:].sum()

        assert cum_a < cum_baseline * 0.10, (
            f"Case A: expected < 10 % of baseline infections post-vaccination, "
            f"but got {cum_a:.0f} vs baseline {cum_baseline:.0f}"
        )

    def test_case_b_prevalence_lower_than_baseline(self):
        """Case B: only nodes 0 and 1 vaccinated (node 2 excluded).

        Post-vaccination cumulative infections should be well below the
        no-vaccine baseline even though only 80 % of agents are protected.
        This is the key regression test: if prob_acquire is not being applied
        during transmission, infections will equal the baseline.
        """
        nodes = ms.NodeGrid(populations=[500, 300, 200])
        start, stop = 2000, 2005

        def _run(node_ids_for_yr1):
            def build_campaign(sim, eir_only=False):
                vaccine = ms.MalariaVaccine(
                    name='vax',
                    waning=ms.WaningEffect(),   # permanent full efficacy
                )
                elig = (ms.HumanEligibility(node_ids=node_ids_for_yr1)
                        if node_ids_for_yr1 is not None else None)
                return ms.Campaign(sim, [
                    ms.InterventionDistributorScheduled(
                        name='vacc_yr1', products=vaccine, start=start+1,
                        eligibility=elig, coverage=1.0,
                    ),
                ])
            return ms.Sim(
                human='sis', mosquito='eir_only', nodes=nodes,
                start=start, stop=stop, dt=ss.days(1),
                rand_seed=0, verbose=0,
                build_campaign=build_campaign,
            )

        sim_baseline = ms.Sim(
            human='sis', mosquito='eir_only', nodes=nodes,
            start=start, stop=stop, dt=ss.days(1),
            rand_seed=0, verbose=0,
        )
        sim_b = _run(node_ids_for_yr1=[0, 1])   # nodes 0 and 1 only

        sim_baseline.run()
        sim_b.run()

        import sciris as sc
        ti_vax = sc.findnearest(sim_b.t.yearvec, start + 1)
        cum_baseline = sim_baseline.diseases[0].results.new_infections[ti_vax:].sum()
        cum_b        = sim_b.diseases[0].results.new_infections[ti_vax:].sum()

        assert cum_b < cum_baseline * 0.50, (
            f"Case B: expected < 50 % of baseline infections post-vaccination "
            f"(80 % of agents protected), "
            f"but got {cum_b:.0f} vs baseline {cum_baseline:.0f}. "
            f"If cum_b ≈ cum_baseline the vaccine has no effect."
        )

    def test_case_b_agent_counts_per_node(self):
        """Sanity check: the three nodes have exactly the specified populations
        so the vaccinated / unvaccinated split is what we expect."""
        pops = (500, 300, 200)
        sim  = _vaccine_sim(node_ids_for_yr1=[0, 1], populations=pops, stop=2002)
        disease  = sim.diseases[0]
        node_ids = np.asarray(disease.node_id[sim.people.auids], dtype=int)
        for nid, expected in enumerate(pops):
            count = int((node_ids == nid).sum())
            assert count == pytest.approx(expected, abs=expected * 0.05), (
                f"Node {nid}: expected ~{expected} agents, got {count}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
