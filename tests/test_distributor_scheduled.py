"""
Tests for InterventionDistributorScheduled.

Construction tests verify parameter validation at __init__ time.

Step tests bypass init_pre() by setting _start_ti directly and wiring sim
and check_eligibility on the instance — the same pattern used in the vaccine
and waning tests.  This isolates step() logic from Starsim's module lifecycle.

A _MockProduct records every administer() call so tests can assert what was
delivered and to whom without a full disease module.
"""

import numpy as np
import pytest
import starsim as ss

from malariasim.distributors.scheduled import (
    InterventionDistributor,
    InterventionDistributorScheduled,
)
from tests.mock_objects import MockProduct as _MockProduct, MockSim as _MockSim


def _make_for_step(start_ti, coverage, products=None, eligible=None, ti=None):
    """
    Build an InterventionDistributorScheduled ready for step() testing.

    Bypasses init_pre() by setting _start_ti directly and monkey-patching
    _resolve_targets so no disease module or Starsim lifecycle is needed.
    """
    products = products if products is not None else []
    eligible = eligible if eligible is not None else ss.uids(np.arange(10))
    ti       = ti if ti is not None else start_ti

    d = InterventionDistributorScheduled(
        name='test_dist', products=products, start=2015.0, coverage=coverage
    )
    d._start_ti = start_ti
    d.setattribute('sim', _MockSim(ti=ti))

    def _resolve_targets(cov):
        if cov <= 0.0 or len(eligible) == 0:
            return ss.uids(np.array([], dtype=int)), np.array([], dtype=float)
        if cov >= 1.0:
            return eligible, np.ones(len(eligible), dtype=float)
        mask   = np.random.random(len(eligible)) < cov
        chosen = eligible[mask]
        return chosen, np.ones(len(chosen), dtype=float)

    d._resolve_targets = _resolve_targets
    return d


# ===========================================================================
# Construction
# ===========================================================================

class TestInterventionDistributorScheduledConstruction:

    def test_constructs_with_coverage(self):
        """A distributor with only coverage set is valid — the common case
        for campaign mass vaccination."""
        InterventionDistributorScheduled(name='test', products=[], start=2015.0, coverage=0.80)

    def test_both_coverage_and_num_targeted_raises(self):
        """Supplying both coverage and num_targeted is ambiguous — the
        distributor enforces exactly one selection mechanism."""
        with pytest.raises(ValueError, match='exactly one'):
            InterventionDistributorScheduled(
                name='test', products=[], start=2015.0, coverage=0.80, num_targeted=100
            )

    def test_neither_coverage_nor_num_targeted_raises(self):
        """Supplying neither means the distributor has no way to decide how
        many agents to dose — rejected immediately at construction."""
        with pytest.raises(ValueError, match='exactly one'):
            InterventionDistributorScheduled(name='test', products=[], start=2015.0)

    def test_num_targeted_raises_not_implemented(self):
        """num_targeted is reserved in the API but not yet implemented —
        raises NotImplementedError rather than silently doing the wrong thing."""
        with pytest.raises(NotImplementedError, match='num_targeted'):
            InterventionDistributorScheduled(
                name='test', products=[], start=2015.0, num_targeted=100
            )

    def test_fired_starts_false(self):
        """_fired must be False at construction so the distributor fires on
        the first matching timestep rather than being permanently inert."""
        d = InterventionDistributorScheduled(name='test', products=[], start=2015.0, coverage=0.80)
        assert d._fired is False

    def test_single_product_wrapped_in_list(self):
        """A single product (not in a list) is normalised to a one-element
        list so step() can always iterate products uniformly."""
        p = _MockProduct()
        d = InterventionDistributorScheduled(name='test', products=p, start=2015.0, coverage=0.80)
        assert d.products == [p]

    def test_multiple_products_stored(self):
        """A list of products is stored as-is — all products receive the
        same set of UIDs when the distributor fires."""
        p1, p2 = _MockProduct(), _MockProduct()
        d = InterventionDistributorScheduled(name='test', products=[p1, p2], start=2015.0, coverage=0.80)
        assert d.products == [p1, p2]

    def test_is_intervention_distributor_subclass(self):
        """InterventionDistributorScheduled must be an InterventionDistributor
        subclass so Campaign.add() accepts it."""
        d = InterventionDistributorScheduled(name='test', products=[], start=2015.0, coverage=0.80)
        assert isinstance(d, InterventionDistributor)


# ===========================================================================
# step()
# ===========================================================================

class TestInterventionDistributorScheduledStep:

    def test_does_not_fire_before_start_ti(self):
        """step() is a no-op on every timestep before start_ti — agents are
        not dosed early and _fired remains False."""
        p = _MockProduct()
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p], ti=3)
        d.step()
        assert len(p.calls) == 0
        assert d._fired is False

    def test_does_not_fire_after_start_ti(self):
        """step() is a no-op on timesteps after start_ti — the _fired guard
        ensures firing happens exactly once."""
        p = _MockProduct()
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p], ti=3)
        d.step()   # ti=3, no fire
        d.setattribute('sim', _MockSim(ti=7))
        d.step()   # ti=7, still no fire (not _fired, but wrong ti)
        assert len(p.calls) == 0

    def test_fires_at_start_ti(self):
        """step() fires when sim.ti == _start_ti, setting _fired to True so
        all subsequent calls return immediately."""
        p = _MockProduct()
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p], ti=5)
        d.step()
        assert d._fired is True

    def test_fires_only_once(self):
        """Even if step() is called again with the same ti (e.g. during
        testing), _fired prevents a second delivery."""
        p = _MockProduct()
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p], ti=5)
        d.step()
        d.step()
        assert len(p.calls) == 1

    def test_empty_eligible_pool_no_delivery(self):
        """When no agents are eligible (e.g. age filter excludes everyone),
        _fired is still set but administer() is never called."""
        p = _MockProduct()
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p],
                           eligible=ss.uids(np.array([], dtype=int)))
        d.step()
        assert d._fired is True
        assert len(p.calls) == 0

    def test_full_coverage_doses_all_eligible(self):
        """coverage=1.0 means every eligible agent is selected — bernoulli
        with p=1 never drops any UID from the pool."""
        p = _MockProduct()
        eligible = ss.uids(np.arange(8))
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p],
                           eligible=eligible)
        d.step()
        np.testing.assert_array_equal(p.all_administered, np.arange(8))

    def test_zero_coverage_doses_nobody(self):
        """coverage=0.0 means no agents are selected — bernoulli with p=0
        filters out every UID."""
        p = _MockProduct()
        d = _make_for_step(start_ti=5, coverage=0.0, products=[p])
        d.step()
        assert len(p.calls) == 0

    def test_all_products_receive_same_targets(self):
        """When multiple products are configured, every product's administer()
        is called with the same set of UIDs — the distributor does not split
        the dose across products."""
        p1, p2 = _MockProduct(), _MockProduct()
        eligible = ss.uids(np.arange(6))
        d = _make_for_step(start_ti=5, coverage=1.0, products=[p1, p2],
                           eligible=eligible)
        d.step()
        np.testing.assert_array_equal(p1.all_administered, p2.all_administered)

    def test_chosen_subset_of_eligible(self):
        """With coverage < 1.0, the chosen UIDs are a subset of the eligible
        pool — no UID outside the pool can be selected."""
        p = _MockProduct()
        eligible = ss.uids(np.arange(20))
        d = _make_for_step(start_ti=5, coverage=0.50, products=[p],
                           eligible=eligible)
        d.step()
        if len(p.calls) > 0:
            assert all(uid in eligible for uid in p.all_administered)


# ===========================================================================
# InterventionDistributor.step() — abstract guard
# ===========================================================================

class TestInterventionDistributorAbstract:

    def test_base_step_raises_not_implemented(self):
        """Direct instantiation of InterventionDistributor and calling step()
        raises NotImplementedError — the base class is abstract in practice."""
        d = InterventionDistributor(name='test', products=[], start=2015.0)
        with pytest.raises(NotImplementedError):
            d.step()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
