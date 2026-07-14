"""
Tests for HumanEligibility and NodeEligibility.

Eligibility objects are callables: eligibility(sim) → ss.uids.
No build_campaign() window is needed — these are pure data/strategy objects
with no Updater interaction.

Mocks replace ss.Sim, ss.People, and NodeState since eligibility only needs
sim.people.auids, sim.people.age[uids], and sim.connectors.
"""

import numpy as np
import pytest
import starsim as ss

from malariasim.distributors.eligibility import (
    Eligibility,
    HumanEligibility,
    NodeEligibility,
)
from tests.mock_objects import (
    MockPeople as _MockPeople,
    MockGrid as _MockGrid,
    MockNodeState as _MockNodeState,
    MockSim as _MockSim,
)


# ===========================================================================
# Eligibility base class
# ===========================================================================

class TestEligibilityBase:

    def test_call_raises_not_implemented(self):
        """The base class is abstract — instantiating and calling it must
        raise NotImplementedError so that incomplete subclasses are caught
        at call time rather than silently returning None."""
        e = Eligibility()
        with pytest.raises(NotImplementedError):
            e(_MockSim(ages=[1.0]))


# ===========================================================================
# HumanEligibility
# ===========================================================================

class TestHumanEligibility:

    def test_no_filter_returns_all_alive(self):
        """With no age bounds every alive agent is eligible regardless of age,
        so the result is exactly sim.people.auids."""
        sim = _MockSim(ages=[0.5, 5.0, 25.0, 60.0])
        result = HumanEligibility()(sim)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_min_age_excludes_younger_agents(self):
        """Agents strictly below min_age must not appear in the result.
        min_age=ss.years(5) means a 4-year-old is excluded."""
        sim = _MockSim(ages=[1.0, 4.9, 5.0, 10.0])
        result = HumanEligibility(min_age=ss.years(5))(sim)
        np.testing.assert_array_equal(result, [2, 3])

    def test_max_age_excludes_older_agents(self):
        """Agents at or above max_age must not appear in the result.
        max_age=ss.years(5) means a 5-year-old is excluded (exclusive bound)."""
        sim = _MockSim(ages=[1.0, 4.9, 5.0, 10.0])
        result = HumanEligibility(max_age=ss.years(5))(sim)
        np.testing.assert_array_equal(result, [0, 1])

    def test_min_age_is_inclusive(self):
        """An agent whose age equals min_age exactly must be included —
        the lower bound is closed."""
        sim = _MockSim(ages=[5.0])
        result = HumanEligibility(min_age=ss.years(5))(sim)
        np.testing.assert_array_equal(result, [0])

    def test_max_age_is_exclusive(self):
        """An agent whose age equals max_age exactly must be excluded —
        the upper bound is open."""
        sim = _MockSim(ages=[5.0])
        result = HumanEligibility(max_age=ss.years(5))(sim)
        assert len(result) == 0

    def test_both_bounds_together(self):
        """Only agents within [min_age, max_age) are eligible; those outside
        on either end are excluded."""
        sim = _MockSim(ages=[0.4, 0.5, 4.9, 5.0, 10.0])
        result = HumanEligibility(min_age=ss.years(0.5), max_age=ss.years(5))(sim)
        np.testing.assert_array_equal(result, [1, 2])

    def test_dead_agents_excluded(self):
        """auids only contains alive agents, so dead agents never appear in
        the eligibility pool regardless of their age."""
        alive_mask = np.array([True, False, True, False])
        sim = _MockSim(ages=[1.0, 2.0, 3.0, 4.0], alive_mask=alive_mask)
        result = HumanEligibility()(sim)
        np.testing.assert_array_equal(result, [0, 2])

    def test_dead_agents_excluded_with_age_filter(self):
        """Age filtering operates only on alive agents — a dead agent that
        would otherwise pass the age filter must still be excluded."""
        alive_mask = np.array([True, False, True])
        sim = _MockSim(ages=[2.0, 2.0, 2.0], alive_mask=alive_mask)
        result = HumanEligibility(min_age=ss.years(1), max_age=ss.years(5))(sim)
        np.testing.assert_array_equal(result, [0, 2])

    def test_empty_population_returns_empty(self):
        """An empty sim returns an empty uid array, not an error."""
        sim = _MockSim(ages=[])
        result = HumanEligibility()(sim)
        assert len(result) == 0

    def test_all_filtered_returns_empty(self):
        """When every alive agent is outside the age window the result is
        an empty uid array — no exception should be raised."""
        sim = _MockSim(ages=[20.0, 30.0])
        result = HumanEligibility(min_age=ss.years(0.5), max_age=ss.years(5))(sim)
        assert len(result) == 0

    def test_returns_ss_uids(self):
        """The result must be ss.uids so it is compatible with
        ss.bernoulli.filter() in InterventionDistributorScheduled."""
        sim = _MockSim(ages=[1.0, 2.0])
        result = HumanEligibility()(sim)
        assert isinstance(result, ss.uids)

    def test_min_age_greater_than_max_age_raises(self):
        """min_age >= max_age is a configuration error — the eligible window
        would be empty or inverted, which is almost certainly a mistake."""
        with pytest.raises(ValueError, match='min_age'):
            HumanEligibility(min_age=ss.years(10), max_age=ss.years(5))

    def test_min_age_equal_to_max_age_raises(self):
        """min_age == max_age produces a zero-width window that can never
        match any agent (max_age is exclusive), so it is also rejected."""
        with pytest.raises(ValueError, match='min_age'):
            HumanEligibility(min_age=ss.years(5), max_age=ss.years(5))

    def test_negative_min_age_raises(self):
        """Age cannot be negative — a negative min_age is a configuration
        error that should be caught at construction time."""
        with pytest.raises(ValueError, match='min_age'):
            HumanEligibility(min_age=ss.years(-1))

    def test_negative_max_age_raises(self):
        """A negative max_age would exclude every agent (all ages >= 0),
        which is almost certainly a mistake rather than intent."""
        with pytest.raises(ValueError, match='max_age'):
            HumanEligibility(max_age=ss.years(-0.5))


# ===========================================================================
# NodeEligibility
# ===========================================================================

class TestNodeEligibility:

    def test_default_returns_all_nodes(self):
        """node_ids=None (the default) returns every node ID from NodeState,
        so a node-level intervention targets the full grid by default."""
        sim = _MockSim(n_nodes=4)
        result = NodeEligibility()(sim)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_explicit_node_ids_returned(self):
        """When node_ids is provided the result is exactly that list, even
        if NodeState has more nodes — the user controls the subset."""
        sim = _MockSim(n_nodes=5)
        result = NodeEligibility(node_ids=[0, 2, 4])(sim)
        np.testing.assert_array_equal(result, [0, 2, 4])

    def test_returns_ss_uids(self):
        """The result must be ss.uids so it is compatible with
        ss.bernoulli.filter() alongside HumanEligibility results."""
        sim = _MockSim(n_nodes=3)
        result = NodeEligibility()(sim)
        assert isinstance(result, ss.uids)

    def test_explicit_node_ids_returns_ss_uids(self):
        """An explicit node_ids list is also returned as ss.uids, not a
        plain list or ndarray."""
        sim = _MockSim(n_nodes=5)
        result = NodeEligibility(node_ids=[1, 3])(sim)
        assert isinstance(result, ss.uids)

    def test_no_node_state_raises(self):
        """If no NodeState connector is in the sim, NodeEligibility raises
        RuntimeError with a helpful message rather than returning an empty
        result that would silently dose zero nodes."""
        sim = _MockSim(ages=[1.0])   # no n_nodes → no NodeState connector
        with pytest.raises(RuntimeError, match='NodeState'):
            NodeEligibility()(sim)

    def test_no_node_state_raises_even_with_explicit_ids(self):
        """When explicit node_ids are provided but no NodeState exists, a
        RuntimeError is still raised — validation requires NodeState to be
        present so invalid IDs cannot silently pass through."""
        sim = _MockSim(ages=[1.0])   # no NodeState
        with pytest.raises(RuntimeError, match='NodeState'):
            NodeEligibility(node_ids=[0, 1])(sim)

    def test_invalid_node_id_raises(self):
        """A node ID outside the range 0..n_nodes-1 raises ValueError with
        the offending IDs named — silent pass-through would cause downstream
        array index errors that are much harder to diagnose."""
        sim = _MockSim(n_nodes=3)   # valid IDs: 0, 1, 2
        with pytest.raises(ValueError, match='node_ids'):
            NodeEligibility(node_ids=[0, 5])(sim)

    def test_invalid_node_id_error_names_bad_ids(self):
        """The ValueError message includes the specific invalid IDs so the
        user can correct the mistake without inspecting NodeState manually."""
        sim = _MockSim(n_nodes=3)
        with pytest.raises(ValueError, match='5'):
            NodeEligibility(node_ids=[0, 5])(sim)

    def test_all_invalid_node_ids_raises(self):
        """All IDs invalid (none in NodeState) raises ValueError, not a
        silent empty result."""
        sim = _MockSim(n_nodes=2)   # valid IDs: 0, 1
        with pytest.raises(ValueError, match='node_ids'):
            NodeEligibility(node_ids=[3, 4, 5])(sim)

    def test_single_node_sim(self):
        """A single-node sim (the default when NodeGrid is omitted) returns
        exactly [0]."""
        sim = _MockSim(n_nodes=1)
        result = NodeEligibility()(sim)
        np.testing.assert_array_equal(result, [0])

    def test_explicit_empty_node_ids(self):
        """An explicit empty node_ids list returns an empty ss.uids — the
        distributor will then dose zero nodes, which is a valid no-op."""
        sim = _MockSim(n_nodes=3)
        result = NodeEligibility(node_ids=[])(sim)
        assert isinstance(result, ss.uids)
        assert len(result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
