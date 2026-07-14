"""
Tests for Campaign.

Campaign validates distributor start years at add() time using the sim's
yearvec, and returns a flat start-year-sorted list from as_interventions().

A _MockSim with a simple yearvec stands in for ss.Sim — Campaign only reads
sim.t.yearvec.  InterventionDistributor instances are constructed with an
empty products list; Campaign.add() checks isinstance and start year only.
"""

import numpy as np
import pytest
import starsim as ss

from malariasim.campaign import Campaign
from malariasim.distributors.scheduled import (
    InterventionDistributor,
    InterventionDistributorScheduled,
)
from tests.mock_objects import MockSim as _MockSim


def _dist(start, name='dist', coverage=0.80):
    """Construct a minimal InterventionDistributorScheduled with the given start value."""
    return InterventionDistributorScheduled(name=name, products=[], start=start, coverage=coverage)


# ===========================================================================
# Construction
# ===========================================================================

class TestCampaignConstruction:

    def test_empty_campaign_constructs(self):
        """A Campaign with no items is valid — a scenario with no
        interventions is a legitimate control arm."""
        sim = _MockSim()
        c = Campaign(sim)
        assert c.as_interventions() == []

    def test_items_list_in_constructor(self):
        """Distributors passed as the items list are added in order, with the
        same validation as explicit add() calls."""
        sim = _MockSim()
        d1 = _dist(2012.0, name='d1')
        d2 = _dist(2015.0, name='d2')
        c = Campaign(sim, items=[d1, d2])
        assert len(c.as_interventions()) == 2

    def test_constructor_items_validated(self):
        """Items in the constructor list go through the same start-year check
        as items passed to add() — there is no bypass path."""
        sim = _MockSim(start=2010, stop=2020)
        with pytest.raises(ValueError, match='outside'):
            Campaign(sim, items=[_dist(2025.0)])


# ===========================================================================
# Campaign.add()
# ===========================================================================

class TestCampaignAdd:

    def test_valid_distributor_added(self):
        """A distributor whose start year is within the sim range is accepted
        and appears in as_interventions()."""
        sim = _MockSim()
        d = _dist(2015.0)
        c = Campaign(sim)
        c.add(d)
        assert d in c.as_interventions()

    def test_start_at_sim_start_boundary(self):
        """Start year equal to sim.t.yearvec[0] is valid — the lower bound is
        inclusive so a campaign on the very first timestep is allowed."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        c.add(_dist(2010.0))

    def test_start_at_sim_stop_boundary(self):
        """Start year equal to sim.t.yearvec[-1] is valid — the upper bound is
        inclusive so a campaign on the final timestep is allowed."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        c.add(_dist(2020.0))

    def test_start_before_sim_start_raises(self):
        """A start year before the simulation start is a configuration error —
        the distributor would never fire and the user almost certainly made a
        mistake specifying years."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        with pytest.raises(ValueError, match='outside'):
            c.add(_dist(2005.0))

    def test_start_after_sim_stop_raises(self):
        """A start year after the simulation end is a configuration error —
        same reasoning as start before sim start."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        with pytest.raises(ValueError, match='outside'):
            c.add(_dist(2025.0))

    def test_error_message_contains_start_year(self):
        """The ValueError names the offending start year so the user can
        identify which distributor is misconfigured without a debugger."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        with pytest.raises(ValueError, match='2030'):
            c.add(_dist(2030.0))

    def test_non_distributor_raises_type_error_naming_received_type(self):
        """Passing an object that is not an InterventionDistributor subclass
        raises TypeError naming the received type so the user can identify the
        wrong object without reading the source."""
        sim = _MockSim()
        c = Campaign(sim)
        with pytest.raises(TypeError, match='dict'):
            c.add({'start': 2015.0})

    def test_ss_timepar_start_accepted(self):
        """ss.date (an ss.TimePar) is accepted as a start value alongside
        plain floats — both are converted to years for comparison."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        c.add(_dist(ss.date(2015)))

    def test_ss_timepar_start_out_of_range_raises(self):
        """An out-of-range ss.date raises ValueError just as a plain float
        would — the TimePar path goes through the same conversion."""
        sim = _MockSim(start=2010, stop=2020)
        c = Campaign(sim)
        with pytest.raises(ValueError, match='outside'):
            c.add(_dist(ss.date(2025)))


# ===========================================================================
# Campaign.as_interventions()
# ===========================================================================

class TestCampaignAsInterventions:

    def test_empty_campaign_returns_empty_list(self):
        """as_interventions() on a campaign with no distributors returns []
        rather than raising — ms.Sim handles the empty case gracefully."""
        assert Campaign(_MockSim()).as_interventions() == []

    def test_single_distributor_returned(self):
        """A single distributor is returned in a one-element list."""
        sim = _MockSim()
        d = _dist(2015.0)
        c = Campaign(sim, items=[d])
        assert c.as_interventions() == [d]

    def test_sorted_ascending_by_start_year(self):
        """Distributors are returned sorted by start year ascending so that
        early campaigns appear first — consistent with chronological order."""
        sim = _MockSim()
        d1 = _dist(2018.0, name='d1')
        d2 = _dist(2012.0, name='d2')
        d3 = _dist(2015.0, name='d3')
        result = Campaign(sim, items=[d1, d2, d3]).as_interventions()
        years = [d.start for d in result]
        assert years == sorted(years)

    def test_already_sorted_unchanged(self):
        """A list already in ascending order is returned in the same order —
        the sort is stable with respect to insertion order for equal keys."""
        sim = _MockSim()
        d1 = _dist(2011.0, name='d1')
        d2 = _dist(2015.0, name='d2')
        d3 = _dist(2019.0, name='d3')
        result = Campaign(sim, items=[d1, d2, d3]).as_interventions()
        assert result == [d1, d2, d3]

    def test_returns_all_distributors(self):
        """as_interventions() returns every distributor that was added —
        nothing is dropped during sorting."""
        sim = _MockSim()
        dists = [_dist(float(y), name=f'd{i}') for i, y in enumerate([2014, 2011, 2018, 2016])]
        c = Campaign(sim, items=dists)
        assert set(c.as_interventions()) == set(dists)

    def test_duplicate_names_raises_value_error(self):
        """as_interventions() raises ValueError when two distributors share a
        name — the error names the duplicates so the user can identify them."""
        sim = _MockSim()
        d1 = _dist(2012.0, name='same_name')
        d2 = _dist(2015.0, name='same_name')
        c = Campaign(sim, items=[d1, d2])
        with pytest.raises(ValueError, match='same_name'):
            c.as_interventions()

    def test_duplicate_name_error_includes_start_and_product(self):
        """The duplicate-name error message includes the start year and
        product type of each conflicting distributor so the user can
        identify which entry to rename."""
        sim = _MockSim()
        d1 = _dist(2012.0, name='dup')
        d2 = _dist(2015.0, name='dup')
        c = Campaign(sim, items=[d1, d2])
        with pytest.raises(ValueError, match='2012') as exc_info:
            c.as_interventions()
        assert '2015' in str(exc_info.value)

    def test_returns_copy_not_internal_list(self):
        """Mutating the returned list does not affect the Campaign — callers
        cannot accidentally corrupt the campaign's internal state."""
        sim = _MockSim()
        d = _dist(2015.0)
        c = Campaign(sim, items=[d])
        result = c.as_interventions()
        result.clear()
        assert len(c.as_interventions()) == 1


# ===========================================================================
# Campaign.add() — validate_eligibility
# ===========================================================================

class _RaisingDisease:
    """Mock disease whose validate_eligibility always raises."""
    _supports_intervention_effects = True

    def validate_eligibility(self, eligibility):
        raise ValueError("mock: misaligned age bound")


class TestCampaignValidateEligibility:

    def test_no_disease_in_sim_no_error(self):
        """When _thin_sim has no diseases, Campaign.add() skips validation —
        backward-compatible with tests that use a plain MockSim."""
        sim = _MockSim()  # diseases=[] by default
        c = Campaign(sim)
        c.add(_dist(2015.0))  # should not raise

    def test_abm_disease_no_error(self):
        """A disease with a no-op validate_eligibility (ABM) never raises,
        regardless of the eligibility's age bounds."""
        from tests.mock_objects import MockDisease
        from malariasim.distributors.eligibility import HumanEligibility
        sim = _MockSim(diseases=[MockDisease(prefix='vaccine')])
        c = Campaign(sim)
        c.add(InterventionDistributorScheduled(
            name='v', products=[], start=2015.0, coverage=0.80,
            eligibility=HumanEligibility(min_age=ss.years(7), max_age=ss.years(13)),
        ))

    def test_disease_validate_eligibility_called_and_error_propagated(self):
        """When validate_eligibility raises, Campaign.add() propagates the
        ValueError immediately — the error points at the add() call."""
        from malariasim.distributors.eligibility import HumanEligibility
        sim = _MockSim(diseases=[_RaisingDisease()])
        c = Campaign(sim)
        with pytest.raises(ValueError, match='misaligned'):
            c.add(InterventionDistributorScheduled(
                name='v', products=[], start=2015.0, coverage=0.80,
                eligibility=HumanEligibility(min_age=ss.years(5)),
            ))

    def test_no_eligibility_on_distributor_skips_validation(self):
        """A distributor with no eligibility (e.g. Outbreak with no filter)
        does not trigger validate_eligibility — nothing to validate."""
        sim = _MockSim(diseases=[_RaisingDisease()])
        c = Campaign(sim)
        c.add(_dist(2015.0))  # no eligibility — should not raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
