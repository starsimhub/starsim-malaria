"""
campaign.py — Campaign container for malaria intervention distributors.

Campaign is created inside build_campaign(sim) and returned to ms.Sim.
Passing sim into the builder lets Campaign.add() validate distributor start
years immediately — the error points directly to the add() call rather than
surfacing later in as_interventions() where the offending item is harder to
identify.

ms.Sim calls campaign.as_interventions() after build_campaign() returns to
get the flat, start-year-sorted list of InterventionDistributor instances to
register with ss.Sim.

Typical build_campaign pattern
-------------------------------
    def build_campaign(sim, eir_only=False):
        vaccine = ms.MalariaVaccine(waning=ms.WaningEffect(...))
        campaign = ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'vacc_2015_u5',
                products    = vaccine,
                start       = ss.date(2015),
                eligibility = ms.HumanEligibility(min_age=ss.years(0.5),
                                                  max_age=ss.years(5)),
                coverage    = 0.80,
            ),
        ])
        return campaign

    sim = ms.Sim(..., build_campaign=build_campaign)

# TODO: InterventionProgram — named delivery strategies (e.g.
# SeasonalMalariaChemoprevention) that compose multiple distributors into a
# reusable, domain-specific object.  InterventionProgram.as_interventions()
# returns a list of InterventionDistributor instances; Campaign.add() and
# Campaign.as_interventions() will flatten them alongside bare distributors.
"""

import starsim as ss

from .distributors.scheduled import InterventionDistributor


def _to_year(start):
    """Convert an ss.TimePar or plain float to a float year."""
    return start.years if isinstance(start, ss.TimePar) else float(start)


class Campaign:
    """
    Container for all intervention distributors in a scenario.

    Created inside ``build_campaign(sim)`` and returned to ``ms.Sim``.
    Validates each distributor's start year against the simulation date
    range at ``add()`` time so configuration errors surface immediately at
    the line that caused them.

    Parameters
    ----------
    sim : ss.Sim
        Simulation object.  Used at ``add()`` time to check that each
        distributor's start year falls within ``[sim.t.yearvec[0],
        sim.t.yearvec[-1]]``.
    items : list of InterventionDistributor, optional
        Initial distributors to add.  Equivalent to calling ``add()`` on
        each item after construction; the same validation applies.
    """

    def __init__(self, sim, items=None):
        self._sim   = sim
        self._items = []
        if items is not None:
            for item in items:
                self.add(item)

    def add(self, item):
        """
        Add a distributor to the campaign.

        Validates immediately that ``item`` is an ``InterventionDistributor``
        and that its ``start`` year lies within the simulation date range.

        Parameters
        ----------
        item : InterventionDistributor
            The distributor to add.

        Raises
        ------
        TypeError
            If ``item`` is not an ``InterventionDistributor``.
        ValueError
            If ``item.start`` falls outside ``[sim.t.start, sim.t.stop]``.
        """
        if not isinstance(item, InterventionDistributor):
            raise TypeError(
                f"Campaign.add() expects an InterventionDistributor, "
                f"got {type(item).__name__}."
            )
        yearvec    = self._sim.t.yearvec
        sim_start  = yearvec[0]
        sim_stop   = yearvec[-1]
        start_year = _to_year(item.start)
        if not (sim_start <= start_year <= sim_stop):
            raise ValueError(
                f"Distributor start year {start_year} is outside the "
                f"simulation date range [{sim_start:.1f}, {sim_stop:.1f}]."
            )
        disease = next(
            (d for d in getattr(self._sim, 'diseases', [])
             if hasattr(d, '_supports_intervention_effects')),
            None,
        )
        if disease is not None and getattr(item, 'eligibility', None) is not None:
            disease.validate_eligibility(item.eligibility)
        self._items.append(item)

    def as_interventions(self):
        """
        Return a flat list of distributors sorted by start year (ascending).

        Called by ``ms.Sim`` after ``build_campaign()`` returns.  Sorting
        ensures distributors that fire earliest are processed first, which
        is convenient for any future optimisations that skip already-fired
        distributors.

        Raises
        ------
        ValueError
            If any two distributors share the same ``name``.  The error message
            names the conflicting distributors (name, start year, products) so
            the user can identify and rename the duplicates.

        Returns
        -------
        list of InterventionDistributor
        """
        distributors = list(self._items)
        distributors.sort(key=lambda d: _to_year(d.start))

        seen: dict[str, 'InterventionDistributor'] = {}
        duplicates: list[str] = []
        for d in distributors:
            if d.name in seen:
                if d.name not in duplicates:
                    duplicates.append(d.name)
            else:
                seen[d.name] = d

        if duplicates:
            lines = []
            for d in distributors:
                if d.name in duplicates:
                    product_names = [type(p).__name__ for p in d.products]
                    lines.append(
                        f"  name={d.name!r}  start={_to_year(d.start):.4g}"
                        f"  products={product_names}"
                    )
            raise ValueError(
                "Campaign.as_interventions(): duplicate distributor names found. "
                "Each distributor must have a unique name (starsim requires unique "
                "module names within a sim).\n"
                "Conflicting distributors:\n" + "\n".join(lines)
            )

        return distributors
