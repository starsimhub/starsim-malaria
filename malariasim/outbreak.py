"""
outbreak.py — Outbreak product for seeding malaria infections.

Outbreak is a ss.Product that infects targeted agents immediately when
administered.  Used with InterventionDistributorScheduled to seed infections
into a population at a specific time, node, age group, or coverage level.

Only susceptible agents are infected; already-infected agents are silently
skipped.

Typical usage
-------------
Seed 50 % of all agents at simulation start::

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed',
                products = ms.Outbreak(),
                start    = 2000,
                coverage = 0.5,
            )
        ])

Seed only node 1, 30 % coverage, starting in year 2::

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'seed_node1',
                products    = ms.Outbreak(),
                start       = 2002,
                eligibility = ms.HumanEligibility(node_ids=[1]),
                coverage    = 0.30,
            )
        ])
"""

import numpy as np
import starsim as ss


class Outbreak(ss.Product):
    """
    A product that immediately infects targeted susceptible agents.

    Calls ``MalariaHumanBase.set_prognoses()`` on delivery, which moves agents
    from susceptible to infected and schedules their recovery time.
    Already-infected and non-susceptible agents are silently skipped.

    Parameters
    ----------
    name : str, optional
        Product name passed to ``ss.Product``.
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self._disease = None

    def init_pre(self, sim):
        """Resolve the malaria disease module."""
        super().init_pre(sim)
        for mod in sim.diseases.values():
            if hasattr(mod, '_supports_intervention_effects'):
                self._disease = mod
                return
        raise RuntimeError(
            "Outbreak: no disease module with '_supports_intervention_effects' found. "
            "Ensure MalariaHumanBase (or a subclass) is the human disease module."
        )

    def administer(self, uids, weights):
        """
        Seed infections in ``uids`` via the disease module.

        Parameters
        ----------
        uids : ss.uids
            Target agents / compartments selected by the distributor.
        weights : ndarray
            Per-uid coverage weights.  For ABM agents these are 1.0 (Bernoulli
            already applied); for ODE compartments they carry the continuous
            coverage fraction.
        """
        self._disease.administer_outbreak(uids, weights)
