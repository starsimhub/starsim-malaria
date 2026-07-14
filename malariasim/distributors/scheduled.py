"""
scheduled.py — InterventionDistributor and InterventionDistributorScheduled.

InterventionDistributor
-----------------------
Abstract base class for all malaria intervention distributors.  Holds a list
of products and an eligibility strategy.  Subclasses decide when and how many
targets receive the intervention.

``_deliver_to(targets)`` calls ``product.administer(targets)`` on every
product, using Starsim's standard ``administer()`` vocabulary.

InterventionDistributorScheduled
---------------------------------
Fires exactly once at the specified ``start`` time.  Uses
``ss.bernoulli(p=coverage)`` to sample from the eligible pool, consistent
with Starsim's probabilistic coverage model.

Multiple ``InterventionDistributorScheduled`` instances can share the same
Product — the Product's state arrays are overwritten on each dose, enforcing
the one-active-instance-per-type-per-agent rule without any extra tracking.
"""

import numpy as np
import sciris as sc
import starsim as ss


class InterventionDistributor(ss.Intervention):
    """
    Abstract base class for malaria intervention distributors.

    Parameters
    ----------
    name : str
        Unique name for this distributor.  Required — starsim uses ``name``
        as the ndict key, so two distributors in the same sim must have
        different names.  Choose something descriptive, e.g.
        ``'vaccine_2005_u5'``.
    products : ss.Product or list of ss.Product
        One or more products to deliver to each selected target.
    start : ss.date or float
        Simulation time at or after which the distributor may fire.
        Used by subclasses to resolve the firing timestep in ``init_pre()``.
    eligibility : Eligibility or callable, optional
        Strategy that returns the pool of eligible target IDs (human UIDs
        or node IDs) when called with ``sim``.  Defaults to all alive agents
        when omitted.
    """

    def __init__(self, name, products, start, eligibility=None, **kwargs):
        super().__init__(name=name, eligibility=eligibility, **kwargs)
        self.products = products if isinstance(products, list) else [products]
        self.start    = start

    def init_pre(self, sim):
        """Initialise each product against the sim."""
        super().init_pre(sim)
        for product in self.products:
            product.init_pre(sim)

    def _resolve_targets(self, coverage):
        """Return ``(uids, weights)`` for the given coverage level.

        Two-step: eligibility strategy returns the pool of candidate UIDs,
        then ``disease.apply_coverage()`` samples coverage from that pool.
        When ``eligibility=None``, a bare ``HumanEligibility()`` targets all
        alive agents / compartments.
        """
        from malariasim.distributors.eligibility import HumanEligibility
        if self.eligibility is not None:
            eligible = self.eligibility(self.sim)
        else:
            eligible = HumanEligibility()(self.sim)
        if len(eligible) == 0:
            return ss.uids(np.array([], dtype=int)), np.array([], dtype=float)
        disease = HumanEligibility._find_disease(self.sim)
        return disease.apply_coverage(eligible, coverage)

    def _deliver_to(self, targets, weights):
        """Call ``administer(targets, weights)`` on every product."""
        for product in self.products:
            product.administer(targets, weights)

    def step(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement step(). "
            "Use InterventionDistributorScheduled or another concrete subclass."
        )


class InterventionDistributorScheduled(InterventionDistributor):
    """
    One-shot intervention distributor that fires exactly once at ``start``.

    Uses ``ss.bernoulli(p=coverage)`` to sample from the eligible pool,
    consistent with Starsim's probabilistic coverage model.  After firing,
    the distributor sets an internal ``_fired`` flag and returns immediately
    on all subsequent steps.

    Parameters
    ----------
    name : str
        Unique name for this distributor.  Required — see
        ``InterventionDistributor`` for details.
    products : ss.Product or list of ss.Product
        Products to deliver.
    start : ss.date or float
        Simulation time at which to fire.  Resolved to a timestep index in
        ``init_pre()`` via ``sc.findnearest(sim.t.yearvec, start_year)``.
    eligibility : Eligibility or callable, optional
        Pool of eligible targets.  Defaults to all alive agents.
    coverage : float in (0, 1], optional
        Fraction of eligible targets to deliver to, sampled via
        ``ss.bernoulli``.  Mutually exclusive with ``num_targeted``; exactly
        one must be supplied.
    num_targeted : int, optional
        Exact number of targets to select.  **Not yet implemented** —
        providing this raises ``NotImplementedError`` at construction time.

    Examples
    --------
    Vaccinate 80 % of children under five in 2015::

        ms.InterventionDistributorScheduled(
            name        = 'vacc_2015_u5',
            products    = ms.MalariaVaccine(waning=ms.WaningEffect(...)),
            start       = ss.date(2015),
            eligibility = ms.HumanEligibility(
                              min_age=ss.years(0.5), max_age=ss.years(5)),
            coverage    = 0.80,
        )

    Two campaigns sharing one vaccine definition::

        vaccine = ms.MalariaVaccine(waning=ms.WaningEffect(...))
        ms.InterventionDistributorScheduled(
            name='vacc_2015', products=vaccine, start=ss.date(2015), coverage=0.50)
        ms.InterventionDistributorScheduled(
            name='vacc_2020', products=vaccine, start=ss.date(2020), coverage=0.90)
    """

    def __init__(self, name, products, start, eligibility=None,
                 coverage=None, num_targeted=None, **kwargs):
        super().__init__(name, products, start, eligibility=eligibility, **kwargs)

        if (coverage is None) == (num_targeted is None):
            raise ValueError(
                "Specify exactly one of 'coverage' or 'num_targeted'."
            )
        if num_targeted is not None:
            raise NotImplementedError(
                "'num_targeted' is not yet implemented. Use 'coverage' instead."
            )

        self.coverage      = coverage
        self.num_targeted  = num_targeted
        self._fired        = False
        self._start_ti     = None   # resolved in init_pre()

    def init_pre(self, sim):
        """Resolve ``start`` to a timestep index."""
        super().init_pre(sim)
        start_year = (
            self.start.years
            if isinstance(self.start, ss.TimePar)
            else float(self.start)
        )
        self._start_ti = sc.findnearest(sim.t.yearvec, start_year)

    def step(self):
        """
        Fire once when the simulation reaches ``start``.

        Finds eligible targets via ``check_eligibility()``, samples the
        configured coverage fraction using ``ss.bernoulli``, and calls
        ``_deliver_to()`` on the selected subset.  Returns immediately on
        all subsequent steps.
        """
        if self._fired or self.sim.ti != self._start_ti:
            return
        self._fired = True

        uids, weights = self._resolve_targets(self.coverage)
        if len(uids):
            self._deliver_to(uids, weights)
