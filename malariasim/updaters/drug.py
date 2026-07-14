"""
drug.py — AntiMalarialDrug product.

AntiMalarialDrug
----------------
A ``ss.Product`` subclass that clears active infections by overwriting the
scheduled recovery time (ABM) or applying an ODE clearance rate (ODE).

The product has no paired Updater — the ABM delegates to the existing SIS
recovery loop and the ODE handles clearance inline in its own ``step()``.

Usage::

    drug = AntiMalarialDrug(mean_duration=ss.days(1.5))

Distribution choice: exponential
---------------------------------
Observed parasite clearance times are right-skewed (most infections clear in
1–2 days, long tail to ~3 days).  A gamma or log-normal with mean ~1.5 days
fits the data better, but at 1-day timesteps the discretisation swamps the
distributional difference.  The exponential is used because it is right-skewed,
its mean maps exactly to the ODE clearance rate ``r = 1/mean_duration``, and
one parameter is simpler than two.
"""

import numpy as np
import starsim as ss


class AntiMalarialDrug(ss.Product):
    """
    Antimalarial drug product that clears active infections.

    Calls ``disease.set_drug_clearance(uids, mean_duration_years, weights)``
    on ``administer()``.  The disease module implements the mechanism:

    - ABM (``MalariaHumanBase``): draws exponential clearance durations and
      overwrites ``ti_recovered`` so the SIS loop clears the infection.
    - ODE (``MalariaHumanODE``): stores ``drug_coverage`` and
      ``drug_clear_end_t``; ``step()`` applies clearance rate each timestep.

    Susceptible and exposed agents are no-ops in both cases.

    Parameters
    ----------
    mean_duration : ss.days
        Exponential mean clearance time.  Default: ``ss.days(1.5)`` (~36 hours),
        consistent with observed artemisinin-based therapy clearance times.
    """

    def __init__(self, mean_duration=ss.days(1.5), name=None):
        super().__init__(name=name)
        self.mean_duration = mean_duration

    def init_pre(self, sim):
        super().init_pre(sim)
        self._disease = next(
            (m for m in self.sim.diseases.values()
             if hasattr(m, '_supports_intervention_effects')),
            None,
        )
        if self._disease is None:
            raise RuntimeError(
                'AntiMalarialDrug: no disease module with '
                "'_supports_intervention_effects' found. "
                'Ensure MalariaHumanBase (or a subclass) is registered as the '
                'human disease module.'
            )

    def administer(self, uids, weights=None):
        """
        Administer the drug to ``uids``.

        Parameters
        ----------
        uids : array-like
            UIDs of agents (ABM) or compartment UIDs (ODE) receiving the drug.
        weights : ndarray or None
            Per-uid coverage weights.  ABM callers pass None (defaults to
            all-ones); ODE callers pass the coverage fraction from
            ``apply_coverage``.
        """
        uids = ss.uids(uids) if not isinstance(uids, ss.uids) else uids
        if weights is None:
            weights = np.ones(len(uids), dtype=float)
        self._disease.set_drug_clearance(uids, self.mean_duration.years, weights)
