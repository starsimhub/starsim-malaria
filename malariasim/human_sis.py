"""
MalariaHumanSIS: SIS malaria disease module for human agents.

Thin subclass of MalariaHumanBase that uses a simple SIS epidemiological model:
susceptible → infected → susceptible (no exposed/incubation period).

For a model with an explicit incubation period, use MalariaHumanSEIS.
For the shared infrastructure, see MalariaHumanBase in human_base.py.
"""

from .human_base import MalariaHumanBase


class MalariaHumanSIS(MalariaHumanBase):
    """
    SIS malaria disease module for human agents.

    Susceptible agents are infected by mosquito biting (EIR-driven). After a
    Gaussian-distributed infectious period they return directly to susceptible
    (no lasting immunity). All shared infrastructure — biting heterogeneity,
    age-dependent biting, observable states, symptom framework — is inherited
    from MalariaHumanBase.

    Parameters are identical to MalariaHumanBase; see that class for the full
    parameter reference.

    Examples
    --------
    Default parameters::

        human = MalariaHumanSIS()

    Custom transmission probabilities::

        human = MalariaHumanSIS(
            prob_human_infected_on_bite=0.4,
            prob_mosquito_infected_on_bite=0.3,
        )
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.update_pars(pars, **kwargs)
        # ss.SIS.set_prognoses draws duration from self.pars.dur_inf.
        # Point it at the same object as infection_duration_years so that
        # any user override of infection_duration_years is respected.
        self.pars.dur_inf = self.pars.infection_duration_years
        return
