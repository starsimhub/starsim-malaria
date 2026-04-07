"""
MalariaHumanSIS: SIS malaria disease module for human agents.

Thin subclass of MalariaHumanBase: susceptible → infected → susceptible.
No incubation period.
"""

from .human_base import MalariaHumanBase


class MalariaHumanSIS(MalariaHumanBase):
    """
    SIS malaria disease module.

    Susceptible agents are infected by mosquito biting (EIR-driven). After a
    Gaussian-distributed infectious period they return directly to susceptible.

    Parameters are identical to MalariaHumanBase.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.update_pars(pars, **kwargs)
        # Point dur_inf at infection_duration_years so user overrides are respected
        self.pars.dur_inf = self.pars.infection_duration_years
        return
