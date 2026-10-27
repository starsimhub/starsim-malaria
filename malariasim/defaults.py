"""
defaults.py — canonical demographic and transmission parameters for malariasim.

Centralises values that would otherwise be duplicated across run scripts,
tests, and sweep scripts.  Import from here rather than copying constants.

Age structure
-------------
``AGE_BIN_EDGES`` and ``AGE_BIN_FRACTIONS`` define a Sub-Saharan Africa-style
age pyramid with 20 five-year bins from 0 to 100 years.

Transmission settings (3-node canonical grid)
----------------------------------------------
Named constants for the three nodes used in ``run_malariasim.py``:
  URBAN_*      — high transmission, mostly indoor biting
  PERI_URBAN_* — moderate wet-season transmission
  RURAL_*      — lower transmission with a sharp seasonal peak

Helper functions
----------------
``make_steady_state_demographics()``
    Returns a fresh ``[ss.birth, ss.death]`` pair calibrated so that births
    exactly balance deaths at the initial age distribution.  Must be called
    fresh for each simulation (modules cannot be shared).

``make_age_data()``
    Returns a per-year age-distribution dataframe whose smooth density is
    consistent with the mortality rates passed to ``ss.death``,
    so agent initialisation and mortality are in demographic equilibrium.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import starsim as ss

from .demographics import compute_mortality_schedule

__all__ = ['CommonParams', 'DEFAULT_EMERGENCE', 'AGE_BIN_EDGES', 'AGE_BIN_FRACTIONS',
           'URBAN_EIR', 'PERI_URBAN_EIR', 'RURAL_EIR', 'URBAN_EMERGENCE', 'PERI_URBAN_EMERGENCE', 'RURAL_EMERGENCE',
           'URBAN_F_INDOOR', 'PERI_URBAN_F_INDOOR', 'RURAL_F_INDOOR', 'make_steady_state_demographics', 'make_age_data']


# ------------------------------------------------------------------
# Shared biological parameters
# ------------------------------------------------------------------

@dataclass
class CommonParams:
    """
    Canonical biological parameters, shared by every combination of human and mosquito models

    Give it to ``ms.Sim(common_pars=...)``, or its ``pars()`` to ``ms.Malaria(**cp.pars())``.

    Parameters
    ----------
    prob_human_infected_on_bite : float
        Probability of human infection per infectious mosquito bite (Ross-Macdonald *b*).
    prob_mosquito_infected_on_bite : float
        Probability of mosquito infection per bite on an infectious human (Ross-Macdonald *c*).
    infection_duration_mean_days : float
        Arithmetic mean of the infectious period in days.
    infection_duration_std_days : float
        Arithmetic standard deviation of the infectious period in days (log-normal; an ODE takes it as exponential).
    bites_per_female_per_day : float
        Mosquito biting rate (*a*).
    mosquito_life_expectancy_days : float
        Mean adult female lifespan (1/mu).
    mosquito_eip_days : float
        Extrinsic incubation period (1/sigma).
    """
    prob_human_infected_on_bite:       float = 0.4     # Ross-Macdonald *b*
    prob_mosquito_infected_on_bite:    float = 0.3     # Ross-Macdonald *c*
    infection_duration_mean_days:      float = 200.0   # arithmetic mean of the infectious period
    infection_duration_std_days:       float = 200.0   # arithmetic std dev
    bites_per_female_per_day:          float = 0.03    # mosquito biting rate (a)
    mosquito_life_expectancy_days:     float = 10.0    # mean adult female lifespan (1/mu)
    mosquito_eip_days:                 float = 12.0    # extrinsic incubation period (1/sigma)

    def pars(self):
        """ The parameters of ``ms.Malaria`` """
        return dict(
            prob_human_infected_on_bite    = self.prob_human_infected_on_bite,
            prob_mosquito_infected_on_bite = self.prob_mosquito_infected_on_bite,
            infection_duration             = ss.lognorm_ex(mean=ss.days(self.infection_duration_mean_days), std=ss.days(self.infection_duration_std_days)),
            bites_per_female_per_day       = ss.perday(self.bites_per_female_per_day),
            life_expectancy                = ss.days(self.mosquito_life_expectancy_days),
            eip_days                       = ss.days(self.mosquito_eip_days),
        )


# ---------------------------------------------------------------------------
# Mosquito emergence by month (new adult females per person per day), by default
# ---------------------------------------------------------------------------
DEFAULT_EMERGENCE = [0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005]


AGE_BIN_EDGES = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
]

AGE_BIN_FRACTIONS = [
    0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
    0.035, 0.03, 0.025, 0.02, 0.015, 0.01,
    0.006, 0.0036, 0.0022, 0.0013,
]

# ------------------------------------------------------------------
# Canonical 3-node transmission parameters
# ------------------------------------------------------------------

# Monthly EIR profile for each node: the annual rate (bites / person / year) at the start of each month, as ms.Malaria(eir=...) reads it, or a seasonal shape for ms.eir_to_params(seasonal_profile=...) (malariasim bug 14: was labelled per month)
URBAN_EIR      = [1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1,  1]
PERI_URBAN_EIR = [2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2]
RURAL_EIR      = [5,  5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5]

# Monthly emergence (new adult females / human / day)
URBAN_EMERGENCE      = [0.008, 0.008, 0.015, 0.030, 0.045, 0.060,
                        0.060, 0.045, 0.030, 0.015, 0.008, 0.008]
PERI_URBAN_EMERGENCE = [0.005, 0.005, 0.010, 0.020, 0.030, 0.040,
                        0.040, 0.030, 0.020, 0.010, 0.005, 0.005]
RURAL_EMERGENCE      = [0.002, 0.002, 0.005, 0.010, 0.018, 0.025,
                        0.025, 0.018, 0.010, 0.005, 0.002, 0.002]

# Fraction of bites that occur indoors
URBAN_F_INDOOR      = 0.80
PERI_URBAN_F_INDOOR = 0.70
RURAL_F_INDOOR      = 0.55

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def make_steady_state_demographics(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS):
    """
    Return a fresh ``[ss.birth, ss.death]`` demographics list calibrated for
    a stable, representative population.

    Deaths are by age group (``edges``), at the mean of the rates by single
    year of age that hold the pyramid of ``make_age_data()``, and the birth
    rate balances them, so the population neither grows nor shrinks. Being by
    the age groups, they also run as an ODE stratified by them
    (``ms.Sim(age_bin_edges=edges, solver='ode')``). This is the standard approach for malaria
    modelling, where simulations typically use a representative constant
    population rather than actual census birth/death data. Both are declared,
    so they run as agents or as an ODE.

    A new list must be created for each simulation, since modules cannot be
    shared across sims.
    """
    mu, n, _, max_age = compute_mortality_schedule(edges, fractions)
    birth_rate = float(np.dot(mu, n)/n.sum()) # Deaths per person per year, at the pyramid
    years = np.arange(max_age)
    group = np.clip(np.searchsorted(edges, years, side='right') - 1, 0, len(edges) - 2)
    mu_group = [np.average(mu[group == g], weights=n[group == g]) for g in range(len(edges) - 1)] # The mean hazard in each age group, so that it also runs as an ODE stratified by these groups
    death_rate = ss.by(age={float(e):ss.peryear(m) for e,m in zip(edges[:-1], mu_group)})
    return [ss.birth(rate=ss.peryear(birth_rate)), ss.death(rate=death_rate)]


def make_age_data(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS):
    """
    Return a per-year age-distribution dataframe for initialising agent ages.

    The smooth density n(x) is derived from the same log-linear interpolation
    used to compute per-year mortality rates, so agent initialisation and
    mortality are at the same demographic equilibrium.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age`` (int, 0-99) and ``value`` (float, relative weight).
        Suitable for ``ss.People(age_data=...)``.
    """
    _, n, _, max_age = compute_mortality_schedule(edges, fractions)
    return pd.DataFrame({'age': np.arange(max_age, dtype=int), 'value': n})
