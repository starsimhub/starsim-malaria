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
    Returns a fresh ``[ss.Births, ss.Deaths]`` pair calibrated so that births
    exactly balance deaths at the initial age distribution.  Must be called
    fresh for each simulation (modules cannot be shared).

``make_age_data()``
    Returns a per-year age-distribution dataframe whose smooth density is
    consistent with the mortality rates passed to ``ss.Deaths``,
    so agent initialisation and mortality are in demographic equilibrium.
"""

import numpy as np
import pandas as pd
import starsim as ss

from .demographics import compute_mortality_schedule

# ------------------------------------------------------------------
# Age structure
# ------------------------------------------------------------------

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

# Monthly EIR (bites / person / month) for each node
URBAN_EIR      = [5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5,  5]
PERI_URBAN_EIR = [2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2]
RURAL_EIR      = [1,  1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1]

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

def make_steady_state_demographics():
    """
    Return a fresh ``[ss.Births, ss.Deaths]`` demographics list calibrated for
    a stable, representative population.

    The birth rate is derived from the age-specific mortality schedule so
    that births exactly balance deaths at the initial age distribution.
    This is the standard approach for malaria modelling, where simulations
    typically use a representative constant population rather than actual
    census birth/death data.

    A new list must be created for each simulation — modules cannot be
    shared across ``ss.Sim`` instances.

    Returns
    -------
    list
        ``[ss.Births, ss.Deaths]`` ready to pass as ``demographics=`` to
        ``ms.Sim``.
    """
    mu, _, birth_rate, max_age = compute_mortality_schedule(AGE_BIN_EDGES, AGE_BIN_FRACTIONS)
    births = ss.Births(birth_rate=ss.peryear(birth_rate * 1000))
    df = pd.DataFrame({
        'Time':        2000,
        'AgeGrpStart': np.arange(max_age),
        'mx':          mu,
    })
    metadata = {'data_cols': dict(year='Time', age='AgeGrpStart', value='mx')}
    return [births, ss.Deaths(death_rate=df, rate_units=1, metadata=metadata)]


def make_age_data():
    """
    Return a per-year age-distribution dataframe for initialising agent ages.

    The smooth density n(x) is derived from the same log-linear interpolation
    used to compute per-year mortality rates, so agent initialisation and
    mortality are at the same demographic equilibrium.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age`` (int, 0–99) and ``value`` (float, relative weight).
        Suitable for ``ss.People(age_data=...)``.
    """
    _, n, _, max_age = compute_mortality_schedule(AGE_BIN_EDGES, AGE_BIN_FRACTIONS)
    return pd.DataFrame({'age': np.arange(max_age, dtype=int), 'value': n})


def make_mortality_dataframe():
    """
    Return a ``(dataframe, birth_rate)`` pair suitable for ``ss.Deaths``.

    The dataframe has columns ``Time``, ``AgeGrpStart``, ``mx`` — one row
    per integer age 0 to max_age-1.  Rates are per-year (use ``rate_units=1``).

    Returns
    -------
    df : pandas.DataFrame
        Age-specific mortality rates for ``ss.Deaths(death_rate=df, rate_units=1)``.
    birth_rate : float
        Calibrated births per person per year.
    """
    mu, _, birth_rate, max_age = compute_mortality_schedule(
        AGE_BIN_EDGES, AGE_BIN_FRACTIONS
    )
    df = pd.DataFrame({
        'Time':        2000,
        'AgeGrpStart': np.arange(max_age),
        'mx':          mu,
    })
    return df, birth_rate
