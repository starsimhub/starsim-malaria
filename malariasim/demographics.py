"""
Demographic utilities for malariasim.

``compute_mortality_schedule`` derives per-year mortality rates and a
calibrated birth rate from age-bin population fractions using log-linear
interpolation.  It is the shared mathematical core used by
``make_steady_state_demographics`` and ``make_age_data`` in defaults.py.
"""

import numpy as np


def compute_mortality_schedule(age_bin_edges, age_bin_fractions):
    """
    Derive per-year mortality rates and a calibrated birth rate from age-bin
    population fractions using log-linear interpolation.

    Parameters
    ----------
    age_bin_edges : array-like
        Bin boundaries in years (length n_bins+1), e.g. [0,5,10,...,100].
    age_bin_fractions : array-like
        Relative population weight of each bin (length n_bins). Normalised
        internally.

    Returns
    -------
    mu : ndarray, shape (max_age,)
        Per-year mortality rate for integer ages 0 .. max_age-1.
    n : ndarray, shape (max_age,)
        Smooth per-year population density (relative, sums arbitrarily).
    calibrated_birth_rate : float
        Births per person per year that exactly balance deaths at the initial
        age distribution.
    max_age : int
        Maximum age in years (= age_bin_edges[-1]).
    """
    edges = np.asarray(age_bin_edges, dtype=float)
    fracs = np.asarray(age_bin_fractions, dtype=float)
    fracs /= fracs.sum()

    max_age   = int(edges[-1])
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    widths    = np.diff(edges)
    densities = fracs / widths

    ages_int = np.arange(max_age, dtype=float)
    log_n    = np.interp(ages_int, midpoints, np.log(densities))
    n        = np.exp(log_n)

    mu       = np.log(n[:-1] / n[1:])
    mu_last  = 2.0   # ~86 % annual mortality at the final age
    mu       = np.append(mu, mu_last)

    p_die_1yr          = 1.0 - np.exp(-mu)
    cap_rate           = np.exp(-mu[-1]) * n[-1] / n.sum()
    calibrated_birth_rate = float(np.dot(p_die_1yr, n) / n.sum() + cap_rate)

    return mu, n, calibrated_birth_rate, max_age
