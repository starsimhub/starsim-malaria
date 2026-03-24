"""Shared utility functions for the malariasim package."""

import numpy as np


def interpolate_monthly(profile, t):
    """
    Linearly interpolate a 12-element monthly profile at fractional year *t*.

    Monthly values are treated as anchors at the start of each month
    (0/12, 1/12, …, 11/12). The January value is appended at t = 1.0 to
    close the domain and wrap December back to January.

    Parameters
    ----------
    profile : array-like, shape (12,)
        One value per calendar month.
    t : float
        Fractional position within the year, in [0, 1).

    Returns
    -------
    float
        Interpolated value at time *t*.
    """
    x_months = np.arange(12) / 12
    x_wrap   = np.append(x_months, 1.0)
    y_wrap   = np.append(profile, profile[0])
    return float(np.interp(t, x_wrap, y_wrap))
