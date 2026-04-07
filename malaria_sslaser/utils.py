"""
Shared utility functions for the malaria_sslaser package.
"""

import numpy as np


def interpolate_monthly(profile, t):
    """
    Linearly interpolate a 12-element monthly profile at fractional year t.

    Parameters
    ----------
    profile : array-like of length 12
        Monthly values (index 0 = January).
    t : float
        Fractional year in [0, 1). December wraps to January.

    Returns
    -------
    float
        Interpolated value.
    """
    profile = np.asarray(profile, dtype=float)
    t = t % 1.0
    month_float = t * 12.0
    m = int(np.floor(month_float)) % 12
    frac = month_float - int(np.floor(month_float))
    m_next = (m + 1) % 12
    return float((1.0 - frac) * profile[m] + frac * profile[m_next])
