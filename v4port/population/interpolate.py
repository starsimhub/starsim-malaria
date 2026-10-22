""" malariasim.utils.interpolate_monthly, verbatim """
import numpy as np
def interpolate_monthly(profile, t):
    x_months = np.arange(12) / 12
    x_wrap   = np.append(x_months, 1.0)
    y_wrap   = np.append(profile, profile[0])
    return float(np.interp(t, x_wrap, y_wrap))
