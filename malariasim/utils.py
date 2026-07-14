"""Shared utility functions for the malariasim package."""

import numpy as np

from .defaults import CommonParams


def eir_to_params(
    target_annual_eir,
    common_pars      = None,
    seasonal_profile = None,
    n_months         = 12,
):
    """
    Convert a target annual EIR into the NodeGrid parameters needed by each
    mosquito model type.

    Returns a tuple ``(eir_profile, emergence_profile)`` where both are lists
    of length *n_months* with the same value repeated each month (constant
    transmission).  Pass both to ``ms.NodeGrid(eir=..., emergence=...)``.

    The **eir_only** mosquito model reads ``eir`` directly.

    The **ODE / agents** mosquito models use ``emergence`` (new adult female
    mosquitoes per human per day) and ignore ``eir``.  The emergence rate is
    derived from the Ross-Macdonald SEI equilibrium at full human prevalence:

    .. code-block:: none

        Sm* = e·Nh / (a·c + μ)
        Em* = a·c·Sm* / (σ + μ)
        Im* = σ·Em* / μ
        EIR/day = a·Im* / Nh

        SIS equilibrium prevalence:
            x* = b·EIR / (b·EIR + r)

        Substituting x* into the mosquito SEI equilibrium:
            p_inf = c·x*
            emergence e = EIR/day · μ·(a·p_inf + μ)·(σ + μ) / (a²·p_inf·σ)

    where ``a`` = biting rate, ``μ = 1/life_expectancy``, ``σ = 1/eip``,
    ``c`` = prob_mosquito_infected_on_bite, ``b`` =
    prob_human_infected_on_bite, ``r`` = recovery rate per day.

    Parameters
    ----------
    target_annual_eir : float
        Desired EIR in infective bites per person per year.
    common_pars : CommonParams, optional
        Shared biological parameters.  Defaults to ``CommonParams()`` if None.
    seasonal_profile : array-like of length n_months, optional
        Relative monthly weights for the seasonal shape.  Normalized internally
        so the mean equals 1.0, then scaled to ``target_annual_eir``.  If None,
        a flat (constant) profile is used.
    n_months : int
        Length of the returned profile lists (12 for monthly).

    Returns
    -------
    eir_profile : list of float, length n_months
        Monthly EIR values (bites/person/year) for the eir_only model.
    emergence_profile : list of float, length n_months
        Monthly emergence rates (mosquitoes/human/day) for ODE/agents models.

    Examples
    --------
    ::

        eir, emergence = ms.eir_to_params(target_annual_eir=10)
        nodes = ms.NodeGrid(populations=[1000], eir=[eir], emergence=[emergence])
    """
    if common_pars is None:
        common_pars = CommonParams()

    a   = common_pars.bites_per_female_per_day
    mu  = 1.0 / common_pars.mosquito_life_expectancy_days
    sig = 1.0 / common_pars.mosquito_eip_days
    c   = common_pars.prob_mosquito_infected_on_bite
    b   = common_pars.prob_human_infected_on_bite
    r   = 1.0 / common_pars.infection_duration_mean_days

    # Build the monthly EIR shape (annualized rates, bites/person/year).
    # seasonal_profile weights are normalized so their mean equals 1.0, then
    # scaled so the annual mean equals target_annual_eir.
    if seasonal_profile is None:
        monthly_eir = [float(target_annual_eir)] * n_months
    else:
        weights     = np.asarray(seasonal_profile, dtype=float)
        weights     = weights / weights.mean()          # normalize: mean = 1.0
        monthly_eir = (weights * target_annual_eir).tolist()

    # Compute per-month emergence using quasi-static Ross-Macdonald equilibrium:
    # each month is treated as an independent steady state at that month's EIR.
    emergence_profile = []
    for eir_annual in monthly_eir:
        eir_per_day = eir_annual / 365.0
        x_star = b * eir_per_day / (b * eir_per_day + r)
        p_inf  = c * x_star
        emergence_profile.append(eir_per_day * mu * (a * p_inf + mu) * (sig + mu) / (a ** 2 * p_inf * sig))

    return monthly_eir, emergence_profile


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
