"""
MosqitoesEirOnly: EIR-driven mosquito module (no mosquito agents).

This is the simplest possible mosquito model: the Entomological Inoculation
Rate (EIR) is supplied directly as a parameter rather than computed from an
explicit mosquito population. No mosquito states are tracked.

Use this module when:
  - You want to drive human infection from observed or assumed EIR data
  - You are not interested in mosquito population dynamics
  - You want a fast, simple baseline before switching to MalariaMosquito

To switch to the full Ross-MacDonald mosquito model, replace MosqitoesEirOnly
with MalariaMosquito in your sim — MalariaHumanSIS will detect the change
automatically.

EIR definition
--------------
EIR (Entomological Inoculation Rate) is the expected number of infectious
mosquito bites received per person per year. A value of 1 means one infectious
bite per year on average; hyperendemic settings can exceed 100.

Indoor vs outdoor EIR
---------------------
The total EIR is split into indoor and outdoor components using the f_indoor
parameter. This split matters for interventions:
  - ITNs and IRS reduce indoor exposure
  - Outdoor exposure is unaffected by those interventions

Configuring monthly EIR
-----------------------
Pass a 12-element list to capture seasonal transmission. Each element is the
annualized EIR rate at the start of that calendar month (January = index 0).
Values between months are linearly interpolated, so the EIR transitions
smoothly rather than jumping at month boundaries. The year wraps continuously:
the interpolation between December (index 11) and January (index 0) follows
the same linear rule.

    # Wet-season peak in months 4-9 (May-Oct), low otherwise
    MosqitoesEirOnly(eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2])

Pass a scalar for uniform (non-seasonal) transmission:

    MosqitoesEirOnly(eir=10)
"""

import numpy as np
import starsim as ss


class MosqitoesEirOnly(ss.Module):
    """
    Fixed or seasonally-varying EIR mosquito module (no mosquito agents).

    Provides indoor and outdoor EIR to MalariaHumanSIS based on either a
    constant annual rate or a monthly profile with linear interpolation
    between months.

    Parameters
    ----------
    eir : float or array-like of length 12
        Total annual EIR in infectious bites per person per year.

        - **Scalar**: uniform transmission throughout the year.
          Example: ``eir=10``
        - **12-element list/array**: annualized rates anchored at the start
          of each calendar month (index 0 = January). EIR is linearly
          interpolated between anchor points; December wraps to January.
          Example: ``eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2]``

    f_indoor : float
        Fraction of total EIR occurring indoors. Must be in [0, 1].
        Default: 0.7 (70% of bites indoors).
        The outdoor fraction is automatically ``1 - f_indoor``.

    Examples
    --------
    Constant moderate transmission, default indoor fraction::

        mos = MosqitoesEirOnly(eir=10)

    Seasonal transmission with wet-season peak, mostly indoor biting::

        mos = MosqitoesEirOnly(
            eir=[2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2],
            f_indoor=0.85,
        )

    Used with MalariaHumanSIS::

        sim = ss.Sim(diseases=MalariaHumanSIS(), connectors=mos)
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            eir      = 10.0,  # annual infectious bites per person
            f_indoor = 0.7,   # fraction of bites occurring indoors
        )
        self.update_pars(pars, **kwargs)
        self._validate()
        self._ns_cache = None
        return

    # ------------------------------------------------------------------
    # Starsim module lifecycle
    # ------------------------------------------------------------------

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('eir_annual',         dtype=float, label='EIR — total (bites/person/year)'),
            ss.Result('eir_indoor_annual',  dtype=float, label='EIR — indoor (bites/person/year)'),
            ss.Result('eir_outdoor_annual', dtype=float, label='EIR — outdoor (bites/person/year)'),
        )
        return

    def step(self):
        """Record annualized EIR and write per-timestep EIR to NodeState."""
        sim    = self.sim
        ti     = sim.ti
        annual = self._annual_rate(sim)
        self.results.eir_annual[ti]         = annual
        self.results.eir_indoor_annual[ti]  = annual * self.pars.f_indoor
        self.results.eir_outdoor_annual[ti] = annual * (1.0 - self.pars.f_indoor)

        ns      = self._get_node_state()
        dt_year = sim.t.dt_year
        grid    = ns.grid
        for nid in range(grid.n_nodes):
            # Per-node EIR from NodeGrid if specified, else module-level parameter.
            if grid.eir is not None:
                node_eir = grid.eir[nid] if grid.eir.ndim == 1 else self._interpolate_profile(grid.eir[nid], sim.t.year % 1)
            else:
                node_eir = annual
            f = grid.f_indoor[nid] if grid.f_indoor is not None else self.pars.f_indoor
            ns.eir_indoor[nid]  = node_eir * f         * dt_year
            ns.eir_outdoor[nid] = node_eir * (1.0 - f) * dt_year
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annual_rate(self, sim):
        """
        Return the current annualized total EIR.

        For a scalar eir, returns it directly. For a monthly profile,
        linearly interpolates between the two surrounding month anchors.
        """
        eir = self.pars.eir
        if np.isscalar(eir):
            return float(eir)
        return self._interpolate(sim.t.year % 1)

    def _interpolate(self, t):
        """
        Linearly interpolate the monthly EIR profile at fractional year t.

        Monthly values are treated as anchors at the start of each month:
          month 0 (Jan) → t = 0/12
          month 1 (Feb) → t = 1/12
          ...
          month 11 (Dec) → t = 11/12

        The January value is appended at t = 1.0 to close the [0, 1] domain
        and allow smooth wrapping from December back to January.

        Parameters
        ----------
        t : float
            Fractional position within the current year, in [0, 1).

        Returns
        -------
        float
            Interpolated annualized EIR at time t.
        """
        eir      = self.pars.eir                   # shape (12,)
        x_months = np.arange(12) / 12              # [0/12, 1/12, ..., 11/12]
        x_wrap   = np.append(x_months, 1.0)        # add year-end anchor
        y_wrap   = np.append(eir, eir[0])          # wrap Dec → Jan
        return float(np.interp(t, x_wrap, y_wrap))

    def _validate(self):
        """Raise informative errors for malformed parameters."""
        eir = self.pars.eir
        if np.isscalar(eir):
            if eir < 0:
                raise ValueError(f'eir must be non-negative, got {eir}')
        else:
            eir = np.asarray(eir, dtype=float)
            if eir.shape != (12,):
                raise ValueError(
                    f'Monthly eir must be a 12-element array (one per calendar '
                    f'month), got shape {eir.shape}'
                )
            if np.any(eir < 0):
                raise ValueError('All monthly eir values must be non-negative')
            self.pars.eir = eir

        f = self.pars.f_indoor
        if not (0.0 <= f <= 1.0):
            raise ValueError(f'f_indoor must be in [0, 1], got {f}')
        return

    def _get_node_state(self):
        """Return NodeState, raising RuntimeError if not found."""
        if self._ns_cache is None:
            self._ns_cache = next(
                (m for m in self.sim.modules if getattr(m, '_is_node_state', False)),
                None,
            )
            if self._ns_cache is None:
                raise RuntimeError(
                    f'{type(self).__name__}._get_node_state(): no NodeState '
                    'module found. Use make_sim() or add NodeState manually.'
                )
        return self._ns_cache

    def _interpolate_profile(self, profile, t):
        """Interpolate a 12-element monthly profile at fractional year t."""
        x_months = np.arange(12) / 12
        x_wrap   = np.append(x_months, 1.0)
        y_wrap   = np.append(profile, profile[0])
        return float(np.interp(t, x_wrap, y_wrap))
