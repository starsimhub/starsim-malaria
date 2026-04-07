"""
MosquitoesEirOnly: EIR-driven mosquito module (no mosquito agents).

The EIR is supplied directly as a parameter (scalar or 12-element monthly
profile) rather than computed from an explicit mosquito population.
"""

import numpy as np
import starsim as ss

from .utils import interpolate_monthly


class MosquitoesEirOnly(ss.Module):
    """
    Fixed or seasonally-varying EIR mosquito module.

    Parameters
    ----------
    eir : float or array-like of length 12
        Uniform or monthly **per-timestep** EIR (infectious bites per person per
        timestep), LASER-compatible with ``eir_month_XX`` scenario columns.
        Scalar or 12-point profile with linear interpolation on the calendar year.
    f_indoor : float
        Fraction of bites indoors. Default: 0.7.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            eir=10.0,
            f_indoor=0.7,
        )
        self.update_pars(pars, **kwargs)
        self._validate()
        self._ns_cache = None
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('eir_annual', dtype=float, label='EIR — mean total (per timestep)'),
            ss.Result('eir_indoor_annual', dtype=float, label='EIR — mean indoor (per timestep)'),
            ss.Result('eir_outdoor_annual', dtype=float, label='EIR — mean outdoor (per timestep)'),
        )
        return

    def step(self):
        """Record annualized EIR and write per-timestep EIR to NodeState."""
        sim = self.sim
        ti = sim.ti
        annual_fallback = self._annual_rate(sim)

        ns = self._get_node_state()
        grid = ns.grid
        eir_total_sum = 0.0
        eir_in_sum = 0.0
        eir_out_sum = 0.0
        for nid in range(grid.n_nodes):
            if grid.eir is not None:
                node_eir = grid.eir[nid] if grid.eir.ndim == 1 else interpolate_monthly(grid.eir[nid], sim.t.year % 1)
            else:
                node_eir = annual_fallback
            f = grid.f_indoor[nid] if grid.f_indoor is not None else self.pars.f_indoor
            # Same units as LASER nodes.mos_eir_*: per-timestep infectious biting rate
            # (monthly profile values are interpolated to the current day; not × dt_year).
            ns.eir_indoor[nid] = node_eir * f
            ns.eir_outdoor[nid] = node_eir * (1.0 - f)
            eir_total_sum += node_eir
            eir_in_sum += node_eir * f
            eir_out_sum += node_eir * (1.0 - f)

        # Record mean across nodes (effective EIR, not just module-level param)
        n = grid.n_nodes
        self.results.eir_annual[ti] = eir_total_sum / n
        self.results.eir_indoor_annual[ti] = eir_in_sum / n
        self.results.eir_outdoor_annual[ti] = eir_out_sum / n
        return

    def _annual_rate(self, sim):
        """Return current annualized total EIR."""
        eir = self.pars.eir
        if np.isscalar(eir):
            return float(eir)
        return interpolate_monthly(self.pars.eir, sim.t.year % 1)

    def _validate(self):
        """Validate parameters."""
        eir = self.pars.eir
        if np.isscalar(eir):
            if eir < 0:
                raise ValueError(f'eir must be non-negative, got {eir}')
        else:
            eir = np.asarray(eir, dtype=float)
            if eir.shape != (12,):
                raise ValueError(f'Monthly eir must be a 12-element array, got shape {eir.shape}')
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
                    'module found. Use Sim() or add NodeState manually.'
                )
        return self._ns_cache
