"""
MosquitoesAdultsOnly: node-level adult female mosquito dynamics (MEI model).

Deterministic node-level ODE model:
  M = total adult females
  E = infected (incubating EIP)
  I = infectious

Produces per-node EIR for the human disease module via NodeState.
"""

import numpy as np
import starsim as ss

from .utils import interpolate_monthly


class MosquitoesAdultsOnly(ss.Module):
    """
    Node-level adult female mosquito dynamics with MEI infection states.

    Parameters
    ----------
    emergence : array-like of length 12
        Monthly emergence index (linearly interpolated). Scaled by
        emergence_scale and node population.
    bites_per_female_per_day : float
        Average biting rate. Default: 0.3.
    f_indoor : float
        Indoor bite fraction. Default: 0.7.
    life_expectancy : float
        Mean female lifespan in days. Default: 10.0.
    eip_days : float
        Extrinsic incubation period in days. Default: 10.0.
    emergence_scale : float
        Multiplier for emergence index. Default: 5.0.
    init_females_per_1000 : float
        Initial adult females per 1000 humans. Default: 200.0.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            emergence=None,  # 12-element monthly profile per node, or set via NodeGrid
            bites_per_female_per_day=0.3,
            f_indoor=0.7,
            life_expectancy=10.0,
            eip_days=10.0,
            emergence_scale=5.0,
            init_females_per_1000=200.0,
        )
        self.update_pars(pars, **kwargs)
        self._ns_cache = None
        self._initialized = False
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_susceptible', dtype=float, label='Susceptible mosquitoes'),
            ss.Result('n_exposed', dtype=float, label='Exposed mosquitoes'),
            ss.Result('n_infectious', dtype=float, label='Infectious mosquitoes'),
            ss.Result('n_total', dtype=float, label='Total mosquitoes'),
            ss.Result('eir_annual', dtype=float, label='EIR — total'),
            ss.Result('eir_indoor_annual', dtype=float, label='EIR — indoor'),
            ss.Result('eir_outdoor_annual', dtype=float, label='EIR — outdoor'),
        )
        return

    def step(self):
        """Run one timestep of mosquito dynamics."""
        ns = self._get_node_state()
        grid = ns.grid
        sim = self.sim
        ti = sim.ti
        dt_days = sim.t.dt_year * 365.0
        p = self.pars

        if not self._initialized:
            self._init_state(ns)
            self._initialized = True

        a = float(p.bites_per_female_per_day)
        f_in = float(np.clip(p.f_indoor, 0.0, 1.0))
        life_expectancy = float(max(1e-6, p.life_expectancy))
        mu_daily = 1.0 / life_expectancy
        eip = float(max(0.1, p.eip_days))
        sigma = 1.0 / eip
        surv = float(np.exp(-mu_daily * dt_days))

        # Current state
        M = self._M.copy()
        E = self._E.copy()
        I = self._I.copy()

        # Seasonal emergence
        emergence = self._get_emergence(ns, sim) * dt_days

        # H2M inputs from NodeState
        p_in = np.clip(ns.p_inf_indoor, 0.0, 1.0)
        p_out = np.clip(ns.p_inf_outdoor, 0.0, 1.0)

        # Force of infection on mosquitoes
        lam = a * (f_in * p_in + (1.0 - f_in) * p_out) * dt_days
        S = np.maximum(M - E - I, 0.0)
        new_E = S * (1.0 - np.exp(-lam))

        # EIP progression
        prog = E * (1.0 - np.exp(-sigma * dt_days))

        # Update compartments
        E_next = E * surv + new_E - prog
        I_next = I * surv + prog
        M_next = M * surv + emergence

        E_next = np.maximum(E_next, 0.0)
        I_next = np.maximum(I_next, 0.0)

        # Constraint: E + I <= M
        tot_inf = E_next + I_next
        too_big = tot_inf > np.maximum(M_next, 1e-6)
        if np.any(too_big):
            scale = M_next[too_big] / tot_inf[too_big]
            E_next[too_big] *= scale
            I_next[too_big] *= scale

        self._M[:] = M_next
        self._E[:] = E_next
        self._I[:] = I_next

        # Compute and write EIR to NodeState
        Nh = np.maximum(ns.n_humans, 1).astype(float)
        eir_in = a * f_in * I_next / Nh * dt_days
        eir_out = a * (1.0 - f_in) * I_next / Nh * dt_days
        ns.eir_indoor[:] = eir_in
        ns.eir_outdoor[:] = eir_out

        # Record results
        self.results.n_total[ti] = float(M_next.sum())
        self.results.n_susceptible[ti] = float(S.sum())
        self.results.n_exposed[ti] = float(E_next.sum())
        self.results.n_infectious[ti] = float(I_next.sum())
        self.results.eir_annual[ti] = float((eir_in + eir_out).mean() / dt_days * 365.0) if dt_days > 0 else 0.0
        self.results.eir_indoor_annual[ti] = float(eir_in.mean() / dt_days * 365.0) if dt_days > 0 else 0.0
        self.results.eir_outdoor_annual[ti] = float(eir_out.mean() / dt_days * 365.0) if dt_days > 0 else 0.0
        return

    def _init_state(self, ns):
        """Initialize mosquito compartment arrays."""
        grid = ns.grid
        n = grid.n_nodes
        init_per_1000 = float(self.pars.init_females_per_1000)

        # Use node populations from n_humans (set by human module in step())
        # On first call, n_humans may be zero; use grid populations instead
        node_pops = np.array([grid.total_population * f for f in grid.pop_fractions])
        self._M = (init_per_1000 * node_pops / 1000.0).astype(float)
        self._E = np.zeros(n, dtype=float)
        self._I = np.zeros(n, dtype=float)
        return

    def _get_emergence(self, ns, sim):
        """Compute per-node emergence rate for this timestep."""
        grid = ns.grid
        p = self.pars
        emergence_scale = float(max(0.0, p.emergence_scale))
        node_pops = np.array([grid.total_population * f for f in grid.pop_fractions])

        # Get monthly emergence index per node
        t_frac = sim.t.year % 1
        n_nodes = grid.n_nodes
        idx = np.zeros(n_nodes)
        for nid in range(n_nodes):
            if p.emergence is not None:
                profile = np.asarray(p.emergence)
                if profile.ndim == 1:
                    idx[nid] = interpolate_monthly(profile, t_frac)
                else:
                    idx[nid] = interpolate_monthly(profile[nid], t_frac)
            else:
                idx[nid] = 1.0  # uniform if no profile given

        return emergence_scale * idx * (node_pops / 1000.0)

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
