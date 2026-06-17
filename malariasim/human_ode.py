"""
MalariaHumanODE: age-structured SIS compartmental (ODE) human disease module.

Registered as a disease (diseases= in ss.Sim), the same as MalariaHumanSIS
and MalariaHumanSEIS.  Uses Euler-stepped ODEs over per-node, per-age-bin
Sh/Ih compartments instead of individual human agents.

NodeState interface
-------------------
Reads  : eir_indoor, eir_outdoor           (written by mosquito connector last step)
Writes : p_inf_indoor, p_inf_outdoor, n_humans  (read by mosquito connector this step)

Age structure
-------------
When age_bin_edges is provided (e.g. [0,5,10,...,80,100]), the population
is distributed across age bins. Each step, people:
  - get infected / recover (per bin, same EIR for all ages)
  - age out of bin a into bin a+1 at rate 1/bin_width_a per year
  - are born into the youngest bin at birth_rate_per_year * N_total per year
  - die out of all bins at death_rate_per_year per year

When age_bin_edges=None (default), the model reduces to a single-bin SIS ODE
with no age structure, identical to the original behaviour.

Demographics note
-----------------
When used via ms.Sim(human='ode'), the ss.Births and ss.Deaths modules passed
through demographics= are stripped from the simulation.  Birth and death rates
are instead extracted from those modules and applied internally to the ODE
compartments so the fixed-size compartment population stays consistent.

Force of infection on humans (per susceptible per timestep):

    lambda_h_dt = (eir_indoor[nid] + eir_outdoor[nid]) * b

p_inf written to NodeState:

    p_inf[nid] = c * sum(Ih[nid, :]) / sum(Sh[nid, :] + Ih[nid, :])
"""

import numpy as np
import starsim as ss


class MalariaHumanODE(ss.Disease):
    """
    Age-structured SIS ODE (compartmental) human disease module.

    Drop-in replacement for MalariaHumanSIS in ms.Sim(human='ode').
    Maintains per-node Sh and Ih floats (or per-node, per-age-bin arrays when
    age_bin_edges is provided) and advances them each timestep with an Euler
    step.  Produces the same result keys as the agent-based human modules
    (n_susceptible, n_infected, new_infections) so plots and tests require no
    changes.

    Parameters
    ----------
    prob_human_infected_on_bite : float
        Probability of human infection per infectious mosquito bite
        (Ross-MacDonald *b*). Default: 1.0.

    prob_mosquito_infected_on_bite : float
        Probability of mosquito infection per bite on an infectious human
        (Ross-MacDonald *c*). Default: 1.0.

    infection_duration : ss.dur
        Mean duration of human infection. Default: ``ss.days(200)``
        (~200 days, matching the MalariaHumanSIS default mean).

    init_prev : float
        Fraction of the population infectious at simulation start.
        Default: 0.01 (1%).

    age_bin_edges : array-like or None
        Bin boundaries in years, e.g. [0,5,10,...,80,100].
        None → single-bin mode (no age structure).

    age_bin_fractions : array-like or None
        Relative weight of each age bin (will be normalised). Length must equal
        len(age_bin_edges)-1. If None, a uniform distribution is used.
        Typically computed from age_data by ms.Sim and passed here.

    birth_rate_per_year : ss.peryear
        Crude birth rate (births per person per year). Default: ``ss.peryear(0.0)``.

    death_rate_per_year : ss.peryear
        Crude death rate (deaths per person per year). Default: ``ss.peryear(0.0)``.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            prob_human_infected_on_bite    = 1.0,           # b: M→H probability per bite
            prob_mosquito_infected_on_bite = 1.0,           # c: H→M probability per bite
            infection_duration             = ss.days(200),  # mean infectious period
            init_prev                      = 0.01,          # initial fraction infected
            age_bin_edges                  = None,
            age_bin_fractions              = None,
            birth_rate_per_year            = ss.peryear(0.0),
            death_rate_per_year            = ss.peryear(0.0),
        )
        self.update_pars(pars, **kwargs)

        self._Sh          = None   # ndarray (n_nodes, n_bins)
        self._Ih          = None   # ndarray (n_nodes, n_bins)
        self._bin_widths  = None   # years per bin, shape (n_bins,)
        self._death_rates = None   # per-bin death rates (yr^-1), shape (n_bins,)
        self._birth_rate  = 0.0    # calibrated birth rate (yr^-1)
        self._n_bins      = 1
        self._age_bin_edges = None # stored after init for plotting
        self._ns_cache    = None
        return

    # ------------------------------------------------------------------
    # Starsim module lifecycle
    # ------------------------------------------------------------------

    def init_post(self):
        super().init_post()
        ns      = self._get_node_state()
        grid    = ns.grid
        p       = self.pars
        n_nodes = grid.n_nodes

        n_total = (grid.total_population
                   if grid.total_population is not None
                   else int(self.sim.people.alive.sum()))

        # --- age bins ---
        edges = p.age_bin_edges
        if edges is not None:
            edges            = np.asarray(edges, dtype=float)
            n_bins           = len(edges) - 1
            self._bin_widths = np.diff(edges)
            # Cap the last bin's effective width to the minimum of the
            # preceding bins.  Without this, a wide terminal bin (e.g.
            # 80–100 = 20 yr) fills from the preceding 5-yr bin four times
            # faster than it drains, causing unrealistic accumulation at old
            # ages.  The cap ensures the exit rate equals or exceeds the
            # entry rate, keeping the demographic pyramid stable.  This is a
            # deliberate steady-state modelling choice, not a bug.
            if n_bins > 1:
                self._bin_widths[-1] = min(
                    self._bin_widths[-1], self._bin_widths[:-1].min()
                )
        else:
            n_bins = 1
            self._bin_widths = np.array([100.0])  # single bin — no aging
        self._n_bins = n_bins
        self._age_bin_edges = edges if edges is not None else None

        # age-bin fractions (normalised)
        fracs = p.age_bin_fractions
        if fracs is not None:
            fracs = np.asarray(fracs, dtype=float)
        else:
            fracs = np.ones(n_bins, dtype=float)
        fracs = fracs / fracs.sum()

        # --- age-specific death rates to maintain initial age distribution ---
        # For bin a > 0 at equilibrium: aging_in_{a} = (d_a + 1/w_a) * S_a
        # => d_a = (f_{a-1}/f_a) / w_{a-1} - 1/w_a
        # This gives low rates for young bins, rising rates for older bins —
        # a realistic mortality schedule that keeps the initial pyramid stable.
        global_dr = float(p.death_rate_per_year)
        self._death_rates = np.full(n_bins, global_dr)
        if n_bins > 1:
            for a in range(1, n_bins):
                if fracs[a] > 0:
                    rate = (fracs[a-1] / fracs[a]) / self._bin_widths[a-1] - 1.0 / self._bin_widths[a]
                    self._death_rates[a] = max(rate, 0.0)
            # Calibrate birth rate so bin 0 is also at equilibrium:
            # births/yr = (d_0 + 1/w_0) × S_0 = (d_0 + 1/w_0) × f_0 × N
            self._birth_rate = (self._death_rates[0] + 1.0 / self._bin_widths[0]) * fracs[0]
        else:
            self._birth_rate = float(p.birth_rate_per_year)

        # allocate (n_nodes, n_bins)
        self._Sh = np.zeros((n_nodes, n_bins))
        self._Ih = np.zeros((n_nodes, n_bins))

        for nid in range(n_nodes):
            n_h = n_total * grid.pop_fractions[nid]
            for a in range(n_bins):
                n_a              = n_h * fracs[a]
                i0               = n_a * float(p.init_prev)
                self._Sh[nid, a] = n_a - i0
                self._Ih[nid, a] = i0

        self._write_node_state(ns)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_susceptible',  dtype=float, label='Susceptible humans (ODE)'),
            ss.Result('n_infected',     dtype=float, label='Infectious humans (ODE)'),
            ss.Result('new_infections', dtype=float, label='New infections (ODE)'),
        )
        return

    def step(self):
        """
        Advance human compartments by one timestep (Euler integration).

        Reads EIR from NodeState (written by the mosquito connector on the
        previous step), computes new infections and recoveries for each node
        and age bin, applies births/deaths/aging flows, records results, then
        writes p_inf and n_humans to NodeState for the mosquito connector to
        read this step.
        """
        ns          = self._get_node_state()
        p           = self.pars
        dt_year     = self.dt.years
        r_per_step  = self.dt / p.infection_duration   # dimensionless: fraction of mean duration per step

        b        = p.prob_human_infected_on_bite
        birth_dt = self._birth_rate * dt_year           # scalar
        death_dt = self._death_rates * dt_year          # shape (n_bins,) — age-specific
        # fraction of each bin that ages out per timestep
        aging_dt = dt_year / self._bin_widths   # shape (n_bins,)

        total_new = 0.0

        for nid in range(ns.grid.n_nodes):
            Sh = self._Sh[nid]
            Ih = self._Ih[nid]

            N_total = Sh.sum() + Ih.sum()

            eir_dt  = ns.eir_indoor[nid] + ns.eir_outdoor[nid]
            new_inf = eir_dt * b * Sh           # (n_bins,)
            recover = r_per_step * Ih           # (n_bins,)

            dS_death = death_dt * Sh
            dI_death = death_dt * Ih

            aging_out_S = aging_dt * Sh
            aging_out_I = aging_dt * Ih

            aging_in_S       = np.zeros(self._n_bins)
            aging_in_S[1:]   = aging_out_S[:-1]
            # People aging out of the last bin exit the system (like natural death).
            # Births replenish exactly the total losses so population stays constant.
            aging_in_S[0]    = birth_dt * N_total

            aging_in_I       = np.zeros(self._n_bins)
            aging_in_I[1:]   = aging_out_I[:-1]

            self._Sh[nid] = np.maximum(
                Sh - new_inf + recover - dS_death - aging_out_S + aging_in_S, 0.0)
            self._Ih[nid] = np.maximum(
                Ih + new_inf - recover - dI_death - aging_out_I + aging_in_I, 0.0)

            total_new += new_inf.sum()

        self.results.n_susceptible[self.ti]  = float(self._Sh.sum())
        self.results.n_infected[self.ti]     = float(self._Ih.sum())
        self.results.new_infections[self.ti] = total_new

        self._write_node_state(ns)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_node_state(self, ns):
        """Write per-node human state to NodeState for the mosquito connector."""
        c = float(self.pars.prob_mosquito_infected_on_bite)
        for nid in range(ns.grid.n_nodes):
            Sh    = self._Sh[nid]
            Ih    = self._Ih[nid]
            N     = max(Sh.sum() + Ih.sum(), 1.0)
            p_inf = c * Ih.sum() / N
            ns.n_humans[nid]      = int(round(N))
            ns.p_inf_indoor[nid]  = p_inf
            ns.p_inf_outdoor[nid] = p_inf
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
                    'module found. Use ms.Sim() or add NodeState manually.'
                )
        return self._ns_cache
