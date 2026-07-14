"""
MalariaHumanODE: age-structured SIS compartmental (ODE) human disease module.

Registered as a disease (diseases= in ss.Sim), the same as MalariaHumanSIS
and MalariaHumanSEIS.  Uses Euler-stepped ODEs over per-node, per-age-bin
Sh/Ih compartments instead of individual human agents.

People-based compartment slots
-------------------------------
ss.People is populated with n_nodes * n_bins fixed-slot "agents" — one per
(node, age_bin) compartment.  Starsim aging is disabled (no demographics).
In init_post:

    people.age[uid] = bin_edges[uid % n_bins]   — enables uniform age filtering
    disease.node_id[uid] = uid // n_bins         — enables uniform node filtering

All compartment state (_Sh, _Ih, _prob_acquire, _vaccinated_fraction,
_dose_time) is stored as ss.FloatArr states on the disease module, sized to
n_agents = n_nodes * n_bins.  The Euler step accesses them via
.raw.reshape(n_nodes, n_bins) views for vectorized arithmetic.

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

from .human_base import BiteTarget


class MalariaHumanODE(ss.Disease):
    """
    Age-structured SIS ODE (compartmental) human disease module.

    Drop-in replacement for MalariaHumanSIS in ms.Sim(human='ode').
    Maintains per-node, per-age-bin Sh/Ih compartments as ss.FloatArr states
    and advances them each timestep with a vectorized Euler step.

    Parameters
    ----------
    prob_human_infected_on_bite : float
        Probability of human infection per infectious mosquito bite
        (Ross-MacDonald *b*). Default: 1.0.

    prob_mosquito_infected_on_bite : float
        Probability of mosquito infection per bite on an infectious human
        (Ross-MacDonald *c*). Default: 1.0.

    infection_duration : ss.dur
        Mean duration of the infectious period (the window during which the
        compartment contributes to mosquito infection). Default: ``ss.days(200)``.
        The ODE uses only the mean; std from CommonParams is ignored.

    init_prev : float
        Fraction of the population infectious at simulation start.
        Default: 0.0 (0%).

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

    @classmethod
    def pars_from(cls, cp):
        """Return a parameter dict built from a CommonParams instance.

        Translates the common biological values into ODE-appropriate types
        (scalar infection_duration rather than a stochastic distribution).
        """
        return dict(
            prob_human_infected_on_bite    = cp.prob_human_infected_on_bite,
            prob_mosquito_infected_on_bite = cp.prob_mosquito_infected_on_bite,
            infection_duration             = ss.days(cp.infection_duration_mean_days),
        )

    def __init__(self, common_pars=None, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            prob_human_infected_on_bite    = 1.0,           # b: M→H probability per bite
            prob_mosquito_infected_on_bite = 1.0,           # c: H→M probability per bite
            infection_duration             = ss.days(200),  # mean infectious period
            init_prev                      = 0.0,          # initial fraction infected
            age_bin_edges                  = None,
            age_bin_fractions              = None,
            birth_rate_per_year            = ss.peryear(0.0),
            death_rate_per_year            = ss.peryear(0.0),
        )
        if common_pars is not None:
            self.update_pars(self.pars_from(common_pars))
        self.update_pars(pars, **kwargs)

        self._bin_widths    = None   # years per bin, shape (n_bins,)
        self._death_rates   = None   # per-bin death rates (yr^-1), shape (n_bins,)
        self._birth_rate    = 0.0    # calibrated birth rate (yr^-1)
        self._n_bins        = 1
        self._age_bin_edges = None   # stored after init for plotting
        self._ns_cache      = None

        self.define_states(
            ss.FloatArr('_Sh',                   default=0.0,  label='Susceptible (ODE)'),
            ss.FloatArr('_Ih',                   default=0.0,  label='Infectious (ODE)'),
            ss.FloatArr('_prob_acquire_indoor',  default=1.0,  label='Indoor acquisition probability multiplier'),
            ss.FloatArr('_prob_acquire_outdoor', default=1.0,  label='Outdoor acquisition probability multiplier'),
            ss.FloatArr('_vaccinated_fraction',  default=0.0,  label='Vaccinated fraction per compartment'),
            ss.FloatArr('_itn_fraction',         default=0.0,  label='ITN-covered fraction per compartment'),
            ss.FloatArr('_drug_coverage',        default=0.0,  label='Fraction of compartment covered by active drug dose'),
            ss.FloatArr('_drug_clear_end_t',     default=-1.0, label='Sim time (years) when drug window closes (sentinel -1)'),
            ss.FloatArr('_drug_clearance_rate',  default=0.0,  label='Drug clearance rate 1/mean_duration_years'),
            ss.FloatArr('_dose_time',            default=0.0,  label='Population-weighted mean dose time'),
            ss.IntArr(  'node_id',               default=0,    label='Node index'),
        )

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
        self._n_bins        = n_bins
        self._age_bin_edges = edges

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

        # --- populate People slots ---
        all_uids = self.sim.people.auids  # ss.uids([0, 1, ..., n_nodes*n_bins-1])
        uid_arr  = np.asarray(all_uids)

        # Set people.age = bin_min_age so HumanEligibility age filtering works
        bin_min_ages = edges[:-1] if edges is not None else np.array([0.0])
        self.sim.people.age[all_uids] = bin_min_ages[uid_arr % n_bins]

        # Set node_id
        self.node_id[all_uids] = uid_arr // n_bins

        # Initialise Sh, Ih compartment values
        Sh_init = np.zeros(n_nodes * n_bins)
        Ih_init = np.zeros(n_nodes * n_bins)
        for nid in range(n_nodes):
            n_h = n_total * grid.pop_fractions[nid]
            for a in range(n_bins):
                uid          = nid * n_bins + a
                n_a          = n_h * fracs[a]
                i0           = n_a * float(p.init_prev)
                Sh_init[uid] = n_a - i0
                Ih_init[uid] = i0

        self._Sh[all_uids] = Sh_init
        self._Ih[all_uids] = Ih_init

        self._write_node_state(ns)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_susceptible',  dtype=float, label='Susceptible humans (ODE)'),
            ss.Result('n_infected',     dtype=float, label='Infectious humans (ODE)'),
            ss.Result('new_infections', dtype=float, label='New infections (ODE)'),
        )

    def step_state(self):
        """Reset _prob_acquire to 1.0 before connectors write vaccine/bednet effects."""
        self.reset_acquisition_probability()

    def step(self):
        """
        Advance human compartments by one timestep (vectorized Euler integration).

        Reads EIR from NodeState, computes new infections and recoveries for all
        nodes and age bins simultaneously, applies births/deaths/aging flows,
        ages vaccinated-fraction tracking state, records results, then writes
        p_inf and n_humans to NodeState.
        """
        ns         = self._get_node_state()
        p          = self.pars
        n_nodes    = ns.grid.n_nodes
        n_bins     = self._n_bins
        all_uids   = self.sim.people.auids

        dt_year    = self.dt.years
        r_per_step = self.dt / p.infection_duration   # dimensionless: fraction of mean duration per step
        b          = p.prob_human_infected_on_bite
        birth_dt   = self._birth_rate * dt_year          # scalar
        death_dt   = self._death_rates * dt_year         # (n_bins,) — age-specific
        aging_dt   = dt_year / self._bin_widths          # (n_bins,)

        # Read current state as (n_nodes, n_bins) matrices.
        # Copy Sh/Ih so old values are available throughout the Euler step.
        Sh_mat = self._Sh.raw.reshape(n_nodes, n_bins).copy()
        Ih_mat = self._Ih.raw.reshape(n_nodes, n_bins).copy()
        pa_in_mat  = self._prob_acquire_indoor.raw.reshape(n_nodes, n_bins)
        pa_out_mat = self._prob_acquire_outdoor.raw.reshape(n_nodes, n_bins)

        # Drug clearance — expire finished doses then apply active rate
        t = self.sim.t.year
        expired = (self._drug_clear_end_t > 0) & (self._drug_clear_end_t <= t)
        if len(expired.uids):
            self._drug_coverage[expired.uids]       = 0.0
            self._drug_clear_end_t[expired.uids]    = -1.0
            self._drug_clearance_rate[expired.uids] = 0.0

        dc_mat     = self._drug_coverage.raw.reshape(n_nodes, n_bins)
        dr_mat     = self._drug_clearance_rate.raw.reshape(n_nodes, n_bins)
        drug_clear = dr_mat * dt_year * dc_mat * Ih_mat   # (n_nodes, n_bins)

        # EIR → (n_nodes, 1) for broadcasting across bins; apply per-route acquisition multiplier
        eir_in_dt  = np.asarray(ns.eir_indoor)[:, None]
        eir_out_dt = np.asarray(ns.eir_outdoor)[:, None]

        # Euler fluxes
        new_inf  = b * Sh_mat * (eir_in_dt * pa_in_mat + eir_out_dt * pa_out_mat)  # (n_nodes, n_bins)
        recover  = r_per_step * Ih_mat                    # (n_nodes, n_bins)

        dS_death = death_dt[None, :] * Sh_mat
        dI_death = death_dt[None, :] * Ih_mat

        aging_out_S = aging_dt[None, :] * Sh_mat
        aging_out_I = aging_dt[None, :] * Ih_mat

        aging_in_S          = np.zeros((n_nodes, n_bins))
        aging_in_S[:, 1:]   = aging_out_S[:, :-1]
        N_total             = Sh_mat.sum(axis=1) + Ih_mat.sum(axis=1)  # (n_nodes,)
        # People aging out of the last bin exit the system (like natural death).
        # Births replenish exactly the total losses so population stays roughly constant.
        aging_in_S[:, 0]    = birth_dt * N_total  # births into youngest bin

        aging_in_I          = np.zeros((n_nodes, n_bins))
        aging_in_I[:, 1:]   = aging_out_I[:, :-1]

        Sh_new = np.maximum(Sh_mat - new_inf + recover + drug_clear - dS_death - aging_out_S + aging_in_S, 0.0)
        Ih_new = np.maximum(Ih_mat + new_inf - recover - drug_clear - dI_death - aging_out_I + aging_in_I, 0.0)

        # Age vaccinated-fraction and dose-time alongside Sh/Ih
        vf_mat = self._vaccinated_fraction.raw.reshape(n_nodes, n_bins)
        dt_mat = self._dose_time.raw.reshape(n_nodes, n_bins)
        vf_new, T_new = self._age_vaccinated_state(
            Sh_mat, Ih_mat, vf_mat, dt_mat,
            aging_dt, death_dt,
            Sh_new=Sh_new, Ih_new=Ih_new,
        )

        # Write all updated state back via UID indexing
        self._Sh[all_uids]                 = Sh_new.ravel()
        self._Ih[all_uids]                 = Ih_new.ravel()
        self._vaccinated_fraction[all_uids] = vf_new.ravel()
        self._dose_time[all_uids]           = T_new.ravel()

        self.results.n_susceptible[self.ti]  = float(Sh_new.sum())
        self.results.n_infected[self.ti]     = float(Ih_new.sum())
        self.results.new_infections[self.ti] = float(new_inf.sum())

        self._write_node_state(ns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_node_state(self, ns):
        """Write per-node human state to NodeState for the mosquito connector."""
        n_nodes = ns.grid.n_nodes
        c       = float(self.pars.prob_mosquito_infected_on_bite)
        Sh_mat  = self._Sh.raw.reshape(n_nodes, self._n_bins)
        Ih_mat  = self._Ih.raw.reshape(n_nodes, self._n_bins)
        Sh_sum  = Sh_mat.sum(axis=1)
        Ih_sum  = Ih_mat.sum(axis=1)
        N       = np.maximum(Sh_sum + Ih_sum, 1.0)
        p_inf   = c * Ih_sum / N
        for nid in range(n_nodes):
            ns.n_humans[nid]      = int(round(N[nid]))
            ns.p_inf_indoor[nid]  = p_inf[nid]
            ns.p_inf_outdoor[nid] = p_inf[nid]

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

    @staticmethod
    def _age_vaccinated_state(Sh, Ih, vaccinated_fraction, dose_time, aging_dt, death_dt,
                               Sh_new=None, Ih_new=None):
        """
        Age vaccinated_fraction and dose_time alongside Sh/Ih.

        Works with arrays of any leading shape: call with 2-D (n_nodes, n_bins)
        from step(), or 1-D (n_bins,) from unit tests.  aging_dt and death_dt
        are always 1-D (n_bins,) and broadcast from the right.

        Parameters
        ----------
        Sh, Ih               : ndarray (..., n_bins) — N_OLD (before Euler) — used to
                               compute V = vaccinated_fraction * N_old
        vaccinated_fraction  : ndarray (..., n_bins) — current vaccinated fraction per bin
        dose_time            : ndarray (..., n_bins) — population-weighted mean dose time
        aging_dt, death_dt   : ndarray (n_bins,) — per-bin rates already multiplied by dt
        Sh_new, Ih_new       : ndarray (..., n_bins) or None — N_NEW (after Euler) used as
                               denominator.  Falls back to N_OLD when None (unit-test mode).

        Returns
        -------
        new_fraction, new_dose_time : ndarray (..., n_bins) each
        """
        V        = vaccinated_fraction * (Sh + Ih)   # vaccinated count before aging (N_old)
        V_out    = V * (aging_dt + death_dt)          # (n_bins,) broadcasts against (..., n_bins)
        V_in     = np.zeros_like(V)
        V_in[..., 1:] = V_out[..., :-1]
        V_new    = np.maximum(V - V_out + V_in, 0.0)
        denom    = (Sh_new + Ih_new) if (Sh_new is not None and Ih_new is not None) else (Sh + Ih)
        safe_N   = np.where(denom > 0, denom, 1.0)
        new_fraction = np.where(denom > 0, V_new / safe_N, 0.0)
        T_new = np.where(
            V_new > 0,
            ((V - V_out) * dose_time + V_in * np.roll(dose_time, 1, axis=-1))
            / np.where(V_new > 0, V_new, 1.0),
            0.0,
        )
        T_new[..., 0] = 0.0   # births are unvaccinated
        return new_fraction, T_new

    # ------------------------------------------------------------------
    # Intervention interface (overrides MalariaHumanBase stubs)
    # ------------------------------------------------------------------

    _supports_intervention_effects = True

    def apply_coverage(self, uids, coverage):
        """Return (uids, weights) with coverage as a continuous fraction per compartment."""
        return uids, np.full(len(uids), float(coverage), dtype=float)

    def reset_acquisition_probability(self):
        auids = self.sim.people.auids
        self._prob_acquire_indoor[auids]  = 1.0
        self._prob_acquire_outdoor[auids] = 1.0

    def set_acquisition_probability(self, uids, efficacy, weights, target=BiteTarget.ALL):
        uids = ss.uids(np.asarray(uids))
        val  = 1.0 - weights * efficacy
        if target in (BiteTarget.ALL, BiteTarget.INDOOR):
            self._prob_acquire_indoor[uids]  = val
        if target in (BiteTarget.ALL, BiteTarget.OUTDOOR):
            self._prob_acquire_outdoor[uids] = val

    def pre_administer(self, uids, weights, t_now):
        uids  = ss.uids(np.asarray(uids))
        c_new = np.asarray(weights)
        f_old = np.asarray(self._vaccinated_fraction[uids])
        t_old = np.asarray(self._dose_time[uids])
        f_new = f_old + c_new * (1.0 - f_old)
        safe  = f_new > 0.0
        t_eff = np.where(
            safe,
            (f_old * t_old + c_new * (1.0 - f_old) * t_now) / np.where(safe, f_new, 1.0),
            t_now,
        )
        self._vaccinated_fraction[uids] = f_new
        self._dose_time[uids]           = t_eff
        return t_eff

    def sync_vaccinated_fraction(self, uids, efficacy):
        uids    = np.asarray(uids)
        expired = ss.uids(uids[np.asarray(efficacy[uids]) == 0.0])
        if len(expired) == 0:
            return
        self._vaccinated_fraction[expired] = 0.0
        self._dose_time[expired]           = 0.0

    def get_vaccinated_weights(self, uids):
        return np.asarray(self._vaccinated_fraction[ss.uids(np.asarray(uids))])

    def store_itn_fraction(self, uids, weights):
        """Store ITN coverage fraction per compartment (analogous to _vaccinated_fraction)."""
        self._itn_fraction[ss.uids(np.asarray(uids))] = np.asarray(weights)

    def get_itn_weights(self, uids):
        """Return per-compartment ITN coverage fraction for set_acquisition_probability."""
        return np.asarray(self._itn_fraction[ss.uids(np.asarray(uids))])

    def aggregate_itn_to_node_state(self, uids, combined_block, u_kill, ns):
        """Population-weighted NodeState aggregation for ODE compartments."""
        uids     = ss.uids(np.asarray(uids))
        node_ids = np.asarray(self.node_id[uids], dtype=int)
        n_total  = np.maximum(np.asarray(ns.n_humans, dtype=float), 1.0)
        pop_w    = np.asarray(self._itn_fraction[uids]) * (np.asarray(self._Sh[uids]) + np.asarray(self._Ih[uids]))
        ns.itn_biting_reduction += np.bincount(node_ids, weights=combined_block * pop_w, minlength=ns.grid.n_nodes) / n_total
        ns.itn_kill_prob        += np.bincount(node_ids, weights=u_kill         * pop_w, minlength=ns.grid.n_nodes) / n_total

    def set_drug_clearance(self, uids, mean_duration_years, weights):
        """
        Store drug state for active compartments.

        ``step()`` reads these each timestep and moves
        ``_drug_clearance_rate * dt * _drug_coverage * Ih`` from Ih to Sh until
        ``_drug_clear_end_t`` is reached, then resets both to sentinel values so
        the compartment can be re-dosed.
        """
        uids = ss.uids(np.asarray(uids))
        self._drug_coverage[uids]       = np.asarray(weights)
        self._drug_clear_end_t[uids]    = self.sim.t.year + mean_duration_years
        self._drug_clearance_rate[uids] = 1.0 / mean_duration_years

    def administer_outbreak(self, uids, weights):
        """Seed infections by moving weight * Sh[uid] susceptibles to infected."""
        uids      = ss.uids(np.asarray(uids))
        weights   = np.asarray(weights)
        sh_vals   = np.asarray(self._Sh[uids])
        to_infect = weights * sh_vals
        self._Sh[uids] = sh_vals - to_infect
        ih_vals        = np.asarray(self._Ih[uids])
        self._Ih[uids] = ih_vals + to_infect

    def validate_eligibility(self, eligibility):
        """Raise ValueError if eligibility age bounds don't align to ODE bin edges."""
        if self._age_bin_edges is None:
            return
        edges = self._age_bin_edges
        for attr in ('min_age', 'max_age'):
            val = getattr(eligibility, attr, None)
            if val is None:
                continue
            yr = val.years if isinstance(val, ss.TimePar) else float(val)
            if not any(abs(yr - e) < 1e-9 for e in edges):
                raise ValueError(
                    f"HumanEligibility {attr}={yr} does not align to an ODE age "
                    f"bin edge. Valid edges: {list(edges)}."
                )
