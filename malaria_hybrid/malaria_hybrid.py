"""
Ross-Macdonald malaria model (Starsim version) with interchangeable ODE and
agent-based modules for both humans and mosquitoes.

Classes
-------
HumanODE    — SIS human compartment (ODE) dynamics, registered as a custom module.
HumanABM    — SIS human agent-based dynamics; writes compartment counts to sim.c
              for interoperability with mosquito modules.
MosquitoODE — SEI mosquito compartment (ODE) dynamics with seasonal forcing.
MosquitoABM — SEI mosquito agent-based dynamics with seasonal forcing; manages its
              own pre-allocated numpy arrays separate from ss.People.
Malaria     — Results coordinator: records derived quantities (lambda_h, EIR, R0)
              and owns the combined plot.
Sim         — Custom ss.Sim subclass that owns shared state (sim.c) and shared
              parameters (sim.pars.a, sim.pars.Nh) and exposes modules as properties.
MultiSim    — Custom ss.MultiSim subclass with a combined prevalence plot.

Architecture
------------
Shared state
    sim.c          sc.objdict with Sh, Ih, Sm, Em, Im — written in-place each step.
    sim.pars.a     Biting rate (mosquito bites per mosquito per day).
    sim.pars.Nh    Human population size.

Execution order per timestep (Gauss-Seidel Euler, driven by the Starsim loop):
    1. Human step  (HumanODE or HumanABM)    — reads old Im, writes Sh / Ih.
    2. Mosquito step (MosquitoODE or MosquitoABM) — reads updated Ih, writes Sm / Em / Im.
    3. Malaria.update_results()               — records derived quantities.

The four module combinations (ODE×ODE, ABM×ODE, ODE×ABM, ABM×ABM) all share the
same sim.c interface, so any human module works with any mosquito module.

Example
-------
::

    from malaria_hybrid.malaria_hybrid import HumanODE, HumanABM, MosquitoODE, MosquitoABM, Sim

    sim_pars = dict(a=0.24, Nh=1000, start=0, stop=365*5, dt=1)
    # ODE × ODE
    sim = Sim(pars=sim_pars, human=HumanODE(b=0.15, r=1/120, Ih0=10),
              mosquito=MosquitoODE(c=0.15, mu_m=0.10))
    sim.run()
    sim.malaria.plot()

    # ABM × ABM
    sim2 = Sim(pars=sim_pars, human=HumanABM(b=0.15, r=1/120, Ih0=10),
               mosquito=MosquitoABM(c=0.15, mu_m=0.10))
    sim2.run()
    sim2.malaria.plot()
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

eps = 1e-8  # Guard against divide-by-zero


# =============================================================================
# HumanODE
# =============================================================================

class HumanODE(ss.Module):
    """
    SIS compartment dynamics for the human population.

    Reads shared pars (a, Nh) from sim.pars and compartment state from sim.c.
    Writes Sh and Ih in-place every timestep.

    Module-specific parameters
    --------------------------
    b : float
        Mosquito-to-human transmission probability per bite.
    r : float
        Human recovery rate per day.
    Ih0 : float
        Initial number of infectious humans.

    Shared parameters (set on Sim)
    --------------------------------
    sim.pars.a   — biting rate
    sim.pars.Nh  — human population size
    """

    def __init__(self, **kwargs):
        super().__init__(name='human')
        self.define_pars(
            b   = 0.15,   # mosquito-to-human transmission probability per bite
            r   = 1/120,  # human recovery rate per day
            Ih0 = 10,     # initial infectious humans
        )
        self.update_pars(**kwargs)

    def init_post(self):
        """Initialize Sh and Ih in the shared compartment dict."""
        super().init_post()
        Nh  = self.sim.pars.Nh
        Ih0 = min(self.pars.Ih0, Nh)
        self.sim.c.Sh = Nh - Ih0
        self.sim.c.Ih = Ih0

    def step(self):
        """Euler update for Sh and Ih."""
        comp = self.sim.c
        p    = self.pars
        a    = self.sim.pars.a
        Nh   = self.sim.pars.Nh
        dt   = self.dt.days

        M        = comp.Sm + comp.Em + comp.Im
        M_eff    = max(M, eps)
        lambda_h = a * p.b * (comp.Im / M_eff) * (M / Nh)

        dSh = -lambda_h * comp.Sh + p.r * comp.Ih
        dIh =  lambda_h * comp.Sh - p.r * comp.Ih

        comp.Sh = max(comp.Sh + dSh * dt, 0.0)
        comp.Ih = max(comp.Ih + dIh * dt, 0.0)

    def init_results(self):
        """Define human compartment results."""
        super().init_results()
        self.define_results(
            ss.Result('Sh',     label='Susceptible humans'),
            ss.Result('Ih',     label='Infectious humans'),
            ss.Result('prev_h', label='Human prevalence (Ih/Nh)'),
        )

    def update_results(self):
        """Record human compartment values."""
        super().update_results()
        ti   = self.ti
        comp = self.sim.c
        Nh_eff = max(comp.Sh + comp.Ih, eps)

        self.results.Sh[ti]     = comp.Sh
        self.results.Ih[ti]     = comp.Ih
        self.results.prev_h[ti] = comp.Ih / Nh_eff


# =============================================================================
# MosquitoODE
# =============================================================================

class MosquitoODE(ss.Module):
    """
    SEI compartment dynamics for the mosquito population with seasonal forcing.

    Reads shared pars (a, Nh) from sim.pars and compartment state from sim.c.
    Called *after* HumanODE each timestep, so reads the freshly updated Ih.
    Writes Sm, Em, and Im in-place every timestep.

    Module-specific parameters
    --------------------------
    c : float
        Human-to-mosquito transmission probability per bite.
    mu_m : float
        Mosquito death rate per day.
    sigma_m : float
        Mosquito incubation completion rate per day.
    m0 : float
        Baseline mosquito-to-human ratio.
    Im0 : float
        Initial number of infectious mosquitoes.
    sn_amp, sn_phase, sn_peak, sn_relax : float
        Seasonality parameters.
    year_len : float
        Length of the year in days (used for seasonality and plot x-axis).

    Shared parameters (set on Sim)
    --------------------------------
    sim.pars.a   — biting rate
    sim.pars.Nh  — human population size
    """

    def __init__(self, **kwargs):
        super().__init__(name='mosquito')
        self.define_pars(
            c        = 0.15,   # human-to-mosquito transmission probability per bite
            mu_m     = 0.10,   # mosquito death rate per day
            sigma_m  = 1/12,   # mosquito incubation completion rate per day
            m0       = 5,      # baseline mosquito:human ratio
            Im0      = 100,    # initial infectious mosquitoes
            sn_amp   = 0.65,   # seasonality amplitude
            sn_phase = 120,    # peak timing shift in days
            sn_peak  = 1.5,    # sharpness of the seasonal peak
            sn_relax = 0.15,   # speed mosquitoes track the seasonal target
            year_len = 365,    # length of the year in days
        )
        self.update_pars(**kwargs)

    def init_post(self):
        """Initialize Sm, Em, Im in the shared compartment dict."""
        super().init_post()
        p   = self.pars
        Nh  = self.sim.pars.Nh
        M0  = p.m0 * Nh
        Im0 = min(p.Im0, M0)
        self.sim.c.Sm = M0 - Im0
        self.sim.c.Em = 0.0
        self.sim.c.Im = Im0

    def seasonality(self, t):
        """
        Seasonal multiplier for mosquito abundance.

        Mean is approximately 1, bounded away from negative values.
        """
        p = self.pars
        x = 2 * np.pi * (t - p.sn_phase) / p.year_len
        s = 1 + p.sn_amp * np.sin(x)

        if p.sn_peak != 1.0:
            s     = np.maximum(s, eps)
            s     = s ** p.sn_peak
            tt    = np.linspace(0, p.year_len, 2000)
            xx    = 2 * np.pi * (tt - p.sn_phase) / p.year_len
            s_ref = np.maximum(1 + p.sn_amp * np.sin(xx), eps) ** p.sn_peak
            s    /= s_ref.mean()  # renormalize so annual mean ≈ 1

        return s

    def target_mos_pop(self, t):
        """Seasonal target mosquito population M*(t)."""
        return self.sim.pars.Nh * self.pars.m0 * self.seasonality(t)

    def step(self):
        """Euler update for Sm, Em, and Im."""
        comp = self.sim.c
        p    = self.pars
        a    = self.sim.pars.a
        t    = self.now
        dt   = self.dt.days

        M          = comp.Sm + comp.Em + comp.Im
        M_eff      = max(M, eps)
        Nh_eff     = max(comp.Sh + comp.Ih, eps)
        lambda_m   = a * p.c * (comp.Ih / Nh_eff)
        M_target   = self.target_mos_pop(t)
        births     = p.sn_relax * max(M_target - M, 0.0)
        extra_loss = p.sn_relax * max(M - M_target, 0.0)

        dSm = births     - lambda_m * comp.Sm - p.mu_m * comp.Sm - extra_loss * (comp.Sm / M_eff)
        dEm = lambda_m * comp.Sm - p.sigma_m * comp.Em - p.mu_m * comp.Em - extra_loss * (comp.Em / M_eff)
        dIm = p.sigma_m * comp.Em - p.mu_m * comp.Im - extra_loss * (comp.Im / M_eff)

        comp.Sm = max(comp.Sm + dSm * dt, 0.0)
        comp.Em = max(comp.Em + dEm * dt, 0.0)
        comp.Im = max(comp.Im + dIm * dt, 0.0)

    def init_results(self):
        """Define mosquito compartment results."""
        super().init_results()
        self.define_results(
            ss.Result('Sm',     label='Susceptible mosquitoes'),
            ss.Result('Em',     label='Exposed mosquitoes'),
            ss.Result('Im',     label='Infectious mosquitoes'),
            ss.Result('prev_m', label='Mosquito prevalence (Im/M)'),
            ss.Result('m',      label='Mosquito:human ratio'),
        )

    def update_results(self):
        """Record mosquito compartment values."""
        super().update_results()
        ti   = self.ti
        comp = self.sim.c
        Nh   = self.sim.pars.Nh

        M     = comp.Sm + comp.Em + comp.Im
        M_eff = max(M, eps)

        self.results.Sm[ti]     = comp.Sm
        self.results.Em[ti]     = comp.Em
        self.results.Im[ti]     = comp.Im
        self.results.prev_m[ti] = comp.Im / M_eff
        self.results.m[ti]      = M / Nh


class HumanABM(ss.Module):
    """
    SIS agent-based dynamics for the human population.

    Each Starsim agent is individually susceptible or infectious. Transitions
    are drawn stochastically each timestep using the same force-of-infection
    formula as HumanODE. After updating agent states, Sh and Ih counts are
    written to sim.c so that MosquitoODE and Malaria can read them unchanged.

    States (on sim.people via define_states)
    ----------------------------------------
    infectious : BoolArr
        True when the agent is currently infectious.

    Module-specific parameters
    --------------------------
    b : float
        Mosquito-to-human transmission probability per bite.
    r : float
        Human recovery rate per day.
    Ih0 : float
        Initial number of infectious humans.

    Shared parameters (set on Sim)
    --------------------------------
    sim.pars.a   — biting rate
    sim.pars.Nh  — human population size
    """

    def __init__(self, **kwargs):
        super().__init__(name='human')
        self.define_pars(
            b   = 0.15,   # mosquito-to-human transmission probability per bite
            r   = 1/120,  # human recovery rate per day
            Ih0 = 10,     # initial infectious humans
        )
        self.define_states(
            ss.BoolArr('infectious', default=False, label='Infectious'),
        )
        self.update_pars(**kwargs)

    def init_post(self):
        """Seed initial infections and write starting compartment values."""
        super().init_post()
        Ih0 = int(min(self.pars.Ih0, len(self.sim.people)))
        if Ih0 > 0:
            self.infectious.raw[:Ih0] = True   # positional: first Ih0 agents
        self._sync_compartments()

    def _sync_compartments(self):
        """Write alive-agent counts to sim.c for MosquitoODE / Malaria interop."""
        alive = self.sim.people.alive.raw   # positional bool array (all agents)
        inf   = self.infectious.raw         # positional bool array (all agents)
        n_inf   = int((alive & inf).sum())
        n_alive = int(alive.sum())
        self.sim.c.Ih = n_inf
        self.sim.c.Sh = n_alive - n_inf

    def step(self):
        """Stochastic per-agent infection and recovery transitions."""
        comp = self.sim.c
        p    = self.pars
        a    = self.sim.pars.a
        Nh   = self.sim.pars.Nh
        dt   = self.dt.days

        M        = comp.Sm + comp.Em + comp.Im
        M_eff    = max(M, eps)
        lambda_h = a * p.b * (comp.Im / M_eff) * (M / Nh)

        # Convert rates to per-step probabilities (continuous-time approximation)
        p_inf = 1.0 - np.exp(-lambda_h * dt)
        p_rec = 1.0 - np.exp(-p.r * dt)

        alive = self.sim.people.alive.raw   # positional bool array (all agents)
        inf   = self.infectious.raw         # positional bool array (all agents)
        uids  = self.sim.people.uid.raw     # positional UID array

        # Susceptible → Infectious
        sus_uids = uids[alive & ~inf]
        if len(sus_uids):
            newly_infected = sus_uids[np.random.random(len(sus_uids)) < p_inf]
            self.infectious[newly_infected] = True   # UID-based assignment

        # Infectious → Susceptible (recovery)
        inf_uids = uids[alive & inf]
        if len(inf_uids):
            newly_recovered = inf_uids[np.random.random(len(inf_uids)) < p_rec]
            self.infectious[newly_recovered] = False  # UID-based assignment

        self._sync_compartments()

    def init_results(self):
        """Define human compartment results (identical to HumanODE for interop)."""
        super().init_results()
        self.define_results(
            ss.Result('Sh',     label='Susceptible humans'),
            ss.Result('Ih',     label='Infectious humans'),
            ss.Result('prev_h', label='Human prevalence (Ih/Nh)'),
        )

    def update_results(self):
        """Record human compartment values from sim.c."""
        super().update_results()
        ti     = self.ti
        comp   = self.sim.c
        Nh_eff = max(comp.Sh + comp.Ih, eps)

        self.results.Sh[ti]     = comp.Sh
        self.results.Ih[ti]     = comp.Ih
        self.results.prev_h[ti] = comp.Ih / Nh_eff


class MosquitoABM(ss.Module):
    """
    SEI agent-based mosquito dynamics with seasonal forcing.

    Manages its own pre-allocated numpy arrays (separate from ss.People) to
    avoid mixing mosquito and human agents in the same People object.
    Implements the same SEI dynamics and seasonal population-control logic as
    MosquitoODE, but with individual stochastic mosquito agents.

    Called *after* HumanODE/HumanABM each timestep, so reads the freshly
    updated Ih from sim.c.  Writes Sm, Em, Im in-place every timestep.

    Population dynamics
    -------------------
    Deaths are pre-scheduled at birth from Exponential(1/mu_m). When M
    exceeds the seasonal target, extra deaths are drawn at rate
    sn_relax*(M-M_target)/M per day, mirroring the ODE extra-loss term.
    Births top up toward M_target at rate sn_relax*(M_target-M) per day,
    drawn as Poisson(births*dt) each step.

    Module-specific parameters (identical to MosquitoODE)
    -------------------------------------------------------
    c, mu_m, sigma_m, m0, Im0, sn_amp, sn_phase, sn_peak, sn_relax, year_len

    Shared parameters (set on Sim)
    --------------------------------
    sim.pars.a   — biting rate
    sim.pars.Nh  — human population size
    """

    def __init__(self, **kwargs):
        super().__init__(name='mosquito')
        self.define_pars(
            c        = 0.15,   # human-to-mosquito transmission probability per bite
            mu_m     = 0.10,   # mosquito death rate per day
            sigma_m  = 1/12,   # mosquito incubation completion rate per day
            m0       = 5,      # baseline mosquito:human ratio
            Im0      = 100,    # initial infectious mosquitoes
            sn_amp   = 0.65,   # seasonality amplitude
            sn_phase = 120,    # peak timing shift in days
            sn_peak  = 1.5,    # sharpness of the seasonal peak
            sn_relax = 0.15,   # speed mosquitoes track the seasonal target
            year_len = 365,    # length of the year in days
        )
        self.update_pars(**kwargs)

        # Agent arrays — allocated in init_post once Nh is known
        self._max_mos    = 0
        self.alive       = None   # bool: slot is occupied by a live mosquito
        self.susceptible = None   # bool
        self.exposed     = None   # bool
        self.infectious  = None   # bool
        self.ti_dead     = None   # float: step index when mosquito dies

    def init_post(self):
        """Allocate arrays, seed initial population, initialize sim.c."""
        super().init_post()
        p    = self.pars
        Nh   = self.sim.pars.Nh
        dt   = self.dt.days
        ti   = self.ti

        M0    = p.m0 * Nh
        Im0   = int(min(p.Im0, M0))
        Sm0   = int(round(M0)) - Im0
        n_init = Sm0 + Im0

        # Buffer: 3× peak seasonal equilibrium
        t_sample = np.linspace(0, p.year_len, 100)
        max_s    = max(self.seasonality(t) for t in t_sample)
        self._max_mos = max(100, int(M0 * max_s * 3))

        self.alive       = np.zeros(self._max_mos, dtype=bool)
        self.susceptible = np.zeros(self._max_mos, dtype=bool)
        self.exposed     = np.zeros(self._max_mos, dtype=bool)
        self.infectious  = np.zeros(self._max_mos, dtype=bool)
        self.ti_dead     = np.full(self._max_mos, np.inf)

        if n_init > 0:
            mean_life = 1.0 / (p.mu_m * dt)   # expected lifespan in steps
            self.alive[:n_init]          = True
            self.susceptible[:Sm0]       = True
            self.infectious[Sm0:n_init]  = True
            self.ti_dead[:n_init] = ti + np.random.exponential(mean_life, n_init)

        self._sync_compartments()

    def _sync_compartments(self):
        """Write alive-agent counts to sim.c for HumanODE/HumanABM / Malaria."""
        alive = self.alive
        self.sim.c.Sm = (alive & self.susceptible).sum()
        self.sim.c.Em = (alive & self.exposed).sum()
        self.sim.c.Im = (alive & self.infectious).sum()

    def _kill(self, mask):
        """Clear all state for mosquitoes selected by a boolean mask."""
        self.alive[mask]       = False
        self.susceptible[mask] = False
        self.exposed[mask]     = False
        self.infectious[mask]  = False
        self.ti_dead[mask]     = np.inf

    def seasonality(self, t):
        """Seasonal multiplier (identical to MosquitoODE.seasonality)."""
        p = self.pars
        x = 2 * np.pi * (t - p.sn_phase) / p.year_len
        s = 1 + p.sn_amp * np.sin(x)
        if p.sn_peak != 1.0:
            s     = np.maximum(s, eps)
            s     = s ** p.sn_peak
            tt    = np.linspace(0, p.year_len, 2000)
            xx    = 2 * np.pi * (tt - p.sn_phase) / p.year_len
            s_ref = np.maximum(1 + p.sn_amp * np.sin(xx), eps) ** p.sn_peak
            s    /= s_ref.mean()
        return s

    def target_mos_pop(self, t):
        """Seasonal target mosquito population M*(t)."""
        return self.sim.pars.Nh * self.pars.m0 * self.seasonality(t)

    def step(self):
        """Advance the mosquito population by one timestep."""
        comp = self.sim.c
        p    = self.pars
        a    = self.sim.pars.a
        t    = self.now
        dt   = self.dt.days
        ti   = self.ti

        M          = int(self.alive.sum())
        M_eff      = max(M, eps)
        Nh_eff     = max(comp.Sh + comp.Ih, eps)
        lambda_m   = a * p.c * (comp.Ih / Nh_eff)
        M_target   = self.target_mos_pop(t)
        births     = p.sn_relax * max(M_target - M, 0.0)
        extra_loss = p.sn_relax * max(M - M_target, 0.0)

        # 1. Base deaths (pre-scheduled at birth)
        dying = self.alive & (self.ti_dead <= ti)
        if dying.any():
            self._kill(dying)

        # 2. Extra deaths (population excess over seasonal target)
        if extra_loss > 0.0:
            p_extra = min(extra_loss * dt / M_eff, 1.0)
            extra_mask = self.alive & (np.random.random(self._max_mos) < p_extra)
            if extra_mask.any():
                self._kill(extra_mask)

        # 3. Births — Poisson draw, fill free slots
        n_births = int(np.random.poisson(births * dt))
        if n_births > 0:
            free = np.where(~self.alive)[0]
            if len(free) < n_births:
                print(f'Warning: MosquitoABM buffer full at step {ti}; '
                      f'{n_births - len(free)} births dropped.')
                n_births = len(free)
            if n_births > 0:
                slots      = free[:n_births]
                mean_life  = 1.0 / (p.mu_m * dt)
                self.alive[slots]       = True
                self.susceptible[slots] = True
                self.exposed[slots]     = False
                self.infectious[slots]  = False
                self.ti_dead[slots]     = ti + np.random.exponential(mean_life, n_births)

        # 4. Exposed → Infectious (EIP completion — Bernoulli each step)
        exp_mask = self.alive & self.exposed
        if exp_mask.any():
            p_ei      = 1.0 - np.exp(-p.sigma_m * dt)
            turn_mask = exp_mask & (np.random.random(self._max_mos) < p_ei)
            if turn_mask.any():
                self.exposed[turn_mask]    = False
                self.infectious[turn_mask] = True

        # 5. Susceptible → Exposed (H→M transmission — Bernoulli each step)
        sus_mask = self.alive & self.susceptible
        if sus_mask.any():
            p_se        = 1.0 - np.exp(-lambda_m * dt)
            infect_mask = sus_mask & (np.random.random(self._max_mos) < p_se)
            if infect_mask.any():
                self.susceptible[infect_mask] = False
                self.exposed[infect_mask]     = True

        self._sync_compartments()

    def init_results(self):
        """Define mosquito compartment results (identical to MosquitoODE for interop)."""
        super().init_results()
        self.define_results(
            ss.Result('Sm',     label='Susceptible mosquitoes'),
            ss.Result('Em',     label='Exposed mosquitoes'),
            ss.Result('Im',     label='Infectious mosquitoes'),
            ss.Result('prev_m', label='Mosquito prevalence (Im/M)'),
            ss.Result('m',      label='Mosquito:human ratio'),
        )

    def update_results(self):
        """Record mosquito compartment values from sim.c."""
        super().update_results()
        ti   = self.ti
        comp = self.sim.c
        Nh   = self.sim.pars.Nh

        M     = comp.Sm + comp.Em + comp.Im
        M_eff = max(M, eps)

        self.results.Sm[ti]     = comp.Sm
        self.results.Em[ti]     = comp.Em
        self.results.Im[ti]     = comp.Im
        self.results.prev_m[ti] = comp.Im / M_eff
        self.results.m[ti]      = M / Nh

# =============================================================================
# Malaria
# =============================================================================

class Malaria(ss.Module):
    """
    Results coordinator for derived quantities (lambda_h, EIR, R0) and plot.

    Has no pars of its own.  Reads everything it needs from sim.pars, sim.human,
    sim.mosquito, and sim.c.  step() is a no-op — dynamics are in HumanODE and
    MosquitoODE.
    """

    def __init__(self):
        super().__init__(name='malaria')

    def step(self):
        pass  # HumanODE and MosquitoODE drive all dynamics

    def init_results(self):
        """Define derived quantity results."""
        super().init_results()
        self.define_results(
            ss.Result('lambda_h', label='Human force of infection'),
            ss.Result('eir',      label='EIR (infectious bites per human per day)'),
            ss.Result('R0',       label='R0(t)'),
        )

    def update_results(self):
        """Record derived quantities that depend on both human and mosquito state."""
        super().update_results()
        ti   = self.ti
        comp = self.sim.c
        a    = self.sim.pars.a
        Nh   = self.sim.pars.Nh
        hp   = self.sim.human.pars
        mp   = self.sim.mosquito.pars

        M     = comp.Sm + comp.Em + comp.Im
        M_eff = max(M, eps)
        m     = M / Nh

        self.results.lambda_h[ti] = a * hp.b * (comp.Im / M_eff) * (M / Nh)
        self.results.eir[ti]      = a * comp.Im / Nh
        self.results.R0[ti]       = (m * a**2 * hp.b * mp.c * mp.sigma_m) / (
            hp.r * mp.mu_m * (mp.sigma_m + mp.mu_m)
        )

    def plot(self, do_show=True, **kwargs):
        """Plot prevalence, R0, mosquito density, and force of infection."""
        hr  = self.sim.human.results
        mr  = self.sim.mosquito.results
        r   = self.results
        t   = mr.timevec / self.sim.mosquito.pars.year_len   # convert to years
        kw  = sc.mergedicts(dict(lw=1), kwargs)

        with sc.options.with_style('fancy'):
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
            fig.suptitle('Ross-Macdonald model with seasonality', fontsize=14)

            ax0.plot(t, hr['prev_h'].values, label='Human prevalence (Ih/Nh)', **kw)
            ax0.plot(t, mr['prev_m'].values, label='Mosquito prevalence (Im/M)', **kw)
            ax0.set_ylabel('Prevalence')
            ax0.legend()
            sc.boxoff(ax0)

            ax1.plot(t, r['R0'].values, label='R0(t)', **kw)
            ax1.axhline(1.0, linestyle='--', linewidth=1, color='gray')
            ax1.set_ylabel('R0')
            ax1.legend()
            sc.boxoff(ax1)

            ax2.plot(t, mr['m'].values, label='m(t) = mosquitoes per human', **kw)
            ax2.set_ylabel('Mosquitoes per human')
            ax2.legend()
            sc.boxoff(ax2)

            ax3.plot(t, r['lambda_h'].values, label='Human FOI λh', **kw)
            ax3.plot(t, r['eir'].values, label='EIR', alpha=0.8, **kw)
            ax3.set_ylabel('Rate per day')
            ax3.set_xlabel('Time (years)')
            ax3.legend()
            sc.boxoff(ax3)

            sc.figlayout()

        return ss.return_fig(fig, do_show=do_show)


# =============================================================================
# Sim
# =============================================================================

class Sim(ss.Sim):
    """
    Custom Sim that owns shared malaria model state and parameters.

    Shared state
    ------------
    sim.c          : sc.objdict — compartments Sh, Ih, Sm, Em, Im (written each step).
    sim.pars.a     : float      — biting rate (mosquito bites per mosquito per day).
    sim.pars.Nh    : float      — human population size.

    Module properties (backed by sim.custom.*)
    ------------------------------------------
    sim.human    → sim.custom.human
    sim.mosquito → sim.custom.mosquito
    sim.malaria  → sim.custom.malaria

    Parameters
    ----------
    pars : dict
        May contain shared pars (a, Nh) alongside standard ss.Sim pars
        (start, stop, dt, n_agents, ...).
    human : HumanODE, optional
        Pre-configured human module (default: HumanODE()).
    mosquito : MosquitoODE, optional
        Pre-configured mosquito module (default: MosquitoODE()).

    Example
    -------
    ::

        sim_pars = dict(a=0.24, Nh=1000, start=0, stop=365*5, dt=1, n_agents=1)
        hum = HumanODE(b=0.15, r=1/120, Ih0=10)
        mos = MosquitoODE(c=0.15, mu_m=0.10)
        sim = Sim(pars=sim_pars, human=hum, mosquito=mos)
        sim.run()
        sim.malaria.plot()
    """

    def __init__(self, human=None, mosquito=None, pars=None, **kwargs):
        pars = dict(pars or {})  # make a mutable copy

        # Extract shared pars before passing to ss.Sim (which would reject unknown keys)
        a  = pars.pop('a',  0.24)
        Nh = pars.pop('Nh', 1000)

        human    = human    if human    is not None else HumanODE()
        mosquito = mosquito if mosquito is not None else MosquitoODE()
        malaria  = Malaria()

        # Handle number of agents (TODO, should be able to skip People entirely in compartmental mode)
        pars['n_agents'] = 1 if isinstance(human, HumanODE) else Nh

        # Handle dt -- TODO, make more elegant
        pars.setdefault('dt', ss.days(1))
        if sc.isnumber(pars['dt']):
            pars['dt'] = ss.days(pars['dt'])

        # Shared compartment state — initialized to zero; each module fills its
        # own compartments in init_post() once sim.pars.Nh is available
        self.c = sc.objdict(Sh=0.0, Ih=0.0, Sm=0.0, Em=0.0, Im=0.0)

        super().__init__(
            pars        = pars,
            custom      = [human, mosquito, malaria],
            copy_inputs = False,
            **kwargs
        )

        # Add shared pars to sim.pars using create=True (they don't exist in SimPars)
        self.pars.update(dict(a=a, Nh=Nh), create=True) # TODO: there are more efficient ways to do this

    # -------------------------------------------------------------------------
    # Properties for quick module access
    # -------------------------------------------------------------------------

    @property
    def human(self):
        """Quick access to the human module."""
        return self.custom.human

    @property
    def mosquito(self):
        """Quick access to the mosquito module."""
        return self.custom.mosquito

    @property
    def malaria(self):
        """Quick access to the malaria results module."""
        return self.custom.malaria

    def plot_malaria(self, **kwargs):
        """ Quick access to plotting malaria results """
        return self.malaria.plot(**kwargs)


# =============================================================================
# MultiSim
# =============================================================================

class MultiSim(ss.MultiSim):
    """
    Thin ss.MultiSim subclass with a combined human/mosquito prevalence plot.

    Each contained sim must be a malaria_hybrid.Sim instance (or any Sim that
    exposes sim.human.results.prev_h and sim.mosquito.results.prev_m).

    Methods
    -------
    plot_malaria(**kw)
        Two-panel figure: human prevalence (top) and mosquito prevalence (bottom),
        one line per sim labelled by sim.label, x-axis in years.
    """
    def plot_malaria(self, **kw):
        # Plot
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle('Ross-Macdonald model with seasonality', fontsize=14)
        ax0.set_title('Human prevalence (Ih/Nh)')
        ax1.set_title('Mosquito prevalence (Im/M)')

        for sim in self.sims:
            hr  = sim.human.results
            mr  = sim.mosquito.results
            t   = mr.timevec / sim.mosquito.pars.year_len
            ax0.plot(t, hr.prev_h, label=sim.label, **kw)
            ax1.plot(t, mr.prev_m, label=sim.label, **kw)

        for ax in (ax0, ax1):
            ax.set_ylabel('Prevalence')
            ax.set_xlabel('Time (years)')
            ax.legend()
            sc.boxoff(ax)

        return fig


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    do_show = True
    Nh      = 1000
    hum_kw  = dict(b=0.15, r=1/120, Ih0=10)
    mos_kw  = dict(c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100)

    combos = [
        ('ode', HumanODE,  'ode', MosquitoODE),
        ('abm', HumanABM,  'ode', MosquitoODE),
        ('ode', HumanODE,  'abm', MosquitoABM),
        ('abm', HumanABM,  'abm', MosquitoABM),
    ]

    for hl, H, ml, M in combos:
        label    = f'human={hl}  mosquito={ml}'
        n_agents = Nh if hl == 'abm' else 1
        sim_pars = dict(a=0.24, Nh=Nh, start=0, stop=365*5, dt=1, n_agents=n_agents)
        print(f'\n=== {label} ===')
        sim = Sim(pars=sim_pars, human=H(**hum_kw), mosquito=M(**mos_kw))
        sim.run()
        sim.plot_malaria(do_show=do_show)
