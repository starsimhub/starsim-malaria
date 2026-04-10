"""
Ross-Macdonald malaria model (Starsim version).

Converts the standalone ODE class into a Starsim Module with Euler integration,
following the pattern of malaria_demo/demo_sim_ss.py.

Example
-------
::

    import starsim as ss
    from malaria_hybrid.rossmac_ss import RossMac_SS

    mod = RossMac_SS()
    sim = ss.Sim(modules=mod, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
    sim.run()
    mod.plot()
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

eps = 1e-8  # Avoid floating point errors

default_pars = sc.objdict(
    Nh       = 1000,   # number of humans
    m0       = 5,      # baseline mosquito:human ratio
    a        = 0.24,   # bites per mosquito per day
    b        = 0.15,   # mosquito-to-human transmission probability per bite
    c        = 0.15,   # human-to-mosquito transmission probability per bite
    r        = 1/120,  # human recovery rate per day
    mu_m     = 0.10,   # mosquito death rate per day
    sigma_m  = 1/12,   # mosquito incubation completion rate per day
    sn_amp   = 0.65,   # seasonality amplitude
    sn_phase = 120,    # peak timing shift in days
    sn_peak  = 1.5,    # sharpness of the seasonal peak
    sn_relax = 0.15,   # speed mosquitoes track the seasonal target
    Ih0      = 10,     # initial infectious humans
    Im0      = 100,    # initial infectious mosquitoes
    year_len = 365,    # length of the year in days
)


class RossMac_SS(ss.Module):
    """
    Compartmental Starsim implementation of the seasonal Ross-Macdonald malaria model.

    Compartments (scalars, not per-agent):
        Sh  — susceptible humans
        Ih  — infectious humans
        Sm  — susceptible mosquitoes
        Em  — exposed mosquitoes
        Im  — infectious mosquitoes

    Notes
    -----
    - Human population Nh is held constant (SIS structure).
    - Mosquito population varies seasonally toward a time-varying carrying capacity.
    - R0(t) uses the instantaneous mosquito:human ratio m(t) = M(t)/Nh.
    - Does not need a network or ss.People — self-contained compartmental model.

    Example
    -------
    ::

        import starsim as ss
        from malaria_hybrid.rossmac_ss import RossMac_SS

        mod = RossMac_SS()
        sim = ss.Sim(modules=mod, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
        sim.run()
        sim.modules.rossmac_ss.plot()
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(**default_pars)
        self.update_pars(**kwargs)

        # Compartments (scalars, not per-agent states)
        self.c = sc.objdict(Sh=0.0, Ih=0.0, Sm=0.0, Em=0.0, Im=0.0)
        return

    def seasonality(self, t):
        """
        Seasonal multiplier for mosquito abundance.
        Mean is approximately 1, bounded away from negative values.
        """
        p = self.pars
        x = 2 * np.pi * (t - p.sn_phase) / p.year_len
        s = 1 + p.sn_amp * np.sin(x)

        if p.sn_peak != 1.0:
            s = np.maximum(s, eps)
            s = s ** p.sn_peak
            tt = np.linspace(0, p.year_len, 2000)
            xx = 2 * np.pi * (tt - p.sn_phase) / p.year_len
            s_ref = np.maximum(1 + p.sn_amp * np.sin(xx), eps) ** p.sn_peak
            s /= s_ref.mean()  # renormalize so annual mean ≈ 1

        return s

    def target_mos_pop(self, t):
        """Seasonal target mosquito population M*(t)."""
        p = self.pars
        return p.Nh * p.m0 * self.seasonality(t)

    def init_post(self):
        """Set initial compartment values from parameters."""
        super().init_post()
        p = self.pars
        M0 = p.m0 * p.Nh

        Ih0 = min(p.Ih0, p.Nh)
        Im0 = min(p.Im0, M0)

        self.c.Sh = p.Nh - Ih0
        self.c.Ih = Ih0
        self.c.Sm = M0 - Im0
        self.c.Em = 0.0
        self.c.Im = Im0
        return

    def step(self):
        """Euler integration of the Ross-Macdonald ODE system."""
        p = self.pars
        c = self.c
        t = self.now
        dt = self.dt

        M = c.Sm + c.Em + c.Im
        M_eff = max(M, eps)
        Nh_eff = max(c.Sh + c.Ih, eps)

        # Forces of infection
        lambda_h = p.a * p.b * (c.Im / M_eff) * (M / p.Nh)
        lambda_m = p.a * p.c * (c.Ih / Nh_eff)

        # Human derivatives (SIS)
        dSh = -lambda_h * c.Sh + p.r * c.Ih
        dIh =  lambda_h * c.Sh - p.r * c.Ih

        # Mosquito derivatives (SEI) with seasonal relaxation
        M_target = self.target_mos_pop(t)
        births     = p.sn_relax * max(M_target - M, 0.0)
        extra_loss = p.sn_relax * max(M - M_target, 0.0)

        dSm = births     - lambda_m * c.Sm - p.mu_m * c.Sm - extra_loss * (c.Sm / M_eff)
        dEm = lambda_m * c.Sm - p.sigma_m * c.Em - p.mu_m * c.Em - extra_loss * (c.Em / M_eff)
        dIm = p.sigma_m * c.Em - p.mu_m * c.Im - extra_loss * (c.Im / M_eff)

        # Euler update (clamp to non-negative)
        c.Sh = max(c.Sh + dSh * dt, 0.0)
        c.Ih = max(c.Ih + dIh * dt, 0.0)
        c.Sm = max(c.Sm + dSm * dt, 0.0)
        c.Em = max(c.Em + dEm * dt, 0.0)
        c.Im = max(c.Im + dIm * dt, 0.0)
        return

    def init_results(self):
        """Initialize results for compartments and derived quantities."""
        super().init_results()
        self.define_results(
            ss.Result('Sh',       label='Susceptible humans'),
            ss.Result('Ih',       label='Infectious humans'),
            ss.Result('Sm',       label='Susceptible mosquitoes'),
            ss.Result('Em',       label='Exposed mosquitoes'),
            ss.Result('Im',       label='Infectious mosquitoes'),
            ss.Result('prev_h',   label='Human prevalence (Ih/Nh)'),
            ss.Result('prev_m',   label='Mosquito prevalence (Im/M)'),
            ss.Result('eir',      label='EIR (infectious bites per human per day)'),
            ss.Result('lambda_h', label='Human force of infection'),
            ss.Result('R0',       label='R0(t)'),
            ss.Result('m',        label='Mosquito:human ratio'),
        )
        return

    def update_results(self):
        """Store current state and derived quantities."""
        super().update_results()
        ti = self.ti
        c = self.c
        p = self.pars

        M = c.Sm + c.Em + c.Im
        M_eff = max(M, eps)
        Nh_eff = max(c.Sh + c.Ih, eps)

        m = M / p.Nh
        lambda_h = p.a * p.b * (c.Im / M_eff) * (M / p.Nh)
        eir = p.a * c.Im / p.Nh
        R0 = (m * p.a**2 * p.b * p.c * p.sigma_m) / (
            p.r * p.mu_m * (p.sigma_m + p.mu_m)
        )

        self.results.Sh[ti]       = c.Sh
        self.results.Ih[ti]       = c.Ih
        self.results.Sm[ti]       = c.Sm
        self.results.Em[ti]       = c.Em
        self.results.Im[ti]       = c.Im
        self.results.prev_h[ti]   = c.Ih / Nh_eff
        self.results.prev_m[ti]   = c.Im / M_eff
        self.results.eir[ti]      = eir
        self.results.lambda_h[ti] = lambda_h
        self.results.R0[ti]       = R0
        self.results.m[ti]        = m
        return

    def plot(self, **kwargs):
        """Plot prevalence, R0, mosquito density, and force of infection."""
        r = self.results
        t = r.timevec / self.pars.year_len  # convert to years
        kw = sc.mergedicts(dict(lw=1), kwargs)

        with sc.options.with_style('fancy'):
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
            fig.suptitle('Ross-Macdonald model with seasonality', fontsize=14)

            ax0.plot(t, r['prev_h'].values, label='Human prevalence (Ih/Nh)', **kw)
            ax0.plot(t, r['prev_m'].values, label='Mosquito prevalence (Im/M)', **kw)
            ax0.set_ylabel('Prevalence')
            ax0.legend()
            sc.boxoff(ax0)

            ax1.plot(t, r['R0'].values, label='R0(t)', **kw)
            ax1.axhline(1.0, linestyle='--', linewidth=1, color='gray')
            ax1.set_ylabel('R0')
            ax1.legend()
            sc.boxoff(ax1)

            ax2.plot(t, r['m'].values, label='m(t) = mosquitoes per human', **kw)
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

        return ss.return_fig(fig)


if __name__ == '__main__':
    mod = RossMac_SS()
    sim = ss.Sim(modules=mod, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
    sim.run()
    mod.plot()
    plt.show()
