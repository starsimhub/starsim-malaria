"""
Adaptation of ChatGPT version of Ross-Macdonald model
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

default_pars = sc.objdict(
    tmax     = 365*5, # simulation length in days
    dt       = 1.0,  # timestep in days
    Nh       = 1000,  # number of humans
    m0       = 5,     # baseline mosquito:human ratio
    a        = 0.24,  # bites per mosquito per day
    b        = 0.15,  # mosquito-to-human transmission probability per bite
    c        = 0.15,  # human-to-mosquito transmission probability per bite
    r        = 1/120, # human recovery rate per day
    mu_m     = 0.10,  # mosquito death rate per day
    sigma_m  = 1/12,  # mosquito incubation completion rate per day
    sn_amp   = 0.65,  # seasonality amplitude
    sn_phase = 120,   # peak timing shift in days
    sn_peak  = 1.5,   # sharpness of the seasonal peak
    sn_relax = 0.15,  # speed mosquitoes track the seasonal target
    Ih0      = 10,    # initial infectious humans
    Im0      = 100,   # initial infectious mosquitoes
    year_len = 365,   # length of the year in days
)

eps = 1e-8 # Avoid floating point errors


class RossMacdonaldModel(sc.prettyobj):
    """
    Seasonal Ross-Macdonald malaria model.

    State variables
    ---------------
    Sh : susceptible humans
    Ih : infectious humans
    Sm : susceptible mosquitoes
    Em : exposed mosquitoes
    Im : infectious mosquitoes

    Notes
    -----
    - Human population Nh is held constant.
    - Mosquito population M(t) varies seasonally toward a time-varying carrying capacity.
    - Seasonality is applied to mosquito abundance by modulating mosquito birth/input.
    - R0(t) is computed using the instantaneous mosquito:human ratio m(t) = M(t)/Nh.
    """

    def __init__(self, **kwargs):
        self.pars = sc.mergedicts(default_pars, kwargs)
        self.results = None
        return

    def seasonality(self, t):
        """
        Seasonal multiplier for mosquito abundance.
        Mean is approximately 1, bounded away from negative values.
        """
        p = self.pars
        x = 2 * np.pi * (t - p.sn_phase) / p.year_len  # map day-of-year to radians with a phase shift
        s = 1 + p.sn_amp * np.sin(x)  # sinusoidal multiplier around 1 for seasonal mosquito abundance

        # Optional sharpening of the seasonal pattern while keeping positivity
        if p.sn_peak != 1.0:
            s = np.maximum(s, eps)  # clamp to keep values positive before exponentiation
            s = s ** p.sn_peak  # sharpen/flatten the seasonal curve via a power transform

            # Renormalize approximately to mean 1 over one year
            tt = np.linspace(0, p.year_len, 2000)
            xx = 2 * np.pi * (tt - p.sn_phase) / p.year_len
            ss = (1 + p.sn_amp * np.sin(xx))  # reference seasonal curve over one year for normalization
            ss = np.maximum(ss, eps) ** p.sn_peak  # apply the same power transform to the reference curve
            s /= ss.mean()  # rescale so the (approximate) annual mean multiplier stays near 1

        return s

    def target_mos_pop(self, t):
        p = self.pars
        return p.Nh * p.m0 * self.seasonality(t)  # seasonal target mosquito population M*(t) from baseline m0 and seasonality

    def compute_R0(self, M):
        """
        Instantaneous Ross-Macdonald R0 using current mosquito density.

        R0 = (m * a^2 * b * c * sigma_m) / (r * mu_m * (sigma_m + mu_m))

        This is equivalent to the classic form with survival/incubation written
        in rate form rather than p^n / -ln(p).
        """
        p = self.pars
        m = M / p.Nh
        return (m * p.a**2 * p.b * p.c * p.sigma_m) / (
            p.r * p.mu_m * (p.sigma_m + p.mu_m)  # recovery × death × (incubation+death) terms in rate-form R0
        )

    def human_step(self, Sh, Ih, lambda_h):
        """
        Human compartment derivatives (SIS).
        """
        p = self.pars
        dSh = -lambda_h * Sh + p.r * Ih  # infections move Sh→Ih; recoveries move Ih→Sh (SIS)
        dIh = lambda_h * Sh - p.r * Ih  # same flows with opposite sign for Ih
        return dSh, dIh

    def mosquito_step(self, t, Sm, Em, Im, lambda_m):
        """
        Mosquito compartment derivatives (SEI).
        """
        p = self.pars
        M = Sm + Em + Im
        M_eff = max(M, eps)

        M_target = self.target_mos_pop(t)  # desired seasonal mosquito population M*(t)
        births = p.sn_relax * max(M_target - M, 0.0)  # add mosquitoes when below target at a relaxation rate
        extra_loss = p.sn_relax * max(M - M_target, 0.0)  # remove mosquitoes when above target at a relaxation rate

        dSm = births - lambda_m * Sm - p.mu_m * Sm - extra_loss * (Sm / M_eff)  # Sm gains births; loses infection/death; shares seasonal cull
        dEm = lambda_m * Sm - p.sigma_m * Em - p.mu_m * Em - extra_loss * (Em / M_eff)  # Em gains infections; loses progression/death; shares cull
        dIm = p.sigma_m * Em - p.mu_m * Im - extra_loss * (Im / M_eff)  # Im gains progressed Em; loses death; shares seasonal cull
        return dSm, dEm, dIm

    def step(self, t, y):
        """
        ODE right-hand side.
        """
        p = self.pars

        Sh, Ih, Sm, Em, Im = y

        Nh = Sh + Ih  # current human population (kept constant later, but computed from state)
        M = Sm + Em + Im  # current adult mosquito population (all mosquito compartments)

        # Keep denominators safe
        Nh_eff = max(Nh, eps)  # avoid divide-by-zero in Ih/Nh
        M_eff = max(M, eps)  # avoid divide-by-zero in Im/M and compartment shares

        # Forces of infection
        lambda_h = p.a * p.b * (Im / M_eff) * (M / p.Nh)  # human FOI from bites × trans prob × frac infectious mosq × mosq density
        lambda_m = p.a * p.c * (Ih / Nh_eff)  # mosquito FOI from bites × trans prob × human infectious prevalence

        dSh, dIh = self.human_step(Sh, Ih, lambda_h)
        dSm, dEm, dIm = self.mosquito_step(t, Sm, Em, Im, lambda_m)

        return np.array([dSh, dIh, dSm, dEm, dIm], dtype=float)

    def run(self):
        """
        Simulate the model with forward Euler.
        """
        p = self.pars
        t = np.arange(0, p.tmax + p.dt, p.dt)  # simulation time grid (days)
        n = len(t)

        M0 = p.m0 * p.Nh  # initialize mosquitoes from baseline mosquito:human ratio

        Ih0 = min(p.Ih0, p.Nh)  # cap initial infections at population size
        Sh0 = p.Nh - Ih0  # remaining humans start susceptible

        Im0 = min(p.Im0, M0)  # cap initial infectious mosquitoes at initial mosquito population
        Em0 = 0.0
        Sm0 = M0 - Em0 - Im0  # remaining mosquitoes start susceptible

        y = np.zeros((n, 5), dtype=float)
        y[0] = [Sh0, Ih0, Sm0, Em0, Im0]

        for i in range(n - 1):
            dy = self.step(t[i], y[i])
            y[i + 1] = y[i] + p.dt * dy  # forward Euler step: y(t+dt) = y(t) + dt·dy/dt
            if np.any(y[i + 1]) < 0:
                print(f'Warning: negative compartment size detected: {y[i + 1]} at time {t[i]}')
            y[i + 1] = np.maximum(y[i + 1], 0.0)  # clamp to keep compartment sizes non-negative

        Sh = y[:, 0]
        Ih = y[:, 1]
        Sm = y[:, 2]
        Em = y[:, 3]
        Im = y[:, 4]

        H = Sh + Ih  # total humans (should be exactly Nh each timestep)
        M = Sm + Em + Im  # total mosquitoes (varies seasonally)

        prev_h = Ih / np.maximum(H, eps)  # human prevalence Ih/Nh (fraction infectious)
        prev_m = Im / np.maximum(M, eps)  # mosquito prevalence Im/M (fraction infectious)
        eir = p.a * Im / p.Nh  # infectious bites per human per day (a·Im spread across Nh humans)
        lambda_h = p.a * p.b * (Im / np.maximum(M, eps)) * (M / p.Nh)  # human FOI using the same mass-action form as in deriv()
        R0 = self.compute_R0(M)
        season = self.seasonality(t)

        self.results = sc.objdict(
            t        = t,
            Sh       = Sh,
            Ih       = Ih,
            Sm       = Sm,
            Em       = Em,
            Im       = Im,
            H        = H,
            M        = M,
            m        = M / p.Nh,  # mosquito:human ratio time series m(t)
            prev_h   = prev_h,
            prev_m   = prev_m,
            eir      = eir,
            lambda_h = lambda_h,
            R0       = R0,
            season   = season,
            M_target = self.target_mos_pop(t),
        )
        return self.results

    def plot(self):
        if self.results is None:
            raise RuntimeError("Model has not been run yet. Call run() first.")

        r = self.results
        t = r.t / self.pars.year_len  # convert to years

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

        # Panel 1: prevalence
        ax0.plot(t, r.prev_h, label="Human prevalence (Ih/Nh)")
        ax0.plot(t, r.prev_m, label="Mosquito prevalence (Im/M)")
        ax0.set_ylabel("Prevalence")
        ax0.set_title("Ross-Macdonald model with seasonality")
        ax0.legend()
        ax0.grid(True, alpha=0.3)

        # Panel 2: R0
        ax1.plot(t, r.R0, label="R0(t)")
        ax1.axhline(1.0, linestyle="--", linewidth=1)
        ax1.set_ylabel("R0")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 3: mosquito density
        ax2.plot(t, r.m, label="Mosquito:human ratio m(t)")
        ax2.plot(t, r.M_target / self.pars.Nh, label="Seasonal target m*(t)", alpha=0.8)
        ax2.set_ylabel("Mosquitoes per human")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 4: force of infection / EIR-like quantity
        ax3.plot(t, r.lambda_h, label="Human FOI λh")
        ax3.plot(t, r.eir, label="Infectious bites per human per day", alpha=0.8)
        ax3.set_ylabel("Rate per day")
        ax3.set_xlabel("Time (years)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig


if __name__ == "__main__":
    model = RossMacdonaldModel()
    model.run()
    model.plot()