"""
ChatGPT version of Ross-Macdonald model
"""

import numpy as np
import matplotlib.pyplot as plt

pars = dict(
    tmax=365*5,            # simulation length in days
    dt=0.25,               # timestep in days
    Nh=1000,               # number of humans
    m0=5,                  # baseline mosquito:human ratio
    a=0.24,                # bites per mosquito per day
    b=0.15,                # mosquito-to-human transmission probability per bite
    c=0.15,                # human-to-mosquito transmission probability per bite
    r=1/120,               # human recovery rate per day
    mu_m=0.10,             # mosquito death rate per day
    sigma_m=1/12,          # mosquito incubation completion rate per day
    seasonal_amp=0.65,     # seasonality amplitude
    seasonal_phase=120,    # peak timing shift in days
    seasonal_peakedness=1.5,  # sharpness of the seasonal peak
    mosquito_relaxation=0.15, # speed mosquitoes track the seasonal target
    Ih0=10,                # initial infectious humans
    Im0=100,               # initial infectious mosquitoes
)


class RossMacdonaldModel:
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

    def __init__(self, pars=None):
        self.pars = self.default_pars()
        if pars:
            self.pars.update(pars)

        self.validate_pars()
        self.results = None

    @staticmethod
    def default_pars():
        return dict(
            # Time
            tmax=365 * 3,       # days
            dt=0.25,            # days

            # Population sizes
            Nh=1000.0,          # human population
            m0=10.0,            # baseline mosquitoes per human at t=0

            # Transmission
            a=0.24,             # bites per mosquito per day
            b=0.30,             # mosquito -> human transmission probability per bite
            c=0.30,             # human -> mosquito transmission probability per bite

            # Human natural history
            r=1 / 120,          # human recovery rate per day

            # Mosquito natural history
            mu_m=0.10,          # mosquito death rate per day
            sigma_m=1 / 12,     # exposed -> infectious mosquito progression rate per day

            # Seasonal mosquito demography
            seasonal_amp=0.50,  # amplitude in [0, 1)
            seasonal_phase=0.0, # phase shift in days
            seasonal_peakedness=1.0,  # >1 makes peak sharper, 1 is sinusoidal
            mosquito_relaxation=0.20, # rate at which M(t) moves toward seasonal target

            # Initial conditions
            Ih0=10.0,
            Im0=100.0,

            # Plotting / misc
            year_length=365.0,
        )

    def validate_pars(self):
        p = self.pars
        assert p["Nh"] > 0
        assert p["m0"] > 0
        assert p["dt"] > 0
        assert p["tmax"] > 0
        assert 0 <= p["b"] <= 1
        assert 0 <= p["c"] <= 1
        assert p["a"] >= 0
        assert p["r"] > 0
        assert p["mu_m"] > 0
        assert p["sigma_m"] > 0
        assert 0 <= p["seasonal_amp"] < 1
        assert p["mosquito_relaxation"] >= 0

    def seasonal_multiplier(self, t):
        """
        Seasonal multiplier for mosquito abundance.
        Mean is approximately 1, bounded away from negative values.
        """
        p = self.pars
        x = 2 * np.pi * (t - p["seasonal_phase"]) / p["year_length"]
        s = 1 + p["seasonal_amp"] * np.sin(x)

        # Optional sharpening of the seasonal pattern while keeping positivity
        if p["seasonal_peakedness"] != 1.0:
            s = np.maximum(s, 1e-8)
            s = s ** p["seasonal_peakedness"]

            # Renormalize approximately to mean 1 over one year
            tt = np.linspace(0, p["year_length"], 2000)
            xx = 2 * np.pi * (tt - p["seasonal_phase"]) / p["year_length"]
            ss = (1 + p["seasonal_amp"] * np.sin(xx))
            ss = np.maximum(ss, 1e-8) ** p["seasonal_peakedness"]
            s /= ss.mean()

        return s

    def target_mosquito_population(self, t):
        p = self.pars
        return p["Nh"] * p["m0"] * self.seasonal_multiplier(t)

    def compute_R0(self, M):
        """
        Instantaneous Ross-Macdonald R0 using current mosquito density.

        R0 = (m * a^2 * b * c * sigma_m) / (r * mu_m * (sigma_m + mu_m))

        This is equivalent to the classic form with survival/incubation written
        in rate form rather than p^n / -ln(p).
        """
        p = self.pars
        m = M / p["Nh"]
        return (m * p["a"]**2 * p["b"] * p["c"] * p["sigma_m"]) / (
            p["r"] * p["mu_m"] * (p["sigma_m"] + p["mu_m"])
        )

    def deriv(self, t, y):
        """
        ODE right-hand side.
        """
        p = self.pars

        Sh, Ih, Sm, Em, Im = y

        Nh = Sh + Ih
        M = Sm + Em + Im

        # Keep denominators safe
        Nh_eff = max(Nh, 1e-12)
        M_eff = max(M, 1e-12)

        # Forces of infection
        lambda_h = p["a"] * p["b"] * (Im / M_eff) * (M / p["Nh"])
        lambda_m = p["a"] * p["c"] * (Ih / Nh_eff)

        # Seasonal mosquito demography:
        # net input to susceptible mosquitoes pushes total M toward seasonal target
        M_target = self.target_mosquito_population(t)
        births = p["mosquito_relaxation"] * max(M_target - M, 0.0)
        extra_loss = p["mosquito_relaxation"] * max(M - M_target, 0.0)

        dSh = -lambda_h * Sh + p["r"] * Ih
        dIh = lambda_h * Sh - p["r"] * Ih

        dSm = births - lambda_m * Sm - p["mu_m"] * Sm - extra_loss * (Sm / M_eff)
        dEm = lambda_m * Sm - p["sigma_m"] * Em - p["mu_m"] * Em - extra_loss * (Em / M_eff)
        dIm = p["sigma_m"] * Em - p["mu_m"] * Im - extra_loss * (Im / M_eff)

        return np.array([dSh, dIh, dSm, dEm, dIm], dtype=float)

    def run(self):
        """
        Simulate the model with forward Euler.
        """
        p = self.pars
        t = np.arange(0, p["tmax"] + p["dt"], p["dt"])
        n = len(t)

        Nh = p["Nh"]
        M0 = p["m0"] * Nh

        Ih0 = min(p["Ih0"], Nh)
        Sh0 = Nh - Ih0

        Im0 = min(p["Im0"], M0)
        Em0 = 0.0
        Sm0 = M0 - Em0 - Im0

        y = np.zeros((n, 5), dtype=float)
        y[0] = [Sh0, Ih0, Sm0, Em0, Im0]

        for i in range(n - 1):
            dy = self.deriv(t[i], y[i])
            y[i + 1] = y[i] + p["dt"] * dy
            y[i + 1] = np.maximum(y[i + 1], 0.0)

            # Enforce constant human population exactly
            H = y[i + 1, 0] + y[i + 1, 1]
            if H > 0:
                y[i + 1, 0] *= Nh / H
                y[i + 1, 1] *= Nh / H

        Sh = y[:, 0]
        Ih = y[:, 1]
        Sm = y[:, 2]
        Em = y[:, 3]
        Im = y[:, 4]

        H = Sh + Ih
        M = Sm + Em + Im

        prev_h = Ih / np.maximum(H, 1e-12)
        prev_m = Im / np.maximum(M, 1e-12)
        eir = p["a"] * Im / p["Nh"]  # infectious bites per human per day
        lambda_h = p["a"] * p["b"] * (Im / np.maximum(M, 1e-12)) * (M / p["Nh"])
        R0 = self.compute_R0(M)
        season = np.array([self.seasonal_multiplier(tt) for tt in t])

        self.results = dict(
            t=t,
            Sh=Sh,
            Ih=Ih,
            Sm=Sm,
            Em=Em,
            Im=Im,
            H=H,
            M=M,
            m=M / p["Nh"],
            prev_h=prev_h,
            prev_m=prev_m,
            eir=eir,
            lambda_h=lambda_h,
            R0=R0,
            season=season,
            M_target=np.array([self.target_mosquito_population(tt) for tt in t]),
        )
        return self.results

    def plot(self):
        if self.results is None:
            raise RuntimeError("Model has not been run yet. Call run() first.")

        r = self.results
        t = r["t"] / self.pars["year_length"]  # convert to years

        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

        # Panel 1: prevalence
        axes[0].plot(t, r["prev_h"], label="Human prevalence (Ih/Nh)")
        axes[0].plot(t, r["prev_m"], label="Mosquito prevalence (Im/M)")
        axes[0].set_ylabel("Prevalence")
        axes[0].set_title("Ross-Macdonald model with seasonality")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel 2: R0
        axes[1].plot(t, r["R0"], label="R0(t)")
        axes[1].axhline(1.0, linestyle="--", linewidth=1)
        axes[1].set_ylabel("R0")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: mosquito density
        axes[2].plot(t, r["m"], label="Mosquito:human ratio m(t)")
        axes[2].plot(t, r["M_target"] / self.pars["Nh"], label="Seasonal target m*(t)", alpha=0.8)
        axes[2].set_ylabel("Mosquitoes per human")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Panel 4: force of infection / EIR-like quantity
        axes[3].plot(t, r["lambda_h"], label="Human FOI λh")
        axes[3].plot(t, r["eir"], label="Infectious bites per human per day", alpha=0.8)
        axes[3].set_ylabel("Rate per day")
        axes[3].set_xlabel("Time (years)")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = RossMacdonaldModel(pars)
    model.run()
    model.plot()