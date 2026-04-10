"""
Classic Ross–Macdonald malaria transmission dynamics (two coupled ODEs).

Implements the continuous-time formulation discussed in Smith et al., 2012
(PLoS Pathogens, Box 4): X = infected humans, Z = infectious mosquitoes,
with constant population sizes H and M.

dX/dt = a b (Z/M)(H - X) - r X
dZ/dt = a c (X/H)(M e^{-gn} - Z) - g Z

where a = biting rate (per mosquito per day), b = P(human infection | infectious
bite), c = P(mosquito infection | bite on infectious human), r = human recovery
rate, g = mosquito mortality rate, n = extrinsic incubation period (EIP).
The term M·e^{-gn} caps infectious mosquitoes at the fraction surviving the EIP
(Smith & McKenzie 1993 form). Note: m = M/H is not explicit here; the parameters
are calibrated for a given transmission intensity setting.
"""

import numpy as np
import sciris as sc

default_pars = sc.objdict(
    # Populations (constant)
    H = 1000.0,
    M = 10_000.0,
    # a: bites per mosquito per unit time; b, c: transmission probabilities
    a = 0.24,  # baseline biting rate (a0)
    b = 0.5,
    c = 0.5,
    # r: human recovery rate; g: mosquito mortality rate; n: EIP (days)
    r = 1 / 200.0,
    g = 1 / 10.0,
    n = 10.0,
    # Seasonality applied to biting rate: a(t) = a0 * (1 + amp*sin(2*pi*(t-phase)/period))
    seasonal = True,
    seas_amp = 0.3,
    seas_period = 365.0,
    seas_phase = 0.0,
    # Initial infected counts (small human seed, near disease-free vectors)
    X0 = 5.0,
    Z0 = 50.0,
)


class RossMacdonaldModel:
    """Two-compartment Ross–Macdonald model with Euler integration."""

    def __init__(self, **kwargs):
        self.pars = sc.mergedicts(default_pars, kwargs)
        self.t = None
        self.X = None
        self.Z = None
        return

    def R0(self):
        """Basic reproduction number (spectral radius of the next-generation matrix).

        R0 = sqrt(a² b c exp(-g n) / (r g))

        This is the geometric mean of the two one-way reproduction numbers
        (mosquito→human and human→mosquito), incorporating EIP survival.
        """
        p = self.pars
        return np.sqrt((p.a ** 2) * p.b * p.c * np.exp(-p.g * p.n) / (p.r * p.g))

    def _a_t(self, t):
        """Seasonal biting rate a(t)."""
        p = self.pars
        a0 = p.a
        if not p.seasonal:
            return a0
        return a0 * (1.0 + p.seas_amp * np.sin(2.0 * np.pi * (t - p.seas_phase) / p.seas_period))

    def _deriv(self, t, X, Z):
        p = self.pars
        a = self._a_t(t)
        x = X / p.H
        z = Z / p.M
        # Human equation: infectious bites drive new infections
        dX = a * p.b * z * (p.H - X) - p.r * X
        # Mosquito equation (Smith-McKenzie): cap infectious pool at M·exp(-g·n)
        dZ = a * p.c * x * (p.M * np.exp(-p.g * p.n) - Z) - p.g * Z
        return dX, dZ

    def run(self, start=0.0, stop=3*365.0, dt=1.0, X0=None, Z0=None):
        """Forward-Euler integration of the coupled ODEs."""
        p = self.pars
        if X0 is None:
            X0 = p.X0
        if Z0 is None:
            Z0 = p.Z0

        t = np.arange(start, stop + 0.5 * dt, dt)
        n = len(t)
        X = np.empty(n)
        Z = np.empty(n)
        X[0] = X0
        Z[0] = Z0

        for i in range(n - 1):
            dX, dZ = self._deriv(t[i], X[i], Z[i])
            X[i + 1] = X[i] + dt * dX
            Z[i + 1] = Z[i] + dt * dZ
            # Keep states physically valid (numerical guardrails)
            X[i + 1] = np.clip(X[i + 1], 0.0, p.H)
            Z[i + 1] = np.clip(Z[i + 1], 0.0, p.M)

        self.t = t
        self.X = X
        self.Z = Z

    def plot(self):
        """Plot human and mosquito infectious fractions through time."""
        import matplotlib.pyplot as plt

        if self.t is None or self.X is None or self.Z is None:
            raise RuntimeError("Call run() before plot().")

        p = self.pars
        t = self.t
        x = self.X / p.H
        z = self.Z / p.M

        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
        ax.plot(t, x, color="C0", label="Humans infectious fraction $x=X/H$")
        ax.plot(t, z, color="C1", label="Mosquitoes infectious fraction $z=Z/M$")
        ax.set_xlabel("Time")
        ax.set_ylabel("Infectious fraction")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        title = f"Ross–Macdonald ($R_0 \\approx {self.R0():.2f}$, $H={p.H:g}$, $M={p.M:g}$)"
        if p.seasonal:
            title += f" | seasonality: amp={p.seas_amp:g}, period={p.seas_period:g}"
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    rmm = RossMacdonaldModel()
    rmm.run()
    rmm.plot()
