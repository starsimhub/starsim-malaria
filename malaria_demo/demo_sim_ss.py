"""
Malaria transmission model with seasonal mosquito density forcing (Starsim version).

Converts the scipy ODE model (demo_sim.py) into a Starsim Module with Euler integration,
following the pattern of TB_SS in tbsim/compartmental/lshtm_ode.py.

Example
-------
::

    import starsim as ss
    from demo_sim_ss import Malaria_SS

    mal = Malaria_SS()
    sim = ss.Sim(modules=mal, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
    sim.run()
    mal.plot()
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Helper: analytical mosquito density from equilibrium prevalence
# ============================================================
def analytical_mosquito_density(ivector, hvector, pij, a, b, c, mu, r, tau):
    xvector = ivector / r
    gvector = xvector * r / (1 - xvector)

    n = len(xvector)
    fvector = np.zeros(n)
    for i in range(n):
        k_i = np.sum(pij[:, i] * xvector * hvector) / np.sum(pij[:, i] * hvector)
        fvector[i] = b * c * k_i / (a * c * k_i / mu + 1)

    fmatrix = np.diag(fvector)
    cvector = np.linalg.solve(pij @ fmatrix, gvector)
    m = cvector * mu / a**2 / np.exp(-mu * tau)
    return m


# ============================================================
# Seasonal forcing functions
# ============================================================
def make_seasonal_sinusoidal(amplitude=0.8, peak_day=180):
    def seasonal_fn(t):
        day_of_year = t % 365
        return max(0.0, 1 + amplitude * np.sin(2 * np.pi * (day_of_year - peak_day + 365 / 4) / 365))
    return seasonal_fn


def make_seasonal_normal(peak_day=180, sd_days=60, baseline=0.1):
    peak_val = norm.pdf(peak_day, loc=peak_day, scale=sd_days)
    def seasonal_fn(t):
        day_of_year = t % 365
        return baseline + (1 - baseline) * norm.pdf(day_of_year, loc=peak_day, scale=sd_days) / peak_val
    return seasonal_fn


# ============================================================
# Default parameters
# ============================================================
default_pars = sc.objdict(
    a   = 0.3,      # Biting rate
    b   = 0.1,      # Mosquito-to-human transmission probability
    c   = 0.214,    # Human-to-mosquito transmission probability
    r   = 1 / 150,  # Recovery rate (per day)
    mu  = 1 / 10,   # Mosquito death rate (per day)
    tau = 10,        # Extrinsic incubation period (days)
)

# Default synthetic 3-patch data
default_hvector = np.array([5000.0, 10000.0, 8000.0])
default_ivector = np.array([0.001, 0.003, 0.002])
default_pij = np.array([
    [0.85, 0.10, 0.05],
    [0.08, 0.80, 0.12],
    [0.06, 0.09, 0.85],
])


class Malaria_SS(ss.Module):
    """
    Compartmental Starsim implementation of the malaria transmission model (Euler integration).

    Multi-patch Ross-Macdonald model with seasonal mosquito density forcing and
    human mobility between patches.

    Because this is a self-contained module, it does not need a network or People.

    See default_pars for parameter definitions.

    Example
    -------
    ::

        import starsim as ss
        from demo_sim_ss import Malaria_SS

        mal = Malaria_SS()
        sim = ss.Sim(modules=mal, start=0, stop=365*5, dt=1, n_agents=1)
        sim.run()
        sim.modules.malaria_ss.plot()
    """

    def __init__(self, hvector=None, ivector=None, pij=None, seasonal_fn=None, **kwargs):
        super().__init__()
        self.define_pars(**default_pars)
        self.update_pars(**kwargs)

        # Store patch data
        self.hvector = hvector if hvector is not None else default_hvector.copy()
        self.ivector = ivector if ivector is not None else default_ivector.copy()
        self.pij = pij if pij is not None else default_pij.copy()
        self.seasonal_fn = seasonal_fn if seasonal_fn is not None else make_seasonal_sinusoidal()
        self.numpatch = len(self.hvector)

        # Compartment labels
        self.c_labels = sc.objdict()
        for i in range(self.numpatch):
            self.c_labels[f'X_{i}'] = f'Prevalence (Patch {i+1})'
            self.c_labels[f'C_{i}'] = f'Cumulative incidence (Patch {i+1})'

        # Compartments (scalars, not per-agent states)
        self.c = sc.objdict({key: 0.0 for key in self.c_labels})

        # Will be computed in init_post
        self.m_base = None
        return

    def init_post(self):
        """Compute baseline mosquito density and set initial conditions."""
        super().init_post()
        p = self.pars
        n = self.numpatch

        # Analytical mosquito density from equilibrium
        self.m_base = analytical_mosquito_density(
            self.ivector, self.hvector, self.pij,
            p.a, p.b, p.c, p.mu, p.r, p.tau,
        )
        self.m_base = np.clip(self.m_base, 1e-10, None)

        # Initial prevalence = incidence / recovery rate
        x0 = self.ivector / p.r
        for i in range(n):
            self.c[f'X_{i}'] = x0[i]
            self.c[f'C_{i}'] = 0.0
        return

    def step(self):
        """Euler integration of the multi-patch malaria ODE system."""
        p = self.pars
        c = self.c
        n = self.numpatch
        t = self.now
        dt = float(self.dt)

        # Current state vectors
        X = np.array([c[f'X_{i}'] for i in range(n)])

        # Seasonal mosquito density
        m = self.m_base * self.seasonal_fn(t)

        # Weighted prevalence at each destination
        # k_j = sum_i(pij[i,j] * X[i] * H[i]) / sum_i(pij[i,j] * H[i])
        H = self.hvector
        pij = self.pij
        k = (X * H) @ pij / (H @ pij)

        # Force of infection
        Z_numer = p.a**2 * p.b * p.c * np.exp(-p.mu * p.tau) * k
        Z_denom = p.a * p.c * k + p.mu
        dC = (m * Z_numer / Z_denom) @ pij.T * (1 - X)

        dX = dC - p.r * X

        # Euler update
        for i in range(n):
            c[f'X_{i}'] += dX[i] * dt
            c[f'C_{i}'] += dC[i] * dt
        return

    def init_results(self):
        """Initialize results for all compartments and derived quantities."""
        super().init_results()
        n = self.numpatch
        results = []

        # Compartment results
        for key, label in self.c_labels.items():
            results.append(ss.Result(key, label=label))

        # Derived quantities per patch
        for i in range(n):
            results.append(ss.Result(f'daily_inc_{i}', label=f'Daily incidence (Patch {i+1})'))
            results.append(ss.Result(f'm_seasonal_{i}', label=f'Mosquito density (Patch {i+1})'))
            results.append(ss.Result(f'eir_{i}', label=f'EIR (Patch {i+1})'))

        self.define_results(*results)
        return

    def update_results(self):
        """Store current state and compute derived quantities."""
        super().update_results()
        ti = self.ti
        c = self.c
        p = self.pars
        n = self.numpatch
        t = self.now

        # Store compartment values
        for key in c:
            self.results[key][ti] = c[key]

        # Current state
        X = np.array([c[f'X_{i}'] for i in range(n)])
        m_seasonal = self.m_base * self.seasonal_fn(t)

        # Daily incidence (dC values)
        H = self.hvector
        pij = self.pij
        k = (X * H) @ pij / (H @ pij)
        Z_numer = p.a**2 * p.b * p.c * np.exp(-p.mu * p.tau) * k
        Z_denom = p.a * p.c * k + p.mu
        dC = (m_seasonal * Z_numer / Z_denom) @ pij.T * (1 - X)

        # EIR
        Z = p.a * p.c * k * np.exp(-p.mu * p.tau) / (p.a * p.c * k + p.mu)
        eir = m_seasonal * p.a * (Z @ pij.T)

        for i in range(n):
            self.results[f'daily_inc_{i}'][ti] = dC[i]
            self.results[f'm_seasonal_{i}'][ti] = m_seasonal[i]
            self.results[f'eir_{i}'][ti] = eir[i]
        return

    def plot(self, **kwargs):
        """Plot prevalence, daily incidence, mosquito density, and EIR."""
        n = self.numpatch
        patch_names = [f'Patch {i+1}' for i in range(n)]
        results = self.results
        kw = sc.mergedicts(dict(lw=1, alpha=0.8), kwargs)

        with sc.options.with_style('fancy'):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Malaria Transmission Model with Seasonal Forcing', fontsize=14, fontweight='bold')

            # Prevalence
            ax = axes[0, 0]
            for i, name in enumerate(patch_names):
                ax.plot(results.timevec, results[f'X_{i}'].values, label=name, **kw)
            ax.set_title('Prevalence')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Proportion infected')
            ax.legend()
            sc.boxoff(ax)

            # Daily incidence
            ax = axes[0, 1]
            for i, name in enumerate(patch_names):
                ax.plot(results.timevec, results[f'daily_inc_{i}'].values, label=name, **kw)
            ax.set_title('Daily incidence')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('New infections per person per day')
            ax.legend()
            sc.boxoff(ax)

            # Mosquito density
            ax = axes[1, 0]
            for i, name in enumerate(patch_names):
                ax.plot(results.timevec, results[f'm_seasonal_{i}'].values, label=name, **kw)
            ax.set_title('Seasonal mosquito density m(t)')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Mosquitoes per person')
            ax.legend()
            sc.boxoff(ax)

            # EIR
            ax = axes[1, 1]
            for i, name in enumerate(patch_names):
                ax.plot(results.timevec, results[f'eir_{i}'].values, label=name, **kw)
            ax.set_title('Entomological inoculation rate (EIR)')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('EIR')
            ax.legend()
            sc.boxoff(ax)

            sc.figlayout()

        return ss.return_fig(fig)


# ============================================================
# Run if executed directly
# ============================================================
if __name__ == '__main__':
    mal = Malaria_SS()
    sim = ss.Sim(modules=mal, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
    sim.run()
    mal.plot()
    plt.show()
