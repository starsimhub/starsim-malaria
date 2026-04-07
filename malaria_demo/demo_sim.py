"""
Malaria transmission ODE model with seasonal mosquito density forcing.

Translates mahmud_model/demo_sim_endemic.R to Python using scipy, numpy, and matplotlib.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================
# ODE with time-varying seasonal mosquito density
# ============================================================
def malaria_ode_seasonal(t, x, params):
    n = params["numpatch"]
    X = x[:n]
    # C = x[n:]  # cumulative incidence state variables

    # Seasonal mosquito density: m(t) = m_base * seasonal_factor(t)
    m = params["m_base"] * params["seasonal_fn"](t)

    pij = params["pij"]
    H = params["H"]
    a = params["a"]
    b = params["b"]
    c = params["c"]
    mu = params["mu"]
    tau = params["tau"]
    r = params["r"]

    # Weighted prevalence at each destination (column)
    # k_j = sum_i(pij[i,j] * X[i] * H[i]) / sum_i(pij[i,j] * H[i])
    k = (X * H) @ pij / (H @ pij)

    # Force of infection: dC
    Z_numer = a**2 * b * c * np.exp(-mu * tau) * k
    Z_denom = a * c * k + mu
    dC = (m * Z_numer / Z_denom) @ pij.T * (1 - X)

    dX = dC - r * X

    return np.concatenate([dX, dC])


# ============================================================
# Analytical mosquito density from equilibrium prevalence
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


def main():
    # ============================================================
    # Parameters
    # ============================================================
    a = 0.3
    b = 0.1
    c = 0.214
    r = 1 / 150
    mu = 1 / 10
    tau = 10

    # Synthetic 3-patch data
    numpatch = 3
    hvector = np.array([5000.0, 10000.0, 8000.0])
    ivector = np.array([0.001, 0.003, 0.002])

    # Mobility matrix (rows=residence, cols=destination; rows sum to 1)
    pij = np.array([
        [0.85, 0.10, 0.05],
        [0.08, 0.80, 0.12],
        [0.06, 0.09, 0.85],
    ])

    # Solve for baseline mosquito density
    m_base = analytical_mosquito_density(ivector, hvector, pij, a, b, c, mu, r, tau)
    m_base = np.clip(m_base, 1e-10, None)
    print(f"Baseline mosquito density (m): {np.round(m_base, 4)}")

    # Choose seasonal forcing
    seasonal_fn = make_seasonal_sinusoidal(amplitude=0.8, peak_day=180)
    # Alternative: seasonal_fn = make_seasonal_normal(peak_day=180, sd_days=60, baseline=0.1)

    # ============================================================
    # Run simulation
    # ============================================================
    n_years = 5
    t_span = (1, 365 * n_years)
    t_eval = np.arange(1, 365 * n_years + 1)

    params = dict(
        numpatch=numpatch, c=c, b=b, a=a, mu=mu,
        tau=tau, r=r, pij=pij, H=hvector,
        m_base=m_base, seasonal_fn=seasonal_fn,
    )

    # Start from equilibrium prevalence
    x0 = np.concatenate([ivector / r, np.zeros(numpatch)])

    print(f"Running seasonal simulation for {n_years} years...")
    sol = solve_ivp(
        malaria_ode_seasonal, t_span, x0,
        args=(params,), t_eval=t_eval, method="LSODA",
    )
    print("Simulation complete.")

    times = sol.t
    states = sol.y.T  # shape (n_times, 2*numpatch)

    # ============================================================
    # Derived quantities
    # ============================================================
    patch_names = [f"Patch {i+1}" for i in range(numpatch)]

    # Prevalence
    prev_mat = states[:, :numpatch]

    # Daily incidence (diff of cumulative)
    cum_mat = states[:, numpatch:]
    daily_inc = np.diff(cum_mat, axis=0, prepend=cum_mat[0:1, :])

    # Seasonal m(t) for each patch over time
    m_seasonal = np.array([m_base * seasonal_fn(t) for t in times])

    # EIR: entomological inoculation rate
    eir_mat = np.zeros((len(times), numpatch))
    for ti in range(len(times)):
        X = prev_mat[ti]
        k = (X * hvector) @ pij / (hvector @ pij)
        Z = a * c * k * np.exp(-mu * tau) / (a * c * k + mu)
        eir_mat[ti] = m_seasonal[ti] * a * (Z @ pij.T)

    # ============================================================
    # Interactive plots
    # ============================================================
    plt.ion()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Malaria Transmission Model with Seasonal Forcing", fontsize=14, fontweight="bold")

    # --- 1. Seasonal forcing: mosquito density multiplier over one year ---
    ax = axes[0, 0]
    one_year = np.arange(1, 366)
    forcing_vals = [seasonal_fn(d) for d in one_year]
    ax.plot(one_year, forcing_vals, color="darkgreen", linewidth=1)
    ax.axhline(1, linestyle="--", alpha=0.5, color="gray")
    ax.set_title("Seasonal forcing: mosquito density multiplier")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Multiplier on baseline m")

    # --- 2. Mosquito density m(t) by patch ---
    ax = axes[0, 1]
    for i, name in enumerate(patch_names):
        ax.plot(times, m_seasonal[:, i], linewidth=0.6, label=name)
    ax.set_title("Mosquito density m(t) by patch (seasonal)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mosquitoes per person (m)")
    ax.legend()

    # --- 3. Prevalence dynamics ---
    ax = axes[1, 0]
    for i, name in enumerate(patch_names):
        ax.plot(times, prev_mat[:, i], linewidth=0.7, label=name)
    ax.set_title("Endemic prevalence dynamics with seasonal forcing")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Prevalence (proportion infected)")
    ax.legend()

    # --- 4. Daily incidence by patch ---
    ax = axes[1, 1]
    for i, name in enumerate(patch_names):
        ax.plot(times, daily_inc[:, i], linewidth=0.5, alpha=0.8, label=name)
    ax.set_title("Daily incidence (new infections per person per day)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Daily incidence")
    ax.legend()

    # --- 5. Prevalence zoomed to last year ---
    last_year_mask = times > 365 * (n_years - 1)
    ax = axes[2, 0]
    for i, name in enumerate(patch_names):
        ax.plot(times[last_year_mask], prev_mat[last_year_mask, i], linewidth=0.8, label=name)
    ax.set_title(f"Prevalence: year {n_years} (stabilized seasonal cycle)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Prevalence")
    ax.legend()

    # --- 6. Daily incidence zoomed to last year ---
    ax = axes[2, 1]
    for i, name in enumerate(patch_names):
        ax.plot(times[last_year_mask], daily_inc[last_year_mask, i], linewidth=0.8, label=name)
    ax.set_title(f"Daily incidence: year {n_years} (stabilized seasonal cycle)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Daily incidence")
    ax.legend()

    fig.tight_layout()
    plt.show(block=True)

    print("Done.")
    return


if __name__ == '__main__':
    main()