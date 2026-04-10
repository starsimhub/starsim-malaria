"""
Simplified Gervas et al. (2025) malaria compartmental model.

Single vector: An. funestus (non-resistant and resistant adults + larvae).
Interventions: ITN only (eqs 7–8 without IRS or biolarvicides).

Reference: Gervas et al. Malaria Journal (2025) 24:411

Usage:
    python malaria_hybrid/gervas_model_simple.py
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt


# -- State variable indices --------------------------------------------------
S_H, E_H, I_H, R_H = 0, 1, 2, 3           # Humans (SEIR)
L_B, S_B, E_B, I_B = 4, 5, 6, 7            # An. funestus larvae + non-resistant adults
S_R, E_R, I_R = 8, 9, 10                   # An. funestus resistant adults
N_STATES = 11


def default_parameters():
    """Baseline parameters (Table 2, funestus + ITN only) as an objdict."""
    return sc.objdict({
        # Simulation control
        'n_days': 1095.0,
        'dt': 1.0,
        'p': 10.0,

        # Human demographics
        'Pi_H': 0.046,
        'mu_H': 0.000041,
        'tau': 0.00009,

        # Human disease progression
        'epsilon_H': 0.071,
        'gamma': 0.0035,
        'omega': 0.00055,

        # Transmission probabilities
        'P0': 0.018,
        'P1': 0.28,

        # Mosquito recruitment / maturation (funestus)
        'Lambda_2': 0.85,
        'pi_B': 0.325,
        'mu_L': 0.2,

        # Resistance fraction at recruitment (funestus)
        'Phi': 0.2,

        # Mutation rates (non-resistant <-> resistant, funestus)
        'sigma_B': 0.01,
        'sigma_R': 0.05,

        # Mosquito exposed -> infectious (funestus)
        'epsilon_B': 0.1,
        'epsilon_R': 0.1,

        # Base mosquito mortality (funestus)
        'mu_B0': 0.14,
        'mu_R0': 0.02,

        # Biting rates: max (beta_j1) and min (beta_j0)
        'beta_B1': 0.5,
        'beta_B0': 0.07,
        'beta_R1': 0.7,
        'beta_R0': 0.075,

        # ITN
        'b1': 0.6,
        'T1': 1095.0,
        'xi_B_itn': 1.0,
        'xi_R_itn': 0.644,

        # Funestus resistance intensity (e_2 in [0,1])
        'e2': 0.5,

        'divide_forces_by_nh': True,
        'vector_input_scale': 2000.0,
        # Paper figures show mosquito densities adjusting over ~50–100 days, but
        # the raw Table-2 mosquito rates imply ~1-day larval turnover
        # (Λ₂+μ_L ≈ 1 day⁻¹). Apply a uniform timescale factor to *all* mosquito
        # demographic/progression rates so equilibrium stays the same while the
        # transient dynamics slow down.
        #
        # Interpretation: mosquito rates in the paper are effectively in “per
        # ~50 days” units (or the state variables are scaled); this factor
        # matches the published timescale without changing steady states.
        'mosquito_timescale': 0.02,
        # Small vector seed (see get_initial_conditions). Population grows from
        # larval inflow π_B·k and maturation; do not scale ICs with k or they sit
        # far above the ITN-heavy adult equilibrium and “crash.”
        'vector_seed_scale': 1.0,
    })


# -- Time-dependent functions ------------------------------------------------

def _sigmoid(x):
    """Sigmoid / logistic function: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def intervention_waning(t, T, p):
    """
    Waning factor for a single intervention (inner term of eq 7).

    At the start of each period (t mod T = 0) returns 1 (full efficacy).
    Decays toward 0 as t approaches end of period.
    """
    x = (t % T) / T
    scale = 1.0 + 1.0 / (2.0 ** p)
    term = 1.0 / (1.0 + (1.8 * x) ** p) - 1.0 / (2.0 ** p + 1.0)
    return scale * term


def mosquito_mortality(t, mu_j0, xi_itn, pars):
    """
    Time-dependent adult mosquito mortality (eq 7), ITN only.

    mu_j0 is the table baseline mortality parameter.
    """
    p = pars.p
    f_j = mu_j0
    f_j += pars.b1 * xi_itn * intervention_waning(t, pars.T1, p)
    return _sigmoid(f_j)


def biting_rate(t, beta_j1, beta_j0, xi_j_itn, pars):
    """
    Time-dependent biting rate (eq 8). Only ITN coverage (b1) affects biting rate.
    """
    x = (t % pars.T1) / pars.T1
    denom = 1.0 + x ** pars.p
    return beta_j1 + (beta_j0 - beta_j1) * pars.b1 * xi_j_itn / denom


# -- ODE system --------------------------------------------------------------

def derivatives(t, y, pars):
    """Compute all dY/dt for humans + An. funestus (ITN only)."""
    dy = np.zeros(N_STATES)

    s_H, e_H, i_H, r_H = y[S_H], y[E_H], y[I_H], y[R_H]
    l_B, s_B, e_B, i_B = y[L_B], y[S_B], y[E_B], y[I_B]
    s_Rf, e_Rf, i_Rf = y[S_R], y[E_R], y[I_R]

    re2 = 1.0 - pars.e2
    ms = float(pars.mosquito_timescale)

    bB = biting_rate(t, pars.beta_B1, pars.beta_B0, pars.xi_B_itn, pars)
    bR = biting_rate(t, pars.beta_R1, pars.beta_R0, pars.xi_R_itn * re2, pars)

    n_h = max(s_H + e_H + i_H + r_H, 1e-12)
    bite_mos_to_hum = bB * i_B + bR * i_Rf
    if pars.divide_forces_by_nh:
        lam_H = pars.P0 * bite_mos_to_hum / n_h
        lam_B = pars.P1 * bB * i_H / n_h
        lam_R = pars.P1 * bR * i_H / n_h
    else:
        lam_H = pars.P0 * bite_mos_to_hum
        lam_B = pars.P1 * bB * i_H
        lam_R = pars.P1 * bR * i_H

    mu_B = ms * mosquito_mortality(t, pars.mu_B0, pars.xi_B_itn, pars)
    mu_RR = ms * mosquito_mortality(t, pars.mu_R0, pars.xi_R_itn * re2, pars)

    # Mosquito rates (scaled to match paper timescale)
    pi_B = ms * pars.pi_B
    Lambda_2 = ms * pars.Lambda_2
    mu_L = ms * pars.mu_L
    sigma_B = ms * pars.sigma_B
    sigma_R = ms * pars.sigma_R
    epsilon_B = ms * pars.epsilon_B
    epsilon_R = ms * pars.epsilon_R

    dy[S_H] = pars.Pi_H - lam_H * s_H + pars.omega * r_H - pars.mu_H * s_H
    dy[E_H] = lam_H * s_H - (pars.epsilon_H + pars.mu_H) * e_H
    dy[I_H] = pars.epsilon_H * e_H - (pars.gamma + pars.mu_H + pars.tau) * i_H
    dy[R_H] = pars.gamma * i_H - (pars.omega + pars.mu_H) * r_H

    dy[L_B] = pi_B * pars.vector_input_scale - (Lambda_2 + mu_L) * l_B

    dy[S_B] = ((1 - pars.Phi) * Lambda_2 * l_B
               + sigma_R * s_Rf
               - (lam_B + sigma_B + mu_B) * s_B)

    dy[E_B] = (lam_B * s_B
               + sigma_R * e_Rf
               - (epsilon_B + sigma_B + mu_B) * e_B)

    dy[I_B] = (epsilon_B * e_B
               + sigma_R * i_Rf
               - (sigma_B + mu_B) * i_B)

    dy[S_R] = (pars.Phi * Lambda_2 * l_B
               + sigma_B * s_B
               - (lam_R + sigma_R + mu_RR) * s_Rf)

    dy[E_R] = (lam_R * s_Rf
               + sigma_B * e_B
               - (epsilon_R + sigma_R + mu_RR) * e_Rf)

    dy[I_R] = (epsilon_R * e_Rf
               + sigma_B * i_B
               - (sigma_R + mu_RR) * i_Rf)

    return dy


# -- Integration and initial conditions --------------------------------------

def get_initial_conditions(pars):
    """
    Human population starts near endemic levels; vectors start as a **small seed**.

    Larval inflow ``π_B · vector_input_scale`` fills the immature pool and, through
    ``Λ₂ L_B``, supplies adults, so totals **increase** from a low seed toward a
    carrying capacity set by mortality (including ITN) and transmission. With
    default ``k``, the larval pool approaches its equilibrium in on the order of
    a few days, so adult counts rise quickly at first, then level off — unlike
    scaling all vector compartments by ``k``, which put ~2000 adults at t=0 and
    produced an immediate drop toward equilibrium.
    """
    y0 = np.zeros(N_STATES)
    q = float(pars.vector_seed_scale)

    y0[S_H] = 800.0
    y0[E_H] = 50.0
    y0[I_H] = 120.0
    y0[R_H] = 30.0

    # Baseline seed (~few dozen adults + modest larvae); same relative split as
    # the full model’s shape, scaled only by vector_seed_scale.
    y0[L_B] = 30.0 * q
    y0[S_B] = 20.0 * q
    y0[E_B] = 5.0 * q
    y0[I_B] = 3.0 * q
    y0[S_R] = 8.0 * q
    y0[E_R] = 2.0 * q
    y0[I_R] = 1.0 * q

    return y0


def simulate(pars):
    """Forward Euler integration; returns ``(t, y)``."""
    n_steps = int(pars.n_days / pars.dt)
    t = np.arange(n_steps + 1) * pars.dt
    y = np.zeros((n_steps + 1, N_STATES))
    y[0] = get_initial_conditions(pars)

    for i in range(n_steps):
        dy = derivatives(t[i], y[i], pars)
        y[i + 1] = np.maximum(y[i] + pars.dt * dy, 0.0)

    return t, y


def run(pars=None):
    """
    Default-parameter run: integrate, print summary, and show/save summary plot.
    """
    if pars is None:
        pars = default_parameters()

    print(f'Running Gervas (simple) model: {pars.n_days:.0f} days, dt={pars.dt}')
    t, y = simulate(pars)

    nh = y[-1, S_H] + y[-1, E_H] + y[-1, I_H] + y[-1, R_H]
    n_fun = y[-1, S_B] + y[-1, E_B] + y[-1, I_B] + y[-1, S_R] + y[-1, E_R] + y[-1, I_R]
    prev = y[-1, I_H] / max(nh, 1)
    print(f'Final N_H={nh:.1f}  (S={y[-1,S_H]:.1f} E={y[-1,E_H]:.1f} I={y[-1,I_H]:.1f} R={y[-1,R_H]:.1f})')
    print(f'Final N_funestus={n_fun:.1f}')
    print(f'Malaria prevalence = {prev:.3f}')

    plot(t, y, pars)
    return t, y


# -- Plotting ----------------------------------------------------------------

def plot(t, y, pars=None):
    """Summary plot: humans, An. funestus, biting rates."""
    if pars is None:
        pars = default_parameters()

    step = max(1, int(1.0 / pars.dt))
    tp = t[::step]
    yp = y[::step]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(tp, yp[:, S_H], label='$S_H$ (susceptible)', color='steelblue')
    ax.plot(tp, yp[:, E_H], label='$E_H$ (exposed)', color='orange')
    ax.plot(tp, yp[:, I_H], label='$I_H$ (infectious)', color='crimson')
    ax.plot(tp, yp[:, R_H], label='$R_H$ (recovered)', color='seagreen')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('Human Compartments (SEIR)')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    n_fun = yp[:, S_B] + yp[:, E_B] + yp[:, I_B] + yp[:, S_R] + yp[:, E_R] + yp[:, I_R]
    ax.plot(tp, n_fun, label='Total adults $N_{fun}$', color='black', lw=2)
    ax.plot(tp, yp[:, I_B], label='Infectious non-res. $I_B$', color='crimson')
    ax.plot(tp, yp[:, I_R], label='Infectious resistant $I_R$', color='darkred', ls='--')
    ax.plot(tp, yp[:, L_B], label='Larvae $L_B$', color='gray', ls=':')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('An. funestus')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    prev = yp[:, I_H] / np.maximum(
        yp[:, S_H] + yp[:, E_H] + yp[:, I_H] + yp[:, R_H],
        1e-12,
    )
    ax.plot(tp, prev, color='crimson', lw=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Prevalence')
    ax.set_title('Malaria prevalence ($I_H / N_H$)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    t_plot = np.linspace(0, pars.n_days, 2000)
    re2 = 1.0 - pars.e2
    for beta1, beta0, xi, label, ls in [
        (pars.beta_B1, pars.beta_B0, pars.xi_B_itn, r'$\beta_B$ (sus.)', '-'),
        (pars.beta_R1, pars.beta_R0, pars.xi_R_itn * re2, r'$\beta_R$ (res.)', '--'),
    ]:
        br = np.array([biting_rate(ti, beta1, beta0, xi, pars) for ti in t_plot])
        ax.plot(t_plot, br, label=label, ls=ls)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Biting rate (day$^{-1}$)')
    ax.set_title('Mosquito Biting Rates (eq 8, ITN)')
    ax.legend(fontsize=9)

    fig.suptitle('Gervas et al. (2025) — simplified (funestus + ITN)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    sc.savefig(sc.thispath() / 'gervas_results_simple.png')
    plt.show()


if __name__ == '__main__':
    run()
