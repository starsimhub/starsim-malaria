"""
Gervas et al. (2025) Malaria compartmental model.

Implements the SEIR human / SEI mosquito model with two vector species
(An. arabiensis and An. funestus), each split into susceptible (non-resistant)
and resistant sub-populations, plus larval stages. Includes time-dependent
intervention effects (ITNs, IRS, biolarvicides) on mosquito mortality and
biting rates.

Reference: Gervas et al. Malaria Journal (2025) 24:411

Usage:
    python malaria_hybrid/gervas_model.py
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace


# -- State variable indices --------------------------------------------------
S_H, E_H, I_H, R_H = 0, 1, 2, 3           # Humans (SEIR)
L_A, S_A, E_A, I_A = 4, 5, 6, 7            # An. arabiensis larvae + non-resistant adults
S_r, E_r, I_r = 8, 9, 10                    # An. arabiensis resistant adults
L_B, S_B, E_B, I_B = 11, 12, 13, 14        # An. funestus larvae + non-resistant adults
S_R, E_R, I_R = 15, 16, 17                  # An. funestus resistant adults
N_STATES = 18


# -- Parameters --------------------------------------------------------------
@dataclass
class Parameters:
    """All model parameters from Table 2 with baseline values."""

    # Simulation control
    n_days: float = 1095.0   # 3 years (one full ITN cycle)
    dt: float = 0.05         # Euler timestep (days); smaller helps stiff λ/N_H dynamics
    p: float = 10.0          # Waning shape parameter (>1; inferred from Fig 2)

    # ---- Human demographics ----
    Pi_H: float = 0.046       # Recruitment rate (day^-1)
    mu_H: float = 0.000041    # Natural death rate (day^-1)
    tau: float = 0.00009      # Disease-induced death rate (day^-1)

    # ---- Human disease progression ----
    epsilon_H: float = 0.071  # Exposed -> infectious (day^-1)
    gamma: float = 0.0035     # Recovery rate (day^-1)
    omega: float = 0.00055    # Loss of immunity rate (day^-1)

    # ---- Transmission probabilities ----
    P0: float = 0.018         # Mosquito -> human
    P1: float = 0.28          # Human -> mosquito

    # ---- Mosquito recruitment / maturation ----
    Lambda_1: float = 0.79    # An. arabiensis maturation rate from larvae (day^-1)
    Lambda_2: float = 0.85    # An. funestus maturation rate from larvae (day^-1)
    pi_A: float = 0.325       # An. arabiensis egg-laying / hatching rate (day^-1)
    pi_B: float = 0.325       # An. funestus egg-laying / hatching rate (day^-1)
    mu_L: float = 0.2         # Immature mosquito natural death rate (day^-1)

    # ---- Resistance fractions at recruitment ----
    phi: float = 0.02         # Fraction An. arabiensis recruited resistant
    Phi: float = 0.2          # Fraction An. funestus recruited resistant

    # ---- Mutation rates (non-resistant <-> resistant) ----
    sigma_A: float = 0.02     # Non-resistant -> resistant An. arabiensis (day^-1)
    sigma_r: float = 0.04     # Resistant -> non-resistant An. arabiensis (day^-1)
    sigma_B: float = 0.01     # Non-resistant -> resistant An. funestus (day^-1)
    sigma_R: float = 0.05     # Resistant -> non-resistant An. funestus (day^-1)

    # ---- Mosquito exposed -> infectious rates ----
    epsilon_A: float = 0.1    # Non-resistant An. arabiensis (day^-1)
    epsilon_B: float = 0.1    # Non-resistant An. funestus (day^-1)
    epsilon_r: float = 0.1    # Resistant An. arabiensis (day^-1)
    epsilon_R: float = 0.1    # Resistant An. funestus (day^-1)

    # ---- Base mosquito mortality rates (Table 2 values = actual rates) ----
    mu_A0: float = 0.24       # Non-resistant An. arabiensis (day^-1)
    mu_r0: float = 0.025      # Resistant An. arabiensis (day^-1)
    mu_B0: float = 0.14       # Non-resistant An. funestus (day^-1)
    mu_R0: float = 0.02       # Resistant An. funestus (day^-1)

    # ---- Biting rates: max (beta_j1) and min (beta_j0) ----
    beta_A1: float = 0.5      # Max non-resistant An. arabiensis (day^-1)
    beta_A0: float = 0.07     # Min non-resistant An. arabiensis (day^-1)
    beta_r1: float = 0.6      # Max resistant An. arabiensis (day^-1)
    beta_r0: float = 0.075    # Min resistant An. arabiensis (day^-1)
    beta_B1: float = 0.5      # Max non-resistant An. funestus (day^-1)
    beta_B0: float = 0.07     # Min non-resistant An. funestus (day^-1)
    beta_R1: float = 0.7      # Max resistant An. funestus (day^-1)
    beta_R0: float = 0.075    # Min resistant An. funestus (day^-1)

    # ---- Intervention coverage (0-1) ----
    b1: float = 0.6           # ITN coverage
    b2: float = 0.4           # IRS coverage
    b3: float = 0.3           # Biolarvicide coverage

    # ---- Intervention useful life (days) ----
    T1: float = 1095.0        # ITN (3 years)
    T2: float = 365.0         # IRS (1 year)
    T3: float = 14.0          # Biolarvicides (14 days)

    # ---- ITN initial killing efficacy (ξ_{j0} in table) ----
    xi_A_itn: float = 1.0     # vs susceptible An. arabiensis
    xi_r_itn: float = 0.753   # vs resistant An. arabiensis
    xi_B_itn: float = 1.0     # vs susceptible An. funestus
    xi_R_itn: float = 0.644   # vs resistant An. funestus

    # ---- IRS initial killing efficacy (ξ_{j1} in table) ----
    xi_A_irs: float = 1.0     # vs susceptible An. arabiensis
    xi_r_irs: float = 0.976   # vs resistant An. arabiensis
    xi_B_irs: float = 1.0     # vs susceptible An. funestus
    xi_R_irs: float = 1.0     # vs resistant An. funestus

    # ---- Biolarvicide initial killing efficacy ----
    tau_c: float = 0.98

    # ---- Resistance intensity (Table 2: e_1, e_2 in [0,1]) ----
    # Scales killing efficacy of resistant mosquitoes: ξ_eff = ξ * (1 - e).
    # 0 = fully susceptible to insecticides, 1 = fully resistant.
    # Fig 4 uses moderate resistance intensity = 0.5.
    e1: float = 0.5           # An. arabiensis resistance intensity
    e2: float = 0.5           # An. funestus resistance intensity

    # Eqs. (5)–(6): forces of infection use λ = … / N_H. Appendix 1 prints the
    # same ODEs without /N_H (internal inconsistency in the paper). Match the
    # main-text definition so dynamics align with the published simulations.
    divide_forces_by_nh: bool = True

    # Table 2 π_A, π_B as written yield ~O(1) adult mosquitoes vs N_H≈10³ humans,
    # so P₀(∑βI)/N_H is negligible unless immature input is scaled. This factor
    # multiplies π_A, π_B only (larval inflow); tune to match entomological scale
    # implied by the paper’s figures (order-hundreds to thousands of adults).
    vector_input_scale: float = 2000.0

# -- Time-dependent functions ------------------------------------------------

def _logit(p):
    """Logit transform: ln(p / (1-p))."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _sigmoid(x):
    """Sigmoid / logistic function: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def intervention_waning(t, T, p):
    """
    Waning factor for a single intervention (inner term of eq 7).

    At the start of each period (t mod T = 0) returns 1 (full efficacy).
    Decays toward 0 as t approaches end of period.
    """
    x = (t % T) / T                           # fraction through period [0, 1)
    scale = 1.0 + 1.0 / (2.0 ** p)
    term = 1.0 / (1.0 + (1.8 * x) ** p) - 1.0 / (2.0 ** p + 1.0)
    return scale * term


def mosquito_mortality(t, mu_j0, xi_itn, xi_irs, pars):
    """
    Time-dependent adult mosquito mortality rate (eq 7).

    mu_j0 is the table baseline mortality parameter (eq 7: f_j starts at mu_j0).
    Baseline mortality without interventions = sigmoid(mu_j0).
    """
    p = pars.p
    f_j = mu_j0

    # Additive intervention killing on logit scale
    f_j += pars.b1 * xi_itn * intervention_waning(t, pars.T1, p)
    f_j += pars.b2 * xi_irs * intervention_waning(t, pars.T2, p)
    f_j += pars.b3 * pars.tau_c * intervention_waning(t, pars.T3, p)

    return _sigmoid(f_j)


def biting_rate(t, beta_j1, beta_j0, xi_j_itn, pars):
    """
    Time-dependent biting rate (eq 8).

    Starts at minimum (ITN fresh) and rises toward maximum (ITN depleted).
    Only ITN coverage (b1) affects biting rate.
    """
    x = (t % pars.T1) / pars.T1
    denom = 1.0 + x ** pars.p
    return beta_j1 + (beta_j0 - beta_j1) * pars.b1 * xi_j_itn / denom


def biolarvicide_kill_rate(t, pars):
    """Effective biolarvicide killing rate for larvae (time-dependent)."""
    return pars.b3 * pars.tau_c * intervention_waning(t, pars.T3, pars.p)


# -- ODE system --------------------------------------------------------------

def derivatives(t, y, pars):
    """Compute all 18 dY/dt (eq 9, reconstructed from Fig 1 and eqs 1-8)."""
    dy = np.zeros(N_STATES)

    # Unpack state variables
    s_H, e_H, i_H, r_H = y[S_H], y[E_H], y[I_H], y[R_H]
    l_A, s_A, e_A, i_A = y[L_A], y[S_A], y[E_A], y[I_A]
    s_r, e_r, i_r       = y[S_r], y[E_r], y[I_r]
    l_B, s_B, e_B, i_B = y[L_B], y[S_B], y[E_B], y[I_B]
    s_RR, e_RR, i_RR    = y[S_R], y[E_R], y[I_R]


    # Effective killing efficacies (resistance intensity scales resistant ξ)
    # Susceptible mosquitoes: full ξ (unaffected by resistance intensity)
    # Resistant mosquitoes: ξ_eff = ξ * (1 - e), where e is resistance intensity
    re1 = 1.0 - pars.e1  # arabiensis resistance scaling
    re2 = 1.0 - pars.e2  # funestus resistance scaling

    # Time-dependent biting rates (eq 8)
    bA = biting_rate(t, pars.beta_A1, pars.beta_A0, pars.xi_A_itn, pars)
    br = biting_rate(t, pars.beta_r1, pars.beta_r0, pars.xi_r_itn * re1, pars)
    bB = biting_rate(t, pars.beta_B1, pars.beta_B0, pars.xi_B_itn, pars)
    bR = biting_rate(t, pars.beta_R1, pars.beta_R0, pars.xi_R_itn * re2, pars)

    # Forces of infection: Eqs. (5)–(6) divide by total humans N_H.
    n_h = max(s_H + e_H + i_H + r_H, 1e-12)
    bite_mos_to_hum = bA * i_A + bB * i_B + bR * i_RR + br * i_r
    if pars.divide_forces_by_nh:
        lam_H = pars.P0 * bite_mos_to_hum / n_h
        lam_A = pars.P1 * bA * i_H / n_h
        lam_B = pars.P1 * bB * i_H / n_h
        lam_r = pars.P1 * br * i_H / n_h
        lam_R = pars.P1 * bR * i_H / n_h
    else:
        lam_H = pars.P0 * bite_mos_to_hum
        lam_A = pars.P1 * bA * i_H
        lam_B = pars.P1 * bB * i_H
        lam_r = pars.P1 * br * i_H
        lam_R = pars.P1 * bR * i_H

    # Time-dependent mosquito mortality (eq 7)
    mu_A = mosquito_mortality(t, pars.mu_A0, pars.xi_A_itn, pars.xi_A_irs, pars)
    mu_r = mosquito_mortality(t, pars.mu_r0, pars.xi_r_itn * re1, pars.xi_r_irs * re1, pars)
    mu_B = mosquito_mortality(t, pars.mu_B0, pars.xi_B_itn, pars.xi_B_irs, pars)
    mu_RR = mosquito_mortality(t, pars.mu_R0, pars.xi_R_itn * re2, pars.xi_R_irs * re2, pars)

    # Biolarvicide effect on larvae
    bio = biolarvicide_kill_rate(t, pars)

    # ---- Humans (SEIR) ----
    dy[S_H] = pars.Pi_H - lam_H * s_H + pars.omega * r_H - pars.mu_H * s_H
    dy[E_H] = lam_H * s_H - (pars.epsilon_H + pars.mu_H) * e_H
    dy[I_H] = pars.epsilon_H * e_H - (pars.gamma + pars.mu_H + pars.tau) * i_H
    dy[R_H] = pars.gamma * i_H - (pars.omega + pars.mu_H) * r_H

    # ---- An. arabiensis larvae (constant input π_A, eq 9) ----
    dy[L_A] = pars.pi_A * pars.vector_input_scale - (pars.Lambda_1 + pars.mu_L + bio) * l_A

    # ---- An. arabiensis non-resistant (susceptible) adults ----
    dy[S_A] = ((1 - pars.phi) * pars.Lambda_1 * l_A
               + pars.sigma_r * s_r
               - (lam_A + pars.sigma_A + mu_A) * s_A)

    dy[E_A] = (lam_A * s_A
               + pars.sigma_r * e_r
               - (pars.epsilon_A + pars.sigma_A + mu_A) * e_A)

    dy[I_A] = (pars.epsilon_A * e_A
               + pars.sigma_r * i_r
               - (pars.sigma_A + mu_A) * i_A)

    # ---- An. arabiensis resistant adults ----
    dy[S_r] = (pars.phi * pars.Lambda_1 * l_A
               + pars.sigma_A * s_A
               - (lam_r + pars.sigma_r + mu_r) * s_r)

    dy[E_r] = (lam_r * s_r
               + pars.sigma_A * e_A
               - (pars.epsilon_r + pars.sigma_r + mu_r) * e_r)

    dy[I_r] = (pars.epsilon_r * e_r
               + pars.sigma_A * i_A
               - (pars.sigma_r + mu_r) * i_r)

    # ---- An. funestus larvae (constant input π_B, eq 9) ----
    dy[L_B] = pars.pi_B * pars.vector_input_scale - (pars.Lambda_2 + pars.mu_L + bio) * l_B

    # ---- An. funestus non-resistant (susceptible) adults ----
    dy[S_B] = ((1 - pars.Phi) * pars.Lambda_2 * l_B
               + pars.sigma_R * s_RR
               - (lam_B + pars.sigma_B + mu_B) * s_B)

    dy[E_B] = (lam_B * s_B
               + pars.sigma_R * e_RR
               - (pars.epsilon_B + pars.sigma_B + mu_B) * e_B)

    dy[I_B] = (pars.epsilon_B * e_B
               + pars.sigma_R * i_RR
               - (pars.sigma_B + mu_B) * i_B)

    # ---- An. funestus resistant adults ----
    dy[S_R] = (pars.Phi * pars.Lambda_2 * l_B
               + pars.sigma_B * s_B
               - (lam_R + pars.sigma_R + mu_RR) * s_RR)

    dy[E_R] = (lam_R * s_RR
               + pars.sigma_B * e_B
               - (pars.epsilon_R + pars.sigma_R + mu_RR) * e_RR)

    dy[I_R] = (pars.epsilon_R * e_RR
               + pars.sigma_B * i_B
               - (pars.sigma_R + mu_RR) * i_RR)

    return dy


# -- Integration and initial conditions --------------------------------------

def get_initial_conditions(pars):
    """Set up initial state vector with plausible starting populations."""
    y0 = np.zeros(N_STATES)
    k = pars.vector_input_scale

    # Humans — near disease-free equilibrium N_H ≈ Pi_H / mu_H ≈ 1122
    y0[S_H] = 800.0
    y0[E_H] = 50.0
    y0[I_H] = 120.0
    y0[R_H] = 30.0

    # Mosquito states scale with vector_input_scale (same units as π·k larval inflow).
    y0[L_A] = 0.3 * k
    y0[S_A] = 0.5 * k
    y0[E_A] = 0.3 * k
    y0[I_A] = 0.2 * k
    y0[S_r] = 0.02 * k
    y0[E_r] = 0.01 * k
    y0[I_r] = 0.01 * k

    y0[L_B] = 0.3 * k
    y0[S_B] = 0.4 * k
    y0[E_B] = 0.25 * k
    y0[I_B] = 0.15 * k
    y0[S_R] = 0.08 * k
    y0[E_R] = 0.05 * k
    y0[I_R] = 0.03 * k

    return y0


def _simulate(pars):
    """Forward Euler integration; returns ``(t, y)``."""
    n_steps = int(pars.n_days / pars.dt)
    t = np.arange(n_steps + 1) * pars.dt
    y = np.zeros((n_steps + 1, N_STATES))
    y[0] = get_initial_conditions(pars)

    for i in range(n_steps):
        dy = derivatives(t[i], y[i], pars)
        y[i + 1] = np.maximum(y[i] + pars.dt * dy, 0.0)  # enforce non-negativity

    return t, y


def run_baseline(pars=None):
    """
    Default-parameter run: integrate, print summary, and show/save the
    four-panel summary plot (``gervas_results.png``).
    """
    if pars is None:
        pars = Parameters()

    print(f'Running Gervas et al. model: {pars.n_days:.0f} days, dt={pars.dt}')
    t, y = _simulate(pars)

    nh = y[-1, S_H] + y[-1, E_H] + y[-1, I_H] + y[-1, R_H]
    n1 = y[-1, S_A] + y[-1, E_A] + y[-1, I_A] + y[-1, S_r] + y[-1, E_r] + y[-1, I_r]
    n2 = y[-1, S_B] + y[-1, E_B] + y[-1, I_B] + y[-1, S_R] + y[-1, E_R] + y[-1, I_R]
    prev = y[-1, I_H] / max(nh, 1)
    print(f'Final N_H={nh:.1f}  (S={y[-1,S_H]:.1f} E={y[-1,E_H]:.1f} I={y[-1,I_H]:.1f} R={y[-1,R_H]:.1f})')
    print(f'Final N_1(arabiensis)={n1:.1f}  N_2(funestus)={n2:.1f}')
    print(f'Malaria prevalence = {prev:.3f}')

    plot(t, y, pars)
    return t, y


# -- Plotting ----------------------------------------------------------------

def plot(t, y, pars=None):
    """Four-panel summary plot."""
    if pars is None:
        pars = Parameters()

    # Downsample to daily resolution for plotting
    step = max(1, int(1.0 / pars.dt))
    tp = t[::step]
    yp = y[::step]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1 — Human compartments
    ax = axes[0, 0]
    ax.plot(tp, yp[:, S_H], label='$S_H$ (susceptible)', color='steelblue')
    ax.plot(tp, yp[:, E_H], label='$E_H$ (exposed)', color='orange')
    ax.plot(tp, yp[:, I_H], label='$I_H$ (infectious)', color='crimson')
    ax.plot(tp, yp[:, R_H], label='$R_H$ (recovered)', color='seagreen')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('Human Compartments (SEIR)')
    ax.legend(fontsize=9)

    # Panel 2 — An. arabiensis
    ax = axes[0, 1]
    n1 = yp[:, S_A] + yp[:, E_A] + yp[:, I_A] + yp[:, S_r] + yp[:, E_r] + yp[:, I_r]
    ax.plot(tp, n1, label='Total adults $N_1$', color='black', lw=2)
    ax.plot(tp, yp[:, I_A], label='Infectious non-res. $I_A$', color='crimson')
    ax.plot(tp, yp[:, I_r], label='Infectious resistant $I_r$', color='darkred', ls='--')
    ax.plot(tp, yp[:, L_A], label='Larvae $L_A$', color='gray', ls=':')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('An. arabiensis')
    ax.legend(fontsize=9)

    # Panel 3 — An. funestus
    ax = axes[1, 0]
    n2 = yp[:, S_B] + yp[:, E_B] + yp[:, I_B] + yp[:, S_R] + yp[:, E_R] + yp[:, I_R]
    ax.plot(tp, n2, label='Total adults $N_2$', color='black', lw=2)
    ax.plot(tp, yp[:, I_B], label='Infectious non-res. $I_B$', color='crimson')
    ax.plot(tp, yp[:, I_R], label='Infectious resistant $I_R$', color='darkred', ls='--')
    ax.plot(tp, yp[:, L_B], label='Larvae $L_B$', color='gray', ls=':')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('An. funestus')
    ax.legend(fontsize=9)

    # Panel 4 — Biting rates over time
    ax = axes[1, 1]
    t_plot = np.linspace(0, pars.n_days, 2000)
    for beta1, beta0, xi, label, ls in [
        (pars.beta_A1, pars.beta_A0, pars.xi_A_itn, r'$\beta_A$ (arab. sus.)', '-'),
        (pars.beta_r1, pars.beta_r0, pars.xi_r_itn, r'$\beta_r$ (arab. res.)', '--'),
        (pars.beta_B1, pars.beta_B0, pars.xi_B_itn, r'$\beta_B$ (fun. sus.)', '-'),
        (pars.beta_R1, pars.beta_R0, pars.xi_R_itn, r'$\beta_R$ (fun. res.)', '--'),
    ]:
        br = np.array([biting_rate(ti, beta1, beta0, xi, pars) for ti in t_plot])
        ax.plot(t_plot, br, label=label, ls=ls)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Biting rate (day$^{-1}$)')
    ax.set_title('Mosquito Biting Rates (eq 8)')
    ax.legend(fontsize=9)

    fig.suptitle('Gervas et al. (2025) — Malaria Transmission Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    sc.savefig(sc.thispath()/'gervas_results.png')
    plt.show()


def scenarios(pars_base=None):
    """
    Reproduce Fig. 4 (Gervas et al. 2025): moderate pyrethroid resistance
    intensity (e1 = e2 = 0.5) and 80% coverage for each deployed tool.

    Layout matches the paper narrative: (a–c) An. arabiensis, (e–g) An. funestus
    for ITN only; ITN + IRS; ITN + biolarvicides (other coverages set to 0).

    **Why curves may still differ from the PDF**

    1. **λ divided by N_H (Eqs. 5–6)** — The main text defines forces of infection
       with ``/ N_H``; Appendix 1 omits it. This port now follows the main text
       (``divide_forces_by_nh``). Using the appendix form inflates transmission
       ~1000× for N_H≈10³ and completely changes trajectories.

    2. **Larval → adult flux** — Adults were fed a constant ``(1−φ)Λ`` inflow
       decoupled from ``L_A``, which breaks mass balance. Emergence is now
       ``(1−φ)Λ₁ L_A`` (and ``φΛ₁ L_A`` to resistants), matching a closed pool.

    3. **Vector abundance** — Table 2 ``π_A``, ``π_B`` as written imply very small
       larval pools and O(1) adults vs O(10³) humans, so ``P₀(∑βI)/N_H`` is
       negligible. ``vector_input_scale`` multiplies ``π_A``, ``π_B`` (and ICs)
       to reach plausible mosquito counts; the paper does not give this factor
       explicitly (likely implicit rescaling or different state units in their
       code).

    4. **Numerics & reporting** — Forward Euler with ``dt`` may differ from the
       authors’ Python integrator. Their “57–68% reduction” for IRS may use a
       specific time slice, endpoint, or ratio (e.g. infected fraction within the
       resistant stratum) under a slightly different mortality/biting map.

    Without their scripts, match is approximate; tune ``vector_input_scale`` and
    compare qualitative scenario ordering (ITN vs ITN+IRS vs ITN+bio).
    """
    if pars_base is None:
        pars_base = Parameters()

    cov = 0.8
    scenario_defs = [
        (
            f'ITN {cov:.0%}',
            {'b1': cov, 'b2': 0.0, 'b3': 0.0},
        ),
        (
            f'ITN + IRS ({cov:.0%} each)',
            {'b1': cov, 'b2': cov, 'b3': 0.0},
        ),
        (
            f'ITN + biolarvicide ({cov:.0%} each)',
            {'b1': cov, 'b2': 0.0, 'b3': cov},
        ),
    ]
    panel_letters_top = ('(a)', '(b)', '(c)')
    panel_letters_bot = ('(e)', '(f)', '(g)')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    for col, (scenario_label, bmap) in enumerate(scenario_defs):
        pars = replace(
            pars_base,
            e1=0.5,
            e2=0.5,
            b1=bmap['b1'],
            b2=bmap['b2'],
            b3=bmap['b3'],
        )
        t, y = _simulate(pars)
        step = max(1, int(1.0 / pars.dt))
        tp = t[::step]

        for row in (0, 1):
            ax = axes[row, col]
            if row == 0:
                letter = panel_letters_top[col]
                sp_label = 'An. arabiensis'
                s_nr = y[::step, S_A] + y[::step, E_A] + y[::step, I_A]
                s_res = y[::step, S_r] + y[::step, E_r] + y[::step, I_r]
                i_nr = y[::step, I_A]
                i_res = y[::step, I_r]
            else:
                letter = panel_letters_bot[col]
                sp_label = 'An. funestus'
                s_nr = y[::step, S_B] + y[::step, E_B] + y[::step, I_B]
                s_res = y[::step, S_R] + y[::step, E_R] + y[::step, I_R]
                i_nr = y[::step, I_B]
                i_res = y[::step, I_R]

            ax.plot(tp, s_nr, color='steelblue', lw=1.5, label='Non-resistant adults')
            ax.plot(tp, s_res, color='coral', lw=1.5, label='Resistant adults')
            ax.plot(tp, i_nr, color='steelblue', ls='--', lw=1.0, alpha=0.85, label='Infected non-resistant')
            ax.plot(tp, i_res, color='darkred', ls='--', lw=1.0, alpha=0.85, label='Infected resistant')
            ax.set_title(f'{letter} {scenario_label}\n{sp_label}', fontsize=10)
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel('Time (days)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, frameon=True)
    fig.suptitle(
        'Fig. 4 — Moderate resistance (e₁ = e₂ = 0.5); 80% coverage per active intervention',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    sc.savefig(sc.thispath()/'gervas_fig4.png')
    plt.show()


# -- Entry point -------------------------------------------------------------

if __name__ == '__main__':
    run_baseline()
    scenarios()
