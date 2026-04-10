"""
Best-effort Python translation of the malaria model in:

Gervas et al. (2025), "Modelling the impact of different intervention packages
for malaria control under varying intensities of pyrethroid resistance"

This script focuses on reproducing Figure 4 qualitatively:
- 8 panels
- An. funestus on top, An. arabiensis on bottom
- scenarios: ITN only, ITN+IRS, ITN+Larvicides, ITN+IRS+Larvicides
- moderate resistance intensity: e1 = e2 = 0.5
- intervention coverage: 80%

Important:
1) The paper's Appendix equations appear to have some transcription/notation
   issues in the mosquito recruitment terms. The flow diagram suggests adult
   recruitment should come from larvae, so this script uses larva-to-adult
   transition terms proportional to LA/LB.
2) The paper does not give a fully explicit formula showing exactly how the
   resistance intensities e1/e2 modify the time-varying biting and mortality
   functions. This script therefore uses a transparent approximation:
      - resistant mosquitoes receive a reduced intervention killing effect
        scaled by (1 - resistance_intensity)
      - ITN personal protection in biting rate is also reduced by resistance
3) Consequently this is a faithful equation-to-code translation where possible,
   plus explicit assumptions where the paper is underspecified. Expect a
   qualitative reproduction of Figure 4, not a pixel-perfect match.

References used from the paper:
- compartments and flow diagram: Fig. 1
- force of infection: eqs. (5)-(6)
- mortality and biting functions: eqs. (7)-(8)
- baseline parameters: Table 2
- target figure: Fig. 4

Run:
    python replicate_figure4.py

Outputs:
    figure4_replica.png
"""

from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

@dataclass
class Params:
    # Human demography / disease
    pi_H: float = 0.046
    mu_H: float = 0.000041
    eps_H: float = 0.071
    tau: float = 0.00009
    gamma: float = 0.0035
    omega: float = 0.00055

    # Mosquito recruitment / mutation / progression
    lam1: float = 0.79     # An. arabiensis larva->adult scaling
    lam2: float = 0.85     # An. funestus larva->adult scaling
    phi: float = 0.02      # resistant fraction for arabiensis
    rho: float = 0.20      # resistant fraction for funestus

    sigma_A: float = 0.02
    sigma_r: float = 0.04
    sigma_B: float = 0.01
    sigma_R: float = 0.05

    eps_A: float = 0.1
    eps_r: float = 0.1
    eps_B: float = 0.1
    eps_R: float = 0.1

    # Biting rates: min/max
    beta_A1: float = 0.5
    beta_A0: float = 0.07
    beta_r1: float = 0.6
    beta_r0: float = 0.075
    beta_B1: float = 0.5
    beta_B0: float = 0.07
    beta_R1: float = 0.7
    beta_R0: float = 0.075

    # Natural mortality
    mu_A0: float = 0.24
    mu_r0: float = 0.025
    mu_B0: float = 0.14
    mu_R0: float = 0.02

    # Transmission
    P0: float = 0.018
    P1: float = 0.28

    # Larvae
    pi_A: float = 0.325
    pi_B: float = 0.325
    mu_L: float = 0.2
    tau_L: float = 0.98

    # Initial intervention efficacies from Table 2
    # ITN killing efficacy
    xi_A0: float = 1.0
    xi_r0: float = 0.753
    xi_B0: float = 1.0
    xi_R0: float = 0.644

    # IRS killing efficacy
    xi_A1: float = 1.0
    xi_r1: float = 0.976
    xi_B1: float = 1.0
    xi_R1: float = 1.0

    # Biolarvicide killing efficacy
    xi_L: float = 0.98

    # Tool lifetimes
    T1: float = 3 * 365.0
    T2: float = 365.0
    T3: float = 14.0

    # Shape
    p: float = 4.0

    # Resistance intensities
    e1: float = 0.5   # An. arabiensis
    e2: float = 0.5   # An. funestus

    # Coverages
    b1: float = 0.8   # ITN
    b2: float = 0.0   # IRS
    b3: float = 0.0   # Larvicides


# ---------------------------------------------------------------------
# Time-varying intervention functions
# ---------------------------------------------------------------------

def efficacy_decay(t: float, T: float, p: float) -> float:
    """Shared decay kernel from the paper's eq. (7), simplified into a reusable piece."""
    x = (1.8 * (t % T) / T) ** p
    return (1.0 / (1.0 + x) - 1.0 / (2.0 ** p + 1.0))


def actual_mortality(f: float) -> float:
    # paper: mu_j = exp(f_j(t)) / exp(1 + f_j(t)) = exp(-1) if interpreted literally,
    # which is clearly inconsistent. We use the standard logistic transform instead.
    return 1.0 / (1.0 + np.exp(-f))


def tool_effective_killing(base_xi: float, coverage: float, decay: float, resistance: float, resistant: bool) -> float:
    """
    Best-effort assumption:
    - susceptible mosquitoes keep the nominal intervention efficacy
    - resistant mosquitoes retain only a (1 - resistance) fraction of that killing effect
    """
    scale = (1.0 - resistance) if resistant else 1.0
    return coverage * base_xi * decay * scale


def mortality_rate(t: float, mu0: float, xi_itn: float, xi_irs: float, resistance: float, resistant: bool, pars: Params) -> float:
    d1 = efficacy_decay(t, pars.T1, pars.p)
    d2 = efficacy_decay(t, pars.T2, pars.p)
    d3 = efficacy_decay(t, pars.T3, pars.p)

    # From the paper, mortality is increased by ITN + IRS + biolarvicide effects.
    # We apply biolarvicide only to larvae elsewhere, not adult mortality.
    extra = (
        tool_effective_killing(xi_itn, pars.b1, d1, resistance, resistant)
        + tool_effective_killing(xi_irs, pars.b2, d2, resistance, resistant)
    )
    return mu0 + extra


def biting_rate(t: float, beta_min: float, beta_max: float, resistance: float, resistant: bool, pars: Params) -> float:
    """
    Based on eq. (8): biting rises as ITN efficacy wanes.
    Resistant mosquitoes are less affected by ITN personal protection.
    """
    x = ((t % pars.T1) / pars.T1) ** pars.p
    protection = pars.b1 * (1.0 - resistance if resistant else 1.0)
    return beta_max + (beta_min - beta_max) * protection / (1.0 + x)


def larvicide_mortality(t: float, pars: Params) -> float:
    d3 = efficacy_decay(t, pars.T3, pars.p)
    return pars.mu_L + pars.b3 * pars.xi_L * d3


# ---------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------

IDX = {
    "SH": 0, "EH": 1, "IH": 2, "RH": 3,
    "LA": 4, "SA": 5, "EA": 6, "IA": 7, "Sr": 8, "Er": 9, "Ir": 10,
    "LB": 11, "SB": 12, "EB": 13, "IB": 14, "SR": 15, "ER": 16, "IR": 17,
}


def rhs(t: float, y: np.ndarray, p: Params) -> np.ndarray:
    SH, EH, IH, RH, LA, SA, EA, IA, Sr, Er, Ir, LB, SB, EB, IB, SR, ER, IR = y

    # Time-varying mosquito mortality and biting
    muA = mortality_rate(t, p.mu_A0, p.xi_A0, p.xi_A1, p.e1, False, p)
    mur = mortality_rate(t, p.mu_r0, p.xi_r0, p.xi_r1, p.e1, True, p)
    muB = mortality_rate(t, p.mu_B0, p.xi_B0, p.xi_B1, p.e2, False, p)
    muR = mortality_rate(t, p.mu_R0, p.xi_R0, p.xi_R1, p.e2, True, p)
    muL = larvicide_mortality(t, p)

    betaA = biting_rate(t, p.beta_A0, p.beta_A1, p.e1, False, p)
    betar = biting_rate(t, p.beta_r0, p.beta_r1, p.e1, True, p)
    betaB = biting_rate(t, p.beta_B0, p.beta_B1, p.e2, False, p)
    betaR = biting_rate(t, p.beta_R0, p.beta_R1, p.e2, True, p)

    NH = max(SH + EH + IH + RH, 1e-12)

    # Forces of infection, eqs. (5)-(6)
    lamH = p.P0 * (betaA * IA + betaB * IB + betaR * IR + betar * Ir) / NH
    lamA = p.P1 * betaA * IH / NH
    lamr = p.P1 * betar * IH / NH
    lamB = p.P1 * betaB * IH / NH
    lamR = p.P1 * betaR * IH / NH

    # Human equations
    dSH = p.pi_H + p.omega * RH - (p.mu_H + lamH) * SH
    dEH = lamH * SH - (p.eps_H + p.mu_H) * EH
    dIH = p.eps_H * EH - (p.tau + p.gamma + p.mu_H) * IH
    dRH = p.gamma * IH - (p.omega + p.mu_H) * RH

    # Larvae -> adults:
    # flowchart strongly suggests adults are recruited from larvae
    dLA = p.pi_A - (p.tau_L * p.b3 + p.mu_L + p.lam1) * LA
    dLB = p.pi_B - (p.tau_L * p.b3 + p.mu_L + p.lam2) * LB

    emerge_A_s = (1.0 - p.phi) * p.lam1 * LA
    emerge_A_r = p.phi * p.lam1 * LA
    emerge_B_s = (1.0 - p.rho) * p.lam2 * LB
    emerge_B_r = p.rho * p.lam2 * LB

    # An. arabiensis
    dSA = emerge_A_s + p.sigma_r * Sr - (muA + p.sigma_A + lamA) * SA
    dEA = lamA * SA + p.sigma_r * Er - (p.eps_A + muA + p.sigma_A) * EA
    dIA = p.eps_A * EA + p.sigma_r * Ir - (muA + p.sigma_A) * IA

    dSr = emerge_A_r + p.sigma_A * SA - (mur + p.sigma_r + lamr) * Sr
    dEr = lamr * Sr + p.sigma_A * EA - (p.eps_r + mur + p.sigma_r) * Er
    dIr = p.eps_r * Er + p.sigma_A * IA - (mur + p.sigma_r) * Ir

    # An. funestus
    dSB = emerge_B_s + p.sigma_R * SR - (muB + p.sigma_B + lamB) * SB
    dEB = lamB * SB + p.sigma_R * ER - (p.eps_B + muB + p.sigma_B) * EB
    dIB = p.eps_B * EB + p.sigma_R * IR - (muB + p.sigma_B) * IB

    dSR = emerge_B_r + p.sigma_B * SB - (muR + p.sigma_R + lamR) * SR
    dER = lamR * SR + p.sigma_B * EB - (p.eps_R + muR + p.sigma_R) * ER
    dIR = p.eps_R * ER + p.sigma_B * IB - (muR + p.sigma_R) * IR

    return np.array([
        dSH, dEH, dIH, dRH,
        dLA, dSA, dEA, dIA, dSr, dEr, dIr,
        dLB, dSB, dEB, dIB, dSR, dER, dIR,
    ])


# ---------------------------------------------------------------------
# Simulation / plotting
# ---------------------------------------------------------------------

def initial_conditions() -> np.ndarray:
    """
    Chosen to produce a qualitative figure resembling the paper:
    a transient rise followed by decay under interventions.
    """
    return np.array([
        1000.0, 5.0, 3.0, 0.0,   # humans
        2.0, 0.05, 0.001, 0.0, 0.05, 0.001, 0.0,  # arabiensis
        2.0, 0.08, 0.001, 0.0, 0.08, 0.001, 0.0,  # funestus
    ], dtype=float)


def run_scenario(pars: Params, label: str):
    t_eval = np.linspace(0, 1000, 1001)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, pars),
        t_span=(0.0, 1000.0),
        y0=initial_conditions(),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed for {label}: {sol.message}")
    return sol.t, sol.y


def plot_figure4():
    base = Params(e1=0.5, e2=0.5, b1=0.8)

    scenarios = [
        ("ITN only", replace(base, b2=0.0, b3=0.0)),
        ("ITN + IRS", replace(base, b2=0.8, b3=0.0)),
        ("ITN + Larvicides", replace(base, b2=0.0, b3=0.8)),
        ("ITN + IRS + Larvicides", replace(base, b2=0.8, b3=0.8)),
    ]

    results = [(title, *run_scenario(pars, title)) for title, pars in scenarios]

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, sharey='row')
    fig.subplots_adjust(bottom=0.20, wspace=0.15, hspace=0.28)

    styles = {
        "nonres_exposed": dict(color="black", linestyle="--", linewidth=2.2),
        "nonres_infectious": dict(color="#E6B800", linestyle="-", linewidth=2.2),
        "res_infectious": dict(color="#D62728", linestyle="-.", linewidth=2.2),
        "res_exposed": dict(color="#7F3FBF", linestyle=":", linewidth=2.4),
    }

    # Top row: An. funestus
    # Bottom row: An. arabiensis
    labels_done = False
    for j, (title, t, y) in enumerate(results):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        # funestus
        l1 = ax_top.plot(t, y[IDX["EB"]], label=r"Non-resistant mosquitoes-Exposed ($E_B$)", **styles["nonres_exposed"])[0]
        l2 = ax_top.plot(t, y[IDX["IB"]], label=r"Non-resistant mosquitoes-Infectious ($I_B$)", **styles["nonres_infectious"])[0]
        l3 = ax_top.plot(t, y[IDX["IR"]], label=r"Resistant mosquitoes-Infectious ($I_R$)", **styles["res_infectious"])[0]
        l4 = ax_top.plot(t, y[IDX["ER"]], label=r"Resistant mosquitoes-Exposed ($E_R$)", **styles["res_exposed"])[0]

        # arabiensis
        ax_bot.plot(t, y[IDX["EA"]], **styles["nonres_exposed"])
        ax_bot.plot(t, y[IDX["IA"]], **styles["nonres_infectious"])
        ax_bot.plot(t, y[IDX["Ir"]], **styles["res_infectious"])
        ax_bot.plot(t, y[IDX["Er"]], **styles["res_exposed"])

        ax_top.set_title(title, fontsize=10)
        if j == 0:
            ax_top.set_ylabel("Proportion of the vector population")
            ax_bot.set_ylabel("Proportion of the vector population")

        ax_bot.set_xlabel("Time in days")

        ax_top.text(0.5, 1.08, "An. funestus" if j == 1 else "", transform=ax_top.transAxes,
                    ha="center", va="bottom", fontsize=11, fontstyle="italic", weight="bold")
        ax_bot.text(0.5, 1.08, "An. arabiensis" if j == 1 else "", transform=ax_bot.transAxes,
                    ha="center", va="bottom", fontsize=11, fontstyle="italic", weight="bold")

        ax_top.set_xlim(0, 1000)
        ax_bot.set_xlim(0, 1000)

        # Match the paper's approximate scale
        ax_top.set_ylim(0, 0.20)
        ax_bot.set_ylim(0, 0.20)

        # panel letters
        letters = "abcdefgh"
        ax_top.text(0.02, 0.95, f"({letters[j]})", transform=ax_top.transAxes, ha="left", va="top", fontsize=9)
        ax_bot.text(0.02, 0.95, f"({letters[j+4]})", transform=ax_bot.transAxes, ha="left", va="top", fontsize=9)

    # Shared legend
    fig.legend([l1, l2, l3, l4],
               [l1.get_label(), l2.get_label(), l3.get_label(), l4.get_label()],
               loc="lower center", ncol=2, frameon=False, fontsize=10)

    fig.suptitle("Figure 4 replica (best-effort): moderate resistance intensity (50%), each intervention at 80% coverage",
                 fontsize=12)

    out = "figure4_replica.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_figure4()
