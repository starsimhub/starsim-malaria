"""
run_sweep_b_and_c.py — sweep prob_human_infected_on_bite (b) and
prob_mosquito_infected_on_bite (c) to find parameter combinations that
produce a target equilibrium prevalence.

Setup
-----
- Human  : SIS ABM  (simplest model, cleanest equilibrium signal)
- Mosquito: ODE (dynamic mosquito population — c has feedback effect on EIR)
- Node   : single urban node (high transmission)
- No demographics (births/deaths): we want steady-state transmission only
- Run 20 years; measure mean prevalence over the last 5 years

Output
------
1. Heatmap of equilibrium prevalence for every (b, c) combination.
   The target band (TARGET_LO – TARGET_HI %) is outlined in white.
2. Console table of (b, c) pairs inside the target band, sorted by
   distance from the band midpoint, plus a recommended pair.

Usage
-----
    python run_sweep_b_and_c.py

Adjust B_VALUES, C_VALUES, and TARGET_* at the top of the file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sciris as sc
import starsim as ss
import malariasim as ms

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

B_VALUES   = np.round(np.arange(0.40, 0.50, 0.02), 2)
C_VALUES   = np.round(np.arange(0.20, 0.30, 0.02), 2)

TARGET_LO  = 40.0   # % — lower bound of desired prevalence band
TARGET_HI  = 50.0   # % — upper bound

N_AGENTS   = 5_000  # agents per sim (balance of speed and noise)
N_YEARS    = 20     # total run length
EQUIL_LAST = 5      # years at end used for equilibrium estimate
RAND_SEED  = 0

# Urban node — high transmission, matches row 0 of THREE_NODES in run_malariasim.py
URBAN_NODE = dict(nodes=[N_AGENTS], eir=ms.URBAN_EIR, emergence=ms.URBAN_EMERGENCE, f_indoor=ms.URBAN_F_INDOOR)

# ------------------------------------------------------------------
# Build and run all sims
# ------------------------------------------------------------------

def run_sweep():
    n_total = len(B_VALUES) * len(C_VALUES)
    print(f'Sweep: {len(B_VALUES)} b-values × {len(C_VALUES)} c-values = {n_total} sims '
          f'({N_YEARS} years each, {N_AGENTS:,} agents)')

    sims = []
    for b in B_VALUES:
        for c in C_VALUES:
            sim = ms.Sim(
                human        = 'sis',
                mosquito     = 'ode',
                solver       = 'ode',
                start        = 2000,
                stop         = 2000 + N_YEARS,
                dt           = ss.days(1),
                prob_human_infected_on_bite    = float(b),
                prob_mosquito_infected_on_bite = float(c),
                **URBAN_NODE,
                rand_seed    = RAND_SEED,
                label        = f'b={b:.2f} c={c:.2f}',
                verbose      = 0,
            )
            sims.append(sim)

    msim = ms.MalariaMultiSim(sims)
    T = sc.timer()
    msim.run()
    T.toc(label=f'Total sweep ({n_total} sims)')
    return msim


# ------------------------------------------------------------------
# Extract equilibrium prevalence
# ------------------------------------------------------------------

def extract_prevalence(msim):
    """
    Return a 2-D array (n_b × n_c) of mean prevalence (%) over the
    last EQUIL_LAST years of each sim.
    """
    prev = np.full((len(B_VALUES), len(C_VALUES)), np.nan)

    for idx, sim in enumerate(msim.sims):
        ib = idx // len(C_VALUES)
        ic = idx  % len(C_VALUES)

        res     = sim.results
        timevec = res.timevec.years
        cutoff  = timevec[-1] - EQUIL_LAST
        mask    = timevec >= cutoff

        prev[ib, ic] = float(np.mean(res.malaria.prevalence[mask] * 100))

    return prev


# ------------------------------------------------------------------
# Print recommendation
# ------------------------------------------------------------------

def print_recommendation(prev):
    target_mid = (TARGET_LO + TARGET_HI) / 2.0
    hits = []
    for ib, b in enumerate(B_VALUES):
        for ic, c in enumerate(C_VALUES):
            p = prev[ib, ic]
            if TARGET_LO <= p <= TARGET_HI:
                hits.append((abs(p - target_mid), b, c, p))

    print(f'\nTarget band: {TARGET_LO}–{TARGET_HI}%  (midpoint {target_mid}%)')
    if not hits:
        print('No (b, c) pair fell inside the target band.')
        # find the closest pair
        flat_idx = np.argmin(np.abs(prev - target_mid))
        ib, ic = np.unravel_index(flat_idx, prev.shape)
        print(f'Closest pair: b={B_VALUES[ib]:.2f}  c={C_VALUES[ic]:.2f}  '
              f'prevalence={prev[ib, ic]:.2f}%')
        return

    hits.sort()
    print(f'{"b":>6}  {"c":>6}  {"prevalence":>12}  {"dist from mid":>14}')
    print('-' * 44)
    for dist, b, c, p in hits:
        marker = '  ← recommended' if hits[0] == (dist, b, c, p) else ''
        print(f'{b:6.2f}  {c:6.2f}  {p:11.2f}%  {dist:13.2f}%{marker}')


# ------------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------------

def plot_heatmap(prev, show_plot=True, do_save=True):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap — pcolormesh with shading='flat' needs n+1 edges for n cells
    step = C_VALUES[1] - C_VALUES[0]
    c_edges = np.append(C_VALUES - step / 2, C_VALUES[-1] + step / 2)
    b_edges = np.append(B_VALUES - step / 2, B_VALUES[-1] + step / 2)
    im = ax.pcolormesh(
        c_edges,
        b_edges,
        prev,
        cmap='RdYlGn_r',
        vmin=0, vmax=100,
        shading='flat',
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Equilibrium prevalence (%)', fontsize=11)

    # Highlight cells in the target band with a white border and light fill.
    # Drawing rectangles directly avoids NaN/interpolation artefacts that
    # contourf produces on sparse heatmap grids.
    for ib, b in enumerate(B_VALUES):
        for ic, c in enumerate(C_VALUES):
            if TARGET_LO <= prev[ib, ic] <= TARGET_HI:
                rect = plt.Rectangle(
                    (c - step / 2, b - step / 2), step, step,
                    linewidth=2, edgecolor='white',
                    facecolor='white', alpha=0.30,
                )
                ax.add_patch(rect)

    # Annotate each cell with its prevalence value
    for ib, b in enumerate(B_VALUES):
        for ic, c in enumerate(C_VALUES):
            p = prev[ib, ic]
            txt_color = 'white' if p > 65 or p < 20 else 'black'
            ax.text(c, b, f'{p:.2f}%', ha='center', va='center',
                    fontsize=7, color=txt_color, fontweight='bold')

    ax.set_xlabel('c  (prob mosquito infected on bite)', fontsize=11)
    ax.set_ylabel('b  (prob human infected on bite)',    fontsize=11)
    ax.set_xticks(C_VALUES)
    ax.set_yticks(B_VALUES)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    ax.set_title(
        f'Equilibrium prevalence — human-SIS + mosquito-ODE\n'
        f'(urban node)\n'
        f'White band = {TARGET_LO}–{TARGET_HI}% target  |  '
        f'{N_AGENTS:,} agents  |  mean over last {EQUIL_LAST} of {N_YEARS} years',
        fontsize=10,
    )

    fig.tight_layout()
    if do_save:
        fig.savefig('sweep_b_and_c.png', dpi=150)
        print('Plot saved to sweep_b_and_c.png')
    if show_plot:
        plt.show()
    return fig


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    msim = run_sweep()
    prev = extract_prevalence(msim)
    del msim   # free simulation data before showing plot so exit is fast
    print_recommendation(prev)
    plot_heatmap(prev, show_plot=True, do_save=True)
