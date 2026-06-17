"""
run_population_stability.py — verify population stability with ss.Deaths.

Runs human-SIS + mosquito-EIR-only N_SEEDS times for N_YEARS years and plots
total population and per-age-bin population over time.  Individual seeds are
shown as thin lines; the mean across seeds is shown as a thick line.  A dashed
reference line marks the initial population value.

Figures produced
----------------
population_stability_total.png
    Total population over time.

population_stability_age_bins.png
    Per-age-bin population.

Usage
-----
    python run_population_stability.py

Adjust N_SEEDS, N_YEARS, and N_AGENTS at the top of the file.
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import malariasim as ms
from malariasim.results import _plain_counts

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

N_SEEDS  = 20
N_YEARS  = 50
N_AGENTS = 1_000
START    = 2000

COLOR = 'steelblue'

# ------------------------------------------------------------------
# Build and run sims
# ------------------------------------------------------------------

def run_stability(n_seeds=N_SEEDS, n_years=N_YEARS, n_agents=N_AGENTS):
    """Run N_SEEDS sims and return a MalariaMultiSim."""
    nodes = ms.NodeGrid(populations=[n_agents])
    sims  = []
    for seed in range(n_seeds):
        sim = ms.Sim(
            human         = 'sis',
            mosquito      = 'eir_only',
            age_data      = sc.dataframe(ms.make_age_data()),
            age_bin_edges = ms.AGE_BIN_EDGES,
            start         = START,
            stop          = START + n_years,
            dt            = ss.days(1),
            nodes         = nodes,
            demographics  = ms.make_steady_state_demographics(),
            analyzers     = ms.AgeBinPopulation(),
            rand_seed     = seed,
            label         = f'seed {seed}',
            verbose       = 0,
        )
        sims.append(sim)

    msim = ms.MalariaMultiSim(sims)
    print(f'Running {n_seeds} seeds × {n_years} years ({n_agents:,} agents)...')
    msim.run()
    return msim


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def _draw_runs(ax, timevec, matrix):
    """Draw individual seed lines, ±1 std band, and mean."""
    mean_vals = matrix.mean(axis=0)
    std_vals  = matrix.std(axis=0)
    for i, row in enumerate(matrix):
        ax.plot(timevec, row, lw=0.5, alpha=0.18, color=COLOR,
                label='seeds' if i == 0 else None)
    ax.fill_between(timevec, mean_vals - std_vals, mean_vals + std_vals,
                    alpha=0.25, color=COLOR, label='±1 std')
    ax.plot(timevec, mean_vals, lw=2.0, color=COLOR, label='mean')


def _finish_panel(ax, timevec, init_val, title=None, fontsize=8):
    """Add reference line, labels, grid."""
    ax.axhline(init_val, color='black', lw=1.0, linestyle='--', label='initial')
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Year',  fontsize=fontsize - 1)
    ax.set_ylabel('Count', fontsize=fontsize - 1)
    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, alpha=0.3)
    _plain_counts(ax)


# ------------------------------------------------------------------
# Figure 1: total population
# ------------------------------------------------------------------

def plot_total_population(msim, show_plot=True, do_save=True):
    timevec = msim.sims[0].results.timevec.years
    matrix  = np.array([s.results.n_alive.values for s in msim.sims])
    n_seeds = len(msim.sims)
    n_years = int(round(timevec[-1] - timevec[0]))

    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_runs(ax, timevec, matrix)
    _finish_panel(ax, timevec, matrix[:, 0].mean(), fontsize=11)

    ax.legend(fontsize=9)
    ax.set_title(
        f'Total population stability\n'
        f'{n_seeds} seeds × {n_years} years  |  {N_AGENTS:,} agents',
        fontsize=11,
    )
    ax.set_ylabel('Population (n_alive)', fontsize=10)
    fig.tight_layout()

    if do_save:
        fname = 'population_stability_total.png'
        fig.savefig(fname, dpi=150)
        print(f'Saved {fname}')
    if show_plot:
        plt.show()
    return fig


# ------------------------------------------------------------------
# Figure 2: per-age-bin population grid
# ------------------------------------------------------------------

def plot_age_bin_population(msim, show_plot=True, do_save=True):
    timevec    = msim.sims[0].results.timevec.years
    bin_labels = list(msim.sims[0].analyzers[0].hist.keys())
    n_seeds    = len(msim.sims)
    n_years    = int(round(timevec[-1] - timevec[0]))

    n_panels      = len(bin_labels)
    nrows, ncols  = sc.getrowscols(n_panels)
    fig, axes     = plt.subplots(nrows, ncols,
                                  figsize=(4 * ncols, 3.5 * nrows),
                                  sharex=True)
    fig.suptitle(
        f'Age-bin population stability\n'
        f'{n_seeds} seeds × {n_years} years  |  {N_AGENTS:,} agents',
        fontsize=13,
    )

    for i, (ax, lbl) in enumerate(zip(axes.flat, bin_labels)):
        mat      = np.array([s.analyzers[0].hist[lbl] for s in msim.sims])
        init_val = mat[:, 0].mean()
        _draw_runs(ax, timevec, mat)
        _finish_panel(ax, timevec, init_val, title=lbl)

    axes.flat[0].legend(fontsize=5)

    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()

    if do_save:
        fname = 'population_stability_age_bins.png'
        fig.savefig(fname, dpi=150)
        print(f'Saved {fname}')
    if show_plot:
        plt.show()
    return fig


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    T    = sc.timer()
    msim = run_stability()
    T.toc('Runs complete')

    plot_total_population(msim,   show_plot=True, do_save=True)
    plot_age_bin_population(msim, show_plot=True, do_save=True)
