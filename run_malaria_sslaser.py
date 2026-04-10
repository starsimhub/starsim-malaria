"""
run_malaria_sslaser.py — run the Starsim-native malaria_sslaser model and plot outputs.

Uses the same three-node seasonal EIR setup as ``malaria_sslaser/tests/test_malaria_laser.py``;
this script extends that with a longer horizon (~4 years, comparable to the old LASER nticks=1460)
and saves a simple multi-panel figure.

Usage
-----
    python run_malaria_sslaser.py
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

import malaria_sslaser as ms


# ---------------------------------------------------------------------------
# Spatial forcing: 3 nodes × 12 monthly EIR values (bites / person / year)
# ---------------------------------------------------------------------------

def build_eir_profiles():
    """Per-node monthly EIR profiles (matches integration test)."""
    return np.array([
        [0.01, 0.01, 0.02, 0.04, 0.07, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01],
        [0.02, 0.02, 0.03, 0.08, 0.09, 0.20, 0.12, 0.10, 0.06, 0.03, 0.02, 0.02],
        [0.01, 0.01, 0.02, 0.05, 0.08, 0.11, 0.09, 0.06, 0.04, 0.02, 0.01, 0.01],
    ])


def build_sim():
    nodes = ms.NodeGrid(
        populations=[5000, 3000, 2000],
        labels=['Urban', 'Peri-urban', 'Rural'],
        eir=build_eir_profiles(),
    )

    return ms.Sim(
        human='sis',
        mosquito='eir_only',
        nodes=nodes,
        start=2000,
        stop=2004,
        dt=ss.days(1),
        label='SIS + EIR-only (sslaser)',
        human_pars=dict(
            prob_human_infected_on_bite=0.15,
            infection_duration_years=ss.normal(loc=120 / 365, scale=30 / 365),
            default_infectiousness=0.25,
            init_prev=ss.bernoulli(p=0.01),
        ),
        mosquito_pars=dict(
            f_indoor=0.7,
        ),
        rand_seed=123,
    )


def plot_results(sim, out_path='malaria_results_sslaser.png', show=False):
    """Plot LASER-style panels: infections, prevalence, mean indoor/outdoor EIR."""
    res = sim.results
    hum = res.malariahumansis
    mos = res.mosquitoeseironly
    t = np.arange(len(hum.n_infected))

    n_inf = hum.n_infected
    new_inf = hum.new_infections
    prev = hum.prevalence
    eir_in = mos.eir_indoor_annual
    eir_out = mos.eir_outdoor_annual

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout='constrained')
    fig.suptitle(getattr(sim, 'label', 'malaria_sslaser'))

    axes[0].plot(t, n_inf, color='C0', lw=1.0, label='Total infectious')
    axes[0].plot(t, new_inf, color='C1', lw=1.0, label='New infections (per day)', alpha=0.9)
    axes[0].set_ylabel('People')
    axes[0].set_title('Infections over time')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, lw=0.5)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(t, prev, color='C0', lw=1.0)
    axes[1].set_ylabel('Prevalence')
    axes[1].set_title('Prevalence (infected / alive)')
    axes[1].grid(True, lw=0.5)
    axes[1].set_ylim(bottom=0)

    axes[2].plot(t, eir_in, color='C0', lw=1.0, label='Mean EIR indoor')
    axes[2].plot(t, eir_out, color='C1', lw=1.0, label='Mean EIR outdoor', alpha=0.9)
    axes[2].set_ylabel('EIR')
    axes[2].set_xlabel('Tick (day)')
    axes[2].set_title('Mosquito forcing (EIR)')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, lw=0.5)
    axes[2].set_ylim(bottom=0)
    
    if show:
        plt.show()
    else:
        fig.savefig(out_path, dpi=150)
        sc.printcyan(f'Saved figure: {out_path}')
        plt.close(fig)
    return fig


def run_malaria(do_plot=True, show_plots=False, out_path='malaria_results_sslaser.png'):
    T = sc.timer()
    sim = build_sim()
    sim.run()
    sc.printcyan(f'Elapsed: {T.total:n} s')
    if do_plot:
        plot_results(sim, out_path=out_path, show=show_plots)
    return sim


if __name__ == '__main__':
    run_malaria(show_plots=True)
