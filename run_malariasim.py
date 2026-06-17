"""
run_malaria.py — run and compare malaria model configurations.

Runs four scenarios that cover the two human models (SIS, SEIS) crossed with
the two mosquito models (eir_only, agents), then plots prevalence side by side.

Usage
-----
    python run_malaria.py
"""
import sciris as sc
import starsim as ss
import malariasim as ms


# ------------------------------------------------------------------
# Demographics: age pyramid + balanced birth/death rates
# ------------------------------------------------------------------

AGE_BIN_EDGES     = ms.AGE_BIN_EDGES
AGE_BIN_FRACTIONS = ms.AGE_BIN_FRACTIONS
AGE_DATA          = sc.dataframe(ms.make_age_data())
DEMOGRAPHICS      = ms.make_steady_state_demographics()

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

START = 2000
STOP  = 2002

# ------------------------------------------------------------------
# Three-node spatial configuration
# Node 0 (Urban):      5 000 people — high transmission, mostly indoor biting
# Node 1 (Peri-urban): 3 000 people — moderate transmission
# Node 2 (Rural):      2 000 people — lower transmission, more outdoor biting
# ------------------------------------------------------------------

THREE_NODES = ms.NodeGrid(
    populations = [5_000, 3_000, 2_000],
    labels      = ['Urban', 'Peri-urban', 'Rural'],
    eir         = [ms.URBAN_EIR,      ms.PERI_URBAN_EIR,      ms.RURAL_EIR],
    emergence   = [ms.URBAN_EMERGENCE, ms.PERI_URBAN_EMERGENCE, ms.RURAL_EMERGENCE],
    f_indoor    = [ms.URBAN_F_INDOOR,  ms.PERI_URBAN_F_INDOOR,  ms.RURAL_F_INDOOR],
)

SCENARIOS = [
    dict(label='human-SIS + mosquito-EIR-only',  human='sis',  mosquito='eir_only', nodes=THREE_NODES),
    dict(label='human-SEIS + mosquito-EIR-only', human='seis', mosquito='eir_only', nodes=THREE_NODES),
    dict(label='human-ODE + mosquito-EIR-only',  human='ode',  mosquito='eir_only', nodes=THREE_NODES),
    dict(label='human-SIS + mosquito-ODE',       human='sis',  mosquito='ode',      nodes=THREE_NODES),
    dict(label='human-SEIS + mosquito-ODE',      human='seis', mosquito='ode',      nodes=THREE_NODES),
    dict(label='human-ODE + mosquito-ODE',       human='ode',  mosquito='ode',      nodes=THREE_NODES),
    dict(label='human-SIS + mosquito-agents',    human='sis',  mosquito='agents',   nodes=THREE_NODES),
    dict(label='human-SEIS + mosquito-agents',   human='seis', mosquito='agents',   nodes=THREE_NODES),
    dict(label='human-ODE + mosquito-agents',    human='ode',  mosquito='agents',   nodes=THREE_NODES),
]


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------

def run_malaria(do_monitor=True, do_plot=True, show_plots=False, do_save=True):
    """Run all scenarios and optionally plot results.
    
    Args:
        do_monitor (bool): whether to run the performance monitor (requires running serially)
        do_plot (bool): Plot results
        show_plots (bool): Whether to display plots interactively (only applies if do_plot=True)
        do_save (bool): If do_plot=True, also save figures    
    """

    # ----------------------------------------------------------
    # Build sims
    # ----------------------------------------------------------
    sims    = []
    for scen in SCENARIOS:
        scen = scen.copy()  # copy so SCENARIOS stays intact
        label = scen.pop('label')
        sim = ms.Sim(
            age_data      = AGE_DATA,
            age_bin_edges = AGE_BIN_EDGES,
            start         = START,
            stop          = STOP,
            dt            = ss.days(1),
            label         = label,
            rand_seed     = 0,
            demographics  = DEMOGRAPHICS,
            analyzers     = ms.AgeBinPopulation(),
            **scen,
        )
        sims.append(sim)

    # ----------------------------------------------------------
    # Create a performance monitor
    # ----------------------------------------------------------
    monitor = ms.PerformanceMonitor() if do_monitor else None

    # ----------------------------------------------------------
    # Combine into a MultiSim
    # ----------------------------------------------------------
    msim = ms.MalariaMultiSim(sims, monitor=monitor)
    
    # ----------------------------------------------------------
    # Run sims
    # ----------------------------------------------------------
    if do_monitor: # If monitoring, run the sims individually
        for sim in msim.sims:
            print(f'Running: {label}')
            monitor.start()
            sim.run()
            monitor.stop(sim)
    
    else: # Otherwise, run all the sims in parallel (no monitoring)
        msim.run() 
    
    # ----------------------------------------------------------
    # Optionally plot results
    # ----------------------------------------------------------
    if do_plot:
        msim.plot_all(show_plots=show_plots, do_save=do_save)

    return msim


if __name__ == '__main__':
    T = sc.timer()
    msim = run_malaria(show_plots=True)
    T.toc()
