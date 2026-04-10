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

AGE_BIN_EDGES     = [0,    5,    10,   15,   20,   25,   30,   35,   40,   45,   50,    55,   60,    65,   70,    75,   80,   100]
AGE_BIN_FRACTIONS = [0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005]

# Expand bin fractions to per-year weights for ss.People age_data
ages, vals = [], []
for lo, hi, frac in zip(AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:], AGE_BIN_FRACTIONS):
    per_year = frac / (hi - lo)
    for age in range(lo, hi):
        ages.append(age)
        vals.append(per_year)
AGE_DATA = sc.dataframe({'age': ages, 'value': vals})

DEMOGRAPHICS = [
    ss.Births(birth_rate=ss.peryear(100)),
    ss.Deaths(death_rate=ss.peryear(20)),
]

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

    # Per-node seasonal EIR (bites/person/year) — used by eir_only mosquito model.
    eir = [
        [ 5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5,  5],   # Urban      — high year-round
        [ 2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2],   # Peri-urban — moderate wet-season
        [ 1,  1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1],   # Rural      — lower with sharp peak
    ],

    # Per-node monthly emergence (new females/human/day) — used by agents model.
    emergence = [
        [0.008, 0.008, 0.015, 0.030, 0.045, 0.060, 0.060, 0.045, 0.030, 0.015, 0.008, 0.008],
        [0.005, 0.005, 0.010, 0.020, 0.030, 0.040, 0.040, 0.030, 0.020, 0.010, 0.005, 0.005],
        [0.002, 0.002, 0.005, 0.010, 0.018, 0.025, 0.025, 0.018, 0.010, 0.005, 0.002, 0.002],
    ],

    # Per-node indoor bite fraction.
    f_indoor = [0.80, 0.70, 0.55],
)

SCENARIOS = [
    dict(label='SIS + EIR-only',  human='sis',  mosquito='eir_only', nodes=THREE_NODES),
    dict(label='SEIS + EIR-only', human='seis', mosquito='eir_only', nodes=THREE_NODES),
    dict(label='SIS + agents',    human='sis',  mosquito='agents',   nodes=THREE_NODES),
    dict(label='SEIS + agents',   human='seis', mosquito='agents',   nodes=THREE_NODES),
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
            age_data     = AGE_DATA,
            start        = START,
            stop         = STOP,
            dt           = ss.days(1),
            label        = label,
            rand_seed    = 0,
            demographics = DEMOGRAPHICS,
            analyzers    = ms.AgeBinPopulation(),
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
