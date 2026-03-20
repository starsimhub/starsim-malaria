"""
run_malaria.py — run and compare malaria model configurations.

Runs four scenarios that cover the two human models (SIS, SEIS) crossed with
the two mosquito models (eir_only, agents), then plots prevalence side by side.

Usage
-----
    python run_malaria.py
"""

import pandas as pd
import starsim as ss

from malaria import make_sim, NodeGrid
from performance import PerformanceMonitor
from results import AgeBinPopulation, plot_all


# ------------------------------------------------------------------
# Demographics: age pyramid + balanced birth/death rates
# ------------------------------------------------------------------

AGE_BIN_EDGES     = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,100]
AGE_BIN_FRACTIONS = [0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.035,0.03,0.025,0.02,0.015,0.01,0.005]

# Expand bin fractions to per-year weights for ss.People age_data
_ages, _vals = [], []
for lo, hi, frac in zip(AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:], AGE_BIN_FRACTIONS):
    per_year = frac / (hi - lo)
    for age in range(lo, hi):
        _ages.append(age)
        _vals.append(per_year)
AGE_DATA = pd.DataFrame({'age': _ages, 'value': _vals})

DEMOGRAPHICS = [
    ss.Births(birth_rate=ss.peryear(20)),
    ss.Deaths(death_rate=ss.peryear(20)),
]

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

N_AGENTS = 10_000
START    = 2000
STOP     = 2002

# ------------------------------------------------------------------
# Three-node spatial configuration
# Node 0 (Urban):      5 000 people — high transmission, mostly indoor biting
# Node 1 (Peri-urban): 3 000 people — moderate transmission
# Node 2 (Rural):      2 000 people — lower transmission, more outdoor biting
# ------------------------------------------------------------------

THREE_NODES = NodeGrid(
    populations = [5_000, 3_000, 2_000],
    labels      = ['Urban', 'Peri-urban', 'Rural'],

    # Per-node seasonal EIR (bites/person/year) — used by eir_only mosquito model.
    eir = [
        [ 5,  5,  8, 10, 14, 18, 20, 14,  8,  6,  5,  5],   # Urban    — high year-round
        [ 2,  2,  5,  6,  8, 10, 12,  8,  5,  3,  2,  2],   # Peri-urban — moderate wet-season
        [ 1,  1,  2,  3,  5,  8, 10,  6,  3,  2,  1,  1],   # Rural    — lower with sharp peak
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

monitor = PerformanceMonitor()
sims    = []

for sc in SCENARIOS:
    label = sc.pop('label')
    sim   = make_sim(
        n_agents     = N_AGENTS,
        start        = START,
        stop         = STOP,
        dt           = ss.days(1),
        label        = label,
        rand_seed    = 0,
        people       = ss.People(N_AGENTS, age_data=AGE_DATA),
        demographics = DEMOGRAPHICS,
        analyzers    = AgeBinPopulation(),
        **sc,
    )
    print(f'Running: {label}')
    monitor.start()
    sim.run()
    monitor.stop(sim)
    sims.append(sim)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

plot_all(sims, monitor)
