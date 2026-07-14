"""
EIR calibration tests.

Each of the 9 (human × mosquito) scenario combinations is run for 2 years on a
3-node grid with distinct populations and target EIRs.  An Outbreak seeds
infections at day 30 so mosquito-driven models (ode, agents) have infected
humans to sustain transmission.  After the first year of transient dynamics,
the second-year mean EIR per node is checked against the target within 10 %.
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import malariasim as ms


# ------------------------------------------------------------------
# Shared test configuration
# ------------------------------------------------------------------

START               = 2000
STOP                = 2002   # 2-year run
STOP_SEASONAL       = 2003   # 3-year run for seasonal tests
POPULATIONS       = [15000, 10000, 5000]
TARGET_ANNUAL_EIR = [10, 5, 2]
EIR_TOLERANCE     = 0.1   # 10 %


COMMON_PARS = ms.CommonParams()
# These values mean fewer mosquitoes which means faster tests
COMMON_PARS.bites_per_female_per_day       = 0.3
COMMON_PARS.prob_human_infected_on_bite    = 1.0
COMMON_PARS.prob_mosquito_infected_on_bite = 1.0


NODE_LABELS = ['Urban', 'Peri-urban', 'Rural']

# Pre-compute constant-climate node grid (shared across all scenarios)
_node_eirs, _node_emergences = [], []
for _eir in TARGET_ANNUAL_EIR:
    _e, _em = ms.eir_to_params(_eir, common_pars=COMMON_PARS)
    _node_eirs.append(_e)
    _node_emergences.append(_em)

THREE_NODES = ms.NodeGrid(
    populations = POPULATIONS,
    labels      = NODE_LABELS,
    eir         = _node_eirs,
    emergence   = _node_emergences,
)


def build_outbreak_campaign(sim, eir_only=False):
    """Seed 15 % of agents at day 30 to bootstrap mosquito-driven transmission."""
    campaign = ms.Campaign(sim)
    campaign.add(ms.InterventionDistributorScheduled(
        name     = 'Outbreak',
        products = ms.Outbreak(),
        start    = START + 30 / 365,
        coverage = 0.3,
    ))
    return campaign


SCENARIOS = [
    ('sis',  'eir_only'),
    ('seis', 'eir_only'),
    ('ode',  'eir_only'),
    ('sis',  'ode'),
    ('seis', 'ode'),
    ('ode',  'ode'),
    ('sis',  'agents'),
    ('seis', 'agents'),
    ('ode',  'agents'),
]


# ------------------------------------------------------------------
# Parametrized EIR calibration test
# ------------------------------------------------------------------

@pytest.mark.parametrize('human,mosquito', SCENARIOS)
@sc.timer()
def test_eir_calibration(human, mosquito):
    """Second-year mean EIR per node is within 10 % of the target."""
    sim = ms.Sim(
        human          = human,
        mosquito       = mosquito,
        nodes          = THREE_NODES,
        start          = START,
        stop           = STOP,
        dt             = ss.days(1),
        rand_seed      = 0,
        verbose        = 0,
        common_pars    = COMMON_PARS,
        analyzers      = ms.ResultsAnalyzer(),
        build_campaign = build_outbreak_campaign,
    )
    sim.run()

    az       = sim.analyzers[0]
    n_steps  = len(next(iter(az.node_eir_hist.values())))
    yr2_start = n_steps // 2   # second half = year 2

    for label, target_eir in zip(NODE_LABELS, TARGET_ANNUAL_EIR):
        eir_series  = az.node_eir_hist[label]
        mean_eir_yr2 = eir_series[yr2_start:].mean()
        lo = target_eir * (1 - EIR_TOLERANCE)
        hi = target_eir * (1 + EIR_TOLERANCE)
        print(lo,mean_eir_yr2,hi)
        assert lo <= mean_eir_yr2 <= hi, (
            f'[{human}+{mosquito}] Node "{label}": '
            f'year-2 mean EIR {mean_eir_yr2:.2f} outside '
            f'[{lo:.2f}, {hi:.2f}] (target={target_eir})'
        )


# ------------------------------------------------------------------
# Seasonal node grid
# ------------------------------------------------------------------

SEASONAL_SHAPES = [ms.URBAN_EIR, ms.PERI_URBAN_EIR, ms.RURAL_EIR]

_seasonal_eirs, _seasonal_emergences = [], []
for _eir, _shape in zip(TARGET_ANNUAL_EIR, SEASONAL_SHAPES):
    _e, _em = ms.eir_to_params(_eir, common_pars=COMMON_PARS, seasonal_profile=_shape)
    _seasonal_eirs.append(_e)
    _seasonal_emergences.append(_em)

THREE_NODES_SEASONAL = ms.NodeGrid(
    populations = POPULATIONS,
    labels      = NODE_LABELS,
    eir         = _seasonal_eirs,
    emergence   = _seasonal_emergences,
)


@pytest.mark.parametrize('human,mosquito', SCENARIOS)
@sc.timer()
def test_eir_calibration_seasonal(human, mosquito):
    """Second-year mean EIR per node is within tolerance of target under seasonal transmission."""
    sim = ms.Sim(
        human          = human,
        mosquito       = mosquito,
        nodes          = THREE_NODES_SEASONAL,
        start          = START,
        stop           = STOP_SEASONAL,
        dt             = ss.days(1),
        rand_seed      = 0,
        verbose        = 0,
        common_pars    = COMMON_PARS,
        analyzers      = ms.ResultsAnalyzer(),
        build_campaign = build_outbreak_campaign,
    )
    sim.run()

    az        = sim.analyzers[0]
    n_steps   = len(next(iter(az.node_eir_hist.values())))
    yr3_start = (n_steps * 2) // 3   # last third = year 3

    for label, target_eir in zip(NODE_LABELS, TARGET_ANNUAL_EIR):
        eir_series   = az.node_eir_hist[label]
        mean_eir_yr2 = eir_series[yr3_start:].mean()
        lo = target_eir * (1 - EIR_TOLERANCE)
        hi = target_eir * (1 + EIR_TOLERANCE)
        print(lo, mean_eir_yr2, hi)
        assert lo <= mean_eir_yr2 <= hi, (
            f'[{human}+{mosquito}] Node "{label}": '
            f'year-2 mean EIR {mean_eir_yr2:.2f} outside '
            f'[{lo:.2f}, {hi:.2f}] (target={target_eir})'
        )


if __name__ == '__main__':
    for human, mosquito in SCENARIOS:
        print(f'\n--- {human} + {mosquito} ---')
        test_eir_calibration(human, mosquito)
    for human, mosquito in SCENARIOS:
        print(f'\n--- {human} + {mosquito} (seasonal) ---')
        test_eir_calibration_seasonal(human, mosquito)
