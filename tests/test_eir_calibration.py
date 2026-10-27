"""
EIR calibration tests

Each combination of human model, mosquito model, and solver runs on three nodes with distinct
populations and target EIRs, from `ms.eir_to_params()`. An outbreak seeds infections at day 30,
so that the mosquitoes can become infected. After the first year, the mean EIR in each node in
the second year (the third, for seasonal transmission) is checked against its target, within 10%.
"""

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import malariasim as ms

POPULATIONS       = [15000, 10000, 5000]
TARGET_ANNUAL_EIR = [10, 5, 2]
EIR_TOLERANCE     = 0.1
NODES = ss.geo.geography(names=['Urban', 'Peri-urban', 'Rural'], pop=POPULATIONS)
COMMON_PARS = ms.CommonParams(bites_per_female_per_day=0.3, prob_human_infected_on_bite=1.0, prob_mosquito_infected_on_bite=1.0)

SCENARIOS = [ # Human, mosquitoes, solver
    ('sis',  'eir_only', 'abm'),
    ('seis', 'eir_only', 'abm'),
    ('sis',  'eir_only', 'ode'),
    ('sis',  'ode',      'abm'),
    ('seis', 'ode',      'abm'),
    ('sis',  'ode',      'ode'),
    ('seis', 'ode',      'ode'),
    ('sis',  'agents',   'abm'),
    ('seis', 'agents',   'abm'),
]


def outbreak(mal):
    return ss.Delivery(move=mal.S >> mal.I, at='2000-01-31', coverage=0.3, name='outbreak')


def node_eir(sim):
    """ The annual EIR in each node at each time point """
    mal = sim.diseases.malaria
    if mal.mosquito == 'ode':
        return 365*np.asarray(sim.results.malaria.eir.by('patch'))
    elif mal.mosquito == 'agents':
        return 365*sim.connectors.mosquitoes.eir_by_node
    years = sim.t.yearvec - sim.t.yearvec[0] # EIR given: its value at each time point
    eir = mal.pars.eir.raw
    return np.array([[float(np.interp(y % 1, np.arange(13)/12, list(v.values[i].value for i in range(13)))) if isinstance(v, ss.timeseries) else float(v.value)
                      for v in eir.items] for y in years])


def check(human, mosquito, solver, seasonal, stop):
    profiles = [ms.URBAN_EIR, ms.PERI_URBAN_EIR, ms.RURAL_EIR] if seasonal else [None]*3
    eirs, emergences = zip(*[ms.eir_to_params(eir, common_pars=COMMON_PARS, seasonal_profile=prof) for eir, prof in zip(TARGET_ANNUAL_EIR, profiles)])
    sim = ms.Sim(human=human, mosquito=mosquito, solver=solver, nodes=NODES, start=2000, stop=stop, rand_seed=0, verbose=0,
                 common_pars=COMMON_PARS, eir=list(eirs), emergence=list(emergences), interventions=outbreak)
    sim.run()
    eir = node_eir(sim)
    last = eir[-366:-1].mean(axis=0)
    for label, target, mean in zip(NODES.names, TARGET_ANNUAL_EIR, last):
        lo, hi = target*(1 - EIR_TOLERANCE), target*(1 + EIR_TOLERANCE)
        assert lo <= mean <= hi, f'[{human}+{mosquito}, {solver}] node "{label}": last-year mean EIR {mean:.2f} outside [{lo:.2f}, {hi:.2f}] (target {target})'
    return sim


@pytest.mark.parametrize('human,mosquito,solver', SCENARIOS)
@sc.timer()
def test_eir_calibration(human, mosquito, solver):
    """ The second year's mean EIR in each node is within 10% of the target """
    check(human, mosquito, solver, seasonal=False, stop=2002)


@pytest.mark.parametrize('human,mosquito,solver', SCENARIOS)
@sc.timer()
def test_eir_calibration_seasonal(human, mosquito, solver):
    """ The same with seasonal transmission, in the third year """
    check(human, mosquito, solver, seasonal=True, stop=2003)


if __name__ == '__main__':
    for scen in SCENARIOS:
        test_eir_calibration(*scen)
        test_eir_calibration_seasonal(*scen)
