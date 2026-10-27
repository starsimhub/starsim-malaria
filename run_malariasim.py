"""
run_malariasim.py — run and compare malaria model configurations.

Runs eight scenarios: the two human models (SIS, SEIS) as agents and the SIS as
an ODE (solver='ode'), crossed with the three mosquito models (eir_only, ode,
agents; mosquito agents need human agents), then plots prevalence side by side.

Usage
-----
    python run_malariasim.py
"""
import os
import sciris as sc
import starsim as ss
import malariasim as ms


# ------------------------------------------------------------------
# Transmission parameters & demographics
# ------------------------------------------------------------------
POPULATIONS = [6000, 3000, 1000]
TARGET_ANNUAL_EIR = [4, 6, 8]  # bites/person/year — Urban, Peri-urban, Rural
#TARGET_ANNUAL_EIR = [5, 5, 5]  # bites/person/year — Urban, Peri-urban, Rural
USE_INTERVENTIONS = True
SEASONAL          = False
#SEASONAL_SHAPES   = [ms.URBAN_EIR, ms.URBAN_EIR, ms.URBAN_EIR]  # used when SEASONAL=True
SEASONAL_SHAPES   = [ms.URBAN_EIR, ms.PERI_URBAN_EIR, ms.RURAL_EIR]  # used when SEASONAL=True
COMMON_PARS       = ms.CommonParams()

f_indoors = [0.7, 0.7, 0.7]

node_eirs, node_emergences = [], []
for eir, shape in zip(TARGET_ANNUAL_EIR, SEASONAL_SHAPES):
    e, em = ms.eir_to_params(eir, common_pars=COMMON_PARS,
                              seasonal_profile=shape if SEASONAL else None)
    node_eirs.append(e)
    node_emergences.append(em)

AGE_BIN_EDGES     = ms.AGE_BIN_EDGES
AGE_BIN_FRACTIONS = ms.AGE_BIN_FRACTIONS
AGE_DATA          = ms.make_age_data(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS)

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

START = 2000
STOP  = 2010

# ------------------------------------------------------------------
# Campaign — applied to all scenarios.
# Outbreak (not eir_only): 15 % coverage early in 2000.
# Vaccine (2003): 50 % coverage of under-30s, Peri-urban & Rural, 400-day box, no waning.
# ITN     (2005): 50 % coverage, Urban & Rural.
# Drug    (2007): 50 % coverage of under-20s, Urban & Peri-urban.
# The vaccine and the ITN wane by boxes, which run as agents and as an ODE.
# ------------------------------------------------------------------

def campaign(mal):
    """ The interventions, as a function of the disease (so the outbreak can name its states) """
    out = []
    if mal.mosquito != 'eir_only':
        out.append(ss.Delivery(move=mal.S >> mal.I, at=ss.date('2000-02-06'), coverage=0.15, name='outbreak'))
    if not USE_INTERVENTIONS:
        return out
    node = ss.ppl.patch
    vaccine = ss.Vaccine(name='vaccine', efficacy=1.0, waning=ss.box(dur=ss.days(400)))
    bednet = ms.Bednet(name='product_bednet', repelling=0.8, blocking=0.9, killing=0.6, usage=1.0, waning=ss.box(dur=ss.days(180)))
    drug = ss.Treatment(name='drug', cure=dict(I=1.0))
    out.append(ss.vaccinate(vaccine, at='2003-01-01', coverage=0.5, where=(ss.ppl.age < 30) & (node != 'Urban'), name='vacc_yr1'))
    out.append(ss.vaccinate(bednet, at='2005-01-01', coverage=0.5, where=node != 'Peri-urban', name='bednet_2005'))
    out.append(ss.treat(drug, at='2007-01-01', coverage=0.5, where=(ss.ppl.age < 20) & (node != 'Rural'), name='drug_yr7'))
    return out


THREE_NODES = ss.geo.geography(names=['Urban', 'Peri-urban', 'Rural'], pop=POPULATIONS)
NODE_PARS   = dict(eir=node_eirs, emergence=node_emergences, f_indoor=f_indoors)

_SHAPE_CODES = {
    tuple(ms.URBAN_EIR):      'u',
    tuple(ms.PERI_URBAN_EIR): 'p',
    tuple(ms.RURAL_EIR):      'r',
}

def make_run_prefix():
    pop_tag   = ''.join(str(p // 1000) for p in POPULATIONS)
    eir_tag   = ''.join(str(int(e)) for e in TARGET_ANNUAL_EIR)
    if SEASONAL:
        shape_tag = ''.join(_SHAPE_CODES.get(tuple(s), '?') for s in SEASONAL_SHAPES)
    else:
        shape_tag = 'c' * len(TARGET_ANNUAL_EIR)
    intven_tag = 'yes' if USE_INTERVENTIONS else 'no'
    os.makedirs('malaria_results', exist_ok=True)
    return f'malaria_results/pop-{pop_tag}_eir-{eir_tag}_seasonal-{shape_tag}_intven-{intven_tag}_'

# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

SCENARIOS = [
    dict(label='human-SIS + mosquito-EIR-only',      human='sis',  mosquito='eir_only'),
    dict(label='human-SEIS + mosquito-EIR-only',     human='seis', mosquito='eir_only'),
    dict(label='human-SIS (ODE) + mosquito-EIR-only', human='sis',  mosquito='eir_only', solver='ode'),
    dict(label='human-SIS + mosquito-ODE',           human='sis',  mosquito='ode'),
    dict(label='human-SEIS + mosquito-ODE',          human='seis', mosquito='ode'),
    dict(label='human-SIS (ODE) + mosquito-ODE',     human='sis',  mosquito='ode',      solver='ode'),
    dict(label='human-SIS + mosquito-agents',        human='sis',  mosquito='agents'),
    dict(label='human-SEIS + mosquito-agents',       human='seis', mosquito='agents'),
]


def make_sim(scen):
    scen = dict(scen)
    solver = scen.pop('solver', 'abm')
    return ms.Sim(
        nodes         = THREE_NODES,
        age_data      = AGE_DATA,
        age_bin_edges = AGE_BIN_EDGES,
        start         = START,
        stop          = STOP,
        dt            = ss.days(1),
        rand_seed     = 0,
        demographics  = ms.make_steady_state_demographics(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS),
        common_pars   = COMMON_PARS,
        solver        = solver,
        interventions = campaign,
        **NODE_PARS,
        **scen,
    )


def run_malaria(do_monitor=True, do_pop_plot=False, do_plot=True, show_plots=False, do_save=True, do_mosquito_by_node=True):
    """Run all scenarios and optionally plot results.

    Args:
        do_monitor (bool): whether to run the performance monitor (requires running serially)
        do_pop_plot (bool): plot the population per age bin
        do_plot (bool): plot results
        show_plots (bool): whether to display plots interactively (only applies if do_plot=True)
        do_save (bool): if do_plot=True, also save figures
        do_mosquito_by_node (bool): plot per-node mosquito counts (default on)
    """
    if do_monitor:
        monitor = ms.PerformanceMonitor()
        sims = []
        for scen in SCENARIOS:
            print(f'Running: {scen["label"]}')
            monitor.start()
            sim = make_sim(scen)
            sim.run()
            monitor.stop()
            monitor.record(sim)
            sims.append(sim)
        msim = ms.MalariaMultiSim(sims)
    else:
        msim = ms.MalariaMultiSim([make_sim(s) for s in SCENARIOS])
        msim.run()

    if do_plot:
        prefix = make_run_prefix()
        msim.plot_all(do_pop_plot=do_pop_plot, show_plots=show_plots, do_save=do_save, prefix=prefix, do_mosquito_by_node=do_mosquito_by_node)
        if do_monitor:
            monitor.plot(out_prefix=prefix + 'perf', show_plots=show_plots, do_save=do_save)
    return msim


if __name__ == '__main__':
    T = sc.timer()
    msim = run_malaria(show_plots=True)
    T.toc()
