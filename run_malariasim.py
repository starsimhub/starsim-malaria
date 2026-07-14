"""
run_malariasim.py — run and compare malaria model configurations.

Runs nine scenarios covering the three human models (SIS, SEIS, ODE) crossed
with the three mosquito models (eir_only, ode, agents), then plots prevalence
side by side.

Usage
-----
    python run_malariasim.py
"""
import gc
import os
import sciris as sc
import starsim as ss
import malariasim as ms


def _release_sim(sim):
    """Free heavy runtime arrays after a monitored run.

    Deletes the ss.People agent arrays and the mosquito connector's per-slot
    arrays so the next sim starts with a clean heap.  Stored results
    (sim.results, sim.analyzers) are kept for post-run plotting.
    """
    if hasattr(sim, 'people') and sim.people is not None:
        del sim.people
    _MOS_ARRAYS = ('alive', 'susceptible', 'exposed', 'infectious',
                   'ti_infectious', 'ti_dead', 'node_id', 'Sm', 'Em', 'Im')
    for conn in sim.connectors.values():
        for attr in _MOS_ARRAYS:
            if hasattr(conn, attr):
                try:
                    delattr(conn, attr)
                except AttributeError:
                    pass
    gc.collect()


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
AGE_DATA          = sc.dataframe(ms.make_age_data(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS))
DEMOGRAPHICS      = ms.make_steady_state_demographics(edges=AGE_BIN_EDGES, fractions=AGE_BIN_FRACTIONS)

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

START = 2000
STOP  = 2010

# ------------------------------------------------------------------
# Campaign — applied to all scenarios.
# Outbreak (non-eir_only only): 15 % coverage at year 2000.1.
# Vaccine (year 2001): 100 % coverage, nodes 0 & 2, 400-day box, no waning.
# ITN     (year 2003): 500 % coverage, nodes 1 & 2
# Drug    (year 2005): 50 % coverage, ages 0-20, nodes 0 & 1.
# ------------------------------------------------------------------

def build_campaign(sim, eir_only=False):
    vaccine = ms.MalariaVaccine(name='vaccine', waning=ms.WaningEffect(
        initial_efficacy    = 1.0,
        box_duration        = ss.days(400),
        decay_time_constant = ss.days(0),
    ))
    #bednet_0001 = ms.Bednet(
    #    name      = "product_bednet_0001",
    #    repelling = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    blocking  = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    killing   = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #)
    #bednet_1001 = ms.Bednet(
    #    name      = "product_bednet_1001",
    #    repelling = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    blocking  = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    killing   = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #)
    #bednet_0101 = ms.Bednet(
    #    name      = "product_bednet_0101",
    #    repelling = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    blocking  = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    killing   = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #)
    #bednet_0011 = ms.Bednet(
    #    name      = "product_bednet_0011",
    #    repelling = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    blocking  = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    killing   = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #)
    #bednet_0111 = ms.Bednet(
    #    name      = "product_bednet_0111",
    #    repelling = ms.WaningEffect(initial_efficacy=0.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    blocking  = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    killing   = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #)
    #bednet_1111 = ms.Bednet(
    #    name      = "product_bednet_1111",
    #    repelling = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    blocking  = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    killing   = ms.WaningEffect(initial_efficacy=0.5, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #    usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(90), decay_time_constant=ss.days(0)),
    #)
    bednet = ms.Bednet(
        name      = "product_bednet",
        repelling = ms.WaningEffect(initial_efficacy=0.8, box_duration=ss.days(180), decay_time_constant=ss.days(0)),
        blocking  = ms.WaningEffect(initial_efficacy=0.9, box_duration=ss.days(180), decay_time_constant=ss.days(0)),
        killing   = ms.WaningEffect(initial_efficacy=0.6, box_duration=ss.days(180), decay_time_constant=ss.days(0)),
        usage     = ms.WaningEffect(initial_efficacy=1.0, box_duration=ss.days(180), decay_time_constant=ss.days(0)),
    )

    campaign = ms.Campaign(sim)
    if not eir_only:
        campaign.add(
            ms.InterventionDistributorScheduled(
                name     = 'Outbreak',
                products = ms.Outbreak(),
                start    = START + 0.1,
                coverage = 0.15,
            )
        )
    if USE_INTERVENTIONS:
        campaign.add(
            ms.InterventionDistributorScheduled(
                name        = 'vacc_yr1',
                products    = vaccine,
                start       = 2003,
                eligibility = ms.HumanEligibility(min_age=ss.years(0),
                                                  max_age=ss.years(30),
                                                  node_ids=[1, 2]),
                coverage    = 0.5,
            )
        )
        #campaign.add(
        #    ms.InterventionDistributorScheduled(
        #        name        = 'vacc_yr1+200',
        #        products    = vaccine,
        #        start       = vac_start + 200/365,
        #        eligibility = ms.HumanEligibility(min_age=ss.years(0),
        #                                          max_age=ss.years(30),
        #                                          node_ids=[0,1, 2]),
        #        coverage    = 0.4,
        #    )
        #)
        #campaign.add(
        #    ms.InterventionDistributorScheduled(
        #        name        = 'bednet_1001',
        #        products    = bednet_1001,
        #        start       = 2001,
        #        eligibility = ms.HumanEligibility(node_ids=[0, 1, 2]),
        #        coverage    = 1.0,
        #    )
        #)
        #campaign.add(
        #    ms.InterventionDistributorScheduled(
        #        name        = 'bednet_0101',
        #        products    = bednet_0101,
        #        start       = 2002,
        #        eligibility = ms.HumanEligibility(node_ids=[0, 1, 2]),
        #        coverage    = 1.0,
        #    )
        #)
        #campaign.add(
        #    ms.InterventionDistributorScheduled(
        #        name        = 'bednet_0011',
        #        products    = bednet_0011,
        #        start       = 2003.0,
        #        eligibility = ms.HumanEligibility(node_ids=[0, 1, 2]),
        #        coverage    = 1.0,
        #    )
        #)
        #campaign.add(
        #    ms.InterventionDistributorScheduled(
        #        name        = 'bednet_0111',
        #        products    = bednet_0111,
        #        start       = 2004,
        #        eligibility = ms.HumanEligibility(node_ids=[0, 1, 2]),
        #        coverage    = 1.0,
        #    )
        #)
        campaign.add(
            ms.InterventionDistributorScheduled(
                name        = 'bednet_2003',
                products    = bednet,
                start       = 2005.0,
                eligibility = ms.HumanEligibility(node_ids=[0, 2]),
                coverage    = 0.5,
            )
        )
        campaign.add(
            ms.InterventionDistributorScheduled(
                name        = 'drug_yr5',
                products    = ms.AntiMalarialDrug(mean_duration=ss.days(1.5)),
                start       = 2007,
                eligibility = ms.HumanEligibility(min_age=ss.years(0),
                                                  max_age=ss.years(20),
                                                  node_ids=[0, 1]),
                coverage    = 0.5,
            )
        )
    return campaign


# ------------------------------------------------------------------
# Three-node spatial configuration
# ------------------------------------------------------------------

THREE_NODES = ms.NodeGrid(
    populations = POPULATIONS,
    labels      = ['Urban', 'Peri-urban', 'Rural'],
    eir         = node_eirs,
    emergence   = node_emergences,
    f_indoor    = f_indoors
)

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

SCENARIOS = [
    dict(label='human-SIS + mosquito-EIR-only',  human='sis',  mosquito='eir_only', nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-SEIS + mosquito-EIR-only', human='seis', mosquito='eir_only', nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-ODE + mosquito-EIR-only',  human='ode',  mosquito='eir_only', nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-SIS + mosquito-ODE',       human='sis',  mosquito='ode',      nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-SEIS + mosquito-ODE',      human='seis', mosquito='ode',      nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-ODE + mosquito-ODE',       human='ode',  mosquito='ode',      nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-SIS + mosquito-agents',    human='sis',  mosquito='agents',   nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-SEIS + mosquito-agents',   human='seis', mosquito='agents',   nodes=THREE_NODES, build_campaign=build_campaign),
    dict(label='human-ODE + mosquito-agents',    human='ode',  mosquito='agents',   nodes=THREE_NODES, build_campaign=build_campaign),
]


# ------------------------------------------------------------------
# Build and run
# ------------------------------------------------------------------

def run_malaria(do_monitor=True, do_pop_plot=False, do_plot=True, show_plots=False, do_save=True,
                do_compartments=False, do_mosquito_compartments=False, do_mosquito_by_node=True):
    """Run all scenarios and optionally plot results.

    Args:
        do_monitor (bool): whether to run the performance monitor (requires running serially)
        do_pop_plot (bool): plot the population plot per age bin
        do_plot (bool): Plot results
        show_plots (bool): Whether to display plots interactively (only applies if do_plot=True)
        do_save (bool): If do_plot=True, also save figures
        do_compartments (bool): plot the human S/E/I compartment fractions (default off)
        do_mosquito_compartments (bool): plot mosquito SEI compartments (default off)
        do_mosquito_by_node (bool): plot per-node mosquito counts (default on)
    """

    def _make_sim(scen):
        return ms.Sim(
            age_data      = AGE_DATA,
            age_bin_edges = AGE_BIN_EDGES,
            start         = START,
            stop          = STOP,
            dt            = ss.days(1),
            label         = label,
            rand_seed     = 0,
            demographics  = DEMOGRAPHICS,
            analyzers     = ms.ResultsAnalyzer(),
            common_pars   = COMMON_PARS,
            **scen,
        )

    if do_monitor:
        # Build, run, and release each sim one at a time so that the heavy
        # agent arrays from one run are freed before the next sim is
        # constructed, keeping each scenario's peak RSS independent.
        monitor = ms.PerformanceMonitor()
        sims = []
        for scen in SCENARIOS:
            scen  = scen.copy()
            label = scen.pop('label')
            print(f'Running: {label}')
            monitor.start()
            sim = _make_sim(scen)
            sim.run()
            monitor.stop()
            _release_sim(sim)
            monitor.record(sim)
            sims.append(sim)
        msim = ms.MalariaMultiSim(sims, monitor=monitor)

    else:
        # Build all sims upfront and run in parallel.
        sims = [_make_sim(s) for s in SCENARIOS]
        msim = ms.MalariaMultiSim(sims)
        msim.run()
    
    # ----------------------------------------------------------
    # Optionally plot results
    # ----------------------------------------------------------
    if do_plot:
        msim.plot_all(do_pop_plot=do_pop_plot, show_plots=show_plots, do_save=do_save,
                      prefix=make_run_prefix(), do_compartments=do_compartments,
                      do_mosquito_compartments=do_mosquito_compartments,
                      do_mosquito_by_node=do_mosquito_by_node)

    return msim


if __name__ == '__main__':
    T = sc.timer()
    msim = run_malaria(show_plots=True)
    T.toc()
