"""
Intervention effect tests.

Tests cover four intervention behaviors:
  - Outbreak:  seeds infections in a sim that starts with zero prevalence
  - Drug:      reduces n_infected after mass drug administration
  - Drug (node-targeted): affects only the targeted node
  - Vaccine (vs baseline): vaccinated population accumulates fewer infections
                           than an unvaccinated control
"""
import numpy as np
import sciris as sc
import starsim as ss
import malariasim as ms

# Pre-compute emergence profile for EIR=10 (used by ode/agents mosquito tests)
_COMMON_PARS = ms.CommonParams()
# These values mean fewer mosquitoes which means faster tests
_COMMON_PARS.bites_per_female_per_day       = 0.3
_COMMON_PARS.prob_human_infected_on_bite    = 1.0
_COMMON_PARS.prob_mosquito_infected_on_bite = 1.0
_, _EMERGENCE_10 = ms.eir_to_params(10.0, common_pars=_COMMON_PARS)


def make_sim(human='sis', mosquito='ode', **kw):
    return ms.Sim(
        human     = human,
        mosquito  = mosquito,
        start     = kw.pop('start', 2000),
        stop      = kw.pop('stop',  2001),
        nodes     = kw.pop('nodes', ms.NodeGrid(populations=[1000])),
        dt        = kw.pop('dt',    ss.days(1)),
        rand_seed = kw.pop('rand_seed', 0),
        verbose   = kw.pop('verbose', 0),
        **kw,
    )


# ------------------------------------------------------------------
# Outbreak
# ------------------------------------------------------------------

@sc.timer()
def test_outbreak_seeds_infections():
    """Outbreak at day 30 produces infections in a sim that starts with zero prevalence.

    Single node, zero emergence (no mosquitoes), no initial infections.
    Without the Outbreak there would be zero infections throughout.  With it,
    agents are directly set infectious on day 30, so n_infected must be > 0
    immediately after.
    """
    sc.heading('Testing Outbreak seeds infections from zero prevalence...')

    nodes = ms.NodeGrid(populations=[1000], emergence=[[0.0] * 12])

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed',
                products = ms.Outbreak(),
                start    = 2000 + 30/365,
                coverage = 0.50,
            )
        ])

    sim = make_sim(human='sis', mosquito='ode', nodes=nodes,
                   stop=2000 + 60/365, demographics=[],
                   build_campaign=build_campaign)
    sim.run()

    disease  = sim.diseases[0]
    n_inf    = disease.results.n_infected.values

    # Before the outbreak (first 30 steps) must be zero
    assert n_inf[:30].sum() == 0, (
        f'Expected zero infections before Outbreak; got {n_inf[:30].sum()}'
    )
    # After the outbreak there must be infections
    assert n_inf[30:].sum() > 0, (
        'Expected infections after Outbreak at day 30, but n_infected stayed at zero'
    )


# ------------------------------------------------------------------
# Drug — population-level clearance
# ------------------------------------------------------------------

@sc.timer()
def test_drug_clears_infections():
    """Mass drug administration reduces n_infected by >50% within 7 days.

    Single node, high EIR so transmission sustains infection.  An Outbreak
    seeds 100% on day 10; by day 60 the population is at steady endemic
    prevalence.  A mass drug campaign at day 60 (100% coverage) should
    substantially clear infections within a week.
    """
    sc.heading('Testing drug clears infections...')

    nodes = ms.NodeGrid(populations=[1000], emergence=[_EMERGENCE_10])

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 1.0,
            ),
            ms.InterventionDistributorScheduled(
                name     = 'mda',
                products = ms.AntiMalarialDrug(mean_duration=ss.days(1.5)),
                start    = 2000 + 60/365,
                coverage = 1.0,
            ),
        ])

    sim = make_sim(human='sis', mosquito='ode', nodes=nodes,
                   stop=2000 + 80/365, demographics=[],
                   build_campaign=build_campaign)
    sim.run()

    n_inf   = sim.diseases[0].results.n_infected.values
    pre_mda = n_inf[59]           # day before drug
    post_mda = n_inf[60:68].min() # lowest point in first 7 days after drug

    assert pre_mda > 0, 'Expected endemic prevalence before drug; got 0'
    assert post_mda < pre_mda * 0.5, (
        f'Expected drug to reduce n_infected by >50% within 7 days; '
        f'pre={pre_mda}, post_min={post_mda}'
    )


# ------------------------------------------------------------------
# Drug — node-targeted
# ------------------------------------------------------------------

@sc.timer()
def test_drug_node_targeted():
    """Drug on node 0 only clears node 0; node 1 remains endemic.

    Two equal nodes, high EIR, both seeded at day 10.  Drug administered
    to node 0 only at day 60 (100% coverage).  By day 75, node 0 must
    have substantially fewer infections than just before the drug, while
    node 1 remains near its pre-drug level.
    """
    sc.heading('Testing node-targeted drug: node 0 clears, node 1 stays endemic...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [_EMERGENCE_10, _EMERGENCE_10],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 1.0,
            ),
            ms.InterventionDistributorScheduled(
                name        = 'mda_node0',
                products    = ms.AntiMalarialDrug(mean_duration=ss.days(1.5)),
                start       = 2000 + 60/365,
                eligibility = ms.HumanEligibility(node_ids=[0]),
                coverage    = 1.0,
            ),
        ])

    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2000 + 80/365, demographics=[],
                   build_campaign=build_campaign)
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    n0_inf = int(np.asarray(disease.infected[node0], dtype=bool).sum())
    n1_inf = int(np.asarray(disease.infected[node1], dtype=bool).sum())

    prev_node0 = n0_inf / len(node0)
    prev_node1 = n1_inf / len(node1)

    assert prev_node0 < 0.5, (
        f'Node 0 prevalence after node-targeted drug was {prev_node0:.1%}; '
        f'expected < 50% shortly after 100% MDA'
    )
    assert prev_node1 > 0.5, (
        f'Node 1 prevalence dropped to {prev_node1:.1%} despite receiving no drug; '
        f'expected to remain endemic (>50%)'
    )


# ------------------------------------------------------------------
# Vaccine vs baseline
# ------------------------------------------------------------------

@sc.timer()
def test_vaccine_reduces_infections_vs_baseline():
    """80% vaccine coverage produces fewer total infections than an unvaccinated run.

    Two identical sims: one with no campaign, one with an 80%-coverage vaccine
    on day 30.  The vaccinated sim must accumulate fewer total new infections
    over the full year.  Uses ODE mosquitoes for deterministic comparison.
    """
    sc.heading('Testing vaccine reduces infections vs unvaccinated baseline...')

    nodes = ms.NodeGrid(populations=[2000], emergence=[_EMERGENCE_10])
    kw = dict(
        human     = 'sis',
        mosquito  = 'ode',
        nodes     = nodes,
        stop      = 2001,
        demographics = [],
        rand_seed = 42,
    )

    # Baseline: Outbreak only to seed transmission
    def build_baseline(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 0.15,
            ),
        ])

    # Vaccinated: Outbreak + 80% vaccine on day 30
    def build_vaccinated(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 0.15,
            ),
            ms.InterventionDistributorScheduled(
                name     = 'vax',
                products = ms.MalariaVaccine(waning=ms.WaningEffect(
                    initial_efficacy    = 1.0,
                    box_duration        = ss.days(365),
                    decay_time_constant = ss.days(0),
                )),
                start    = 2000 + 30/365,
                coverage = 0.80,
            ),
        ])

    sim_base = make_sim(**kw, build_campaign=build_baseline)
    sim_vax  = make_sim(**kw, build_campaign=build_vaccinated)
    sim_base.run()
    sim_vax.run()

    inf_base = sim_base.diseases[0].results.n_infected.values.sum()
    inf_vax  = sim_vax.diseases[0].results.n_infected.values.sum()

    assert inf_vax < inf_base, (
        f'Expected vaccine to reduce total infections: '
        f'baseline={inf_base:.0f}, vaccinated={inf_vax:.0f}'
    )


if __name__ == '__main__':
    test_outbreak_seeds_infections()
    test_drug_clears_infections()
    test_drug_node_targeted()
    test_vaccine_reduces_infections_vs_baseline()
