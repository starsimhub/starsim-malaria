"""
Test malaria simulation configurations.
"""
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
import malariasim as ms

n_agents = 1_000
do_plot = False
sc.options(interactive=False)

default_nodes = ms.NodeGrid(populations=[n_agents])


def make_sim(human='sis', mosquito='eir_only', **kw):
    """Helper to create a small, fast sim for testing."""
    return ms.Sim(
        human     = human,
        mosquito  = mosquito,
        start     = kw.pop('start', 2000),
        stop      = kw.pop('stop', 2001),
        nodes     = kw.pop('nodes', default_nodes),
        dt        = kw.pop('dt', ss.days(1)),
        rand_seed = kw.pop('rand_seed', 0),
        verbose   = kw.pop('verbose', 0),
        **kw,
    )


@sc.timer()
def test_sis_eir():
    """Test that SIS + EIR-only runs and produces infections."""
    sc.heading('Testing SIS + EIR-only...')
    sim = make_sim(human='sis', mosquito='eir_only')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with default EIR, but got {n_infected}'
    if do_plot:
        sim.plot()


@sc.timer()
def test_seis_eir():
    """Test that SEIS + EIR-only runs and produces infections."""
    sc.heading('Testing SEIS + EIR-only...')
    sim = make_sim(human='seis', mosquito='eir_only')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with default EIR, but got {n_infected}'
    if do_plot:
        sim.plot()


@sc.timer()
def test_sis_agents():
    """Test that SIS + agent-based mosquitoes runs and produces infections."""
    sc.heading('Testing SIS + agents...')
    sim = make_sim(human='sis', mosquito='agents', human_pars=dict(init_prev=ss.bernoulli(p=0.01)))
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with agent mosquitoes, but got {n_infected}'
    if do_plot:
        sim.plot()


@sc.timer()
def test_sis_ode():
    """Test that SIS + ODE mosquitoes runs and produces infections."""
    sc.heading('Testing SIS + ODE mosquitoes...')
    sim = make_sim(human='sis', mosquito='ode', human_pars=dict(init_prev=ss.bernoulli(p=0.01)))
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with ODE mosquitoes, but got {n_infected}'
    if do_plot:
        sim.plot()


@sc.timer()
def test_seis_ode():
    """Test that SEIS + ODE mosquitoes runs and produces infections."""
    sc.heading('Testing SEIS + ODE mosquitoes...')
    sim = make_sim(human='seis', mosquito='ode', human_pars=dict(init_prev=ss.bernoulli(p=0.01)))
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with ODE mosquitoes, but got {n_infected}'
    if do_plot:
        sim.plot()


@sc.timer()
def test_ode_mosquito_results():
    """Test that ODE mosquito results are populated and biologically sensible."""
    sc.heading('Testing ODE mosquito results...')
    sim = make_sim(human='sis', mosquito='ode', human_pars=dict(init_prev=ss.bernoulli(p=0.01)))
    sim.run()

    mos_res = sim.results['mosquitoesode']
    assert mos_res.n_total.values.sum() > 0,      'Expected non-zero total mosquito population'
    assert mos_res.n_infectious.values.sum() > 0, 'Expected some infectious mosquitoes'
    assert mos_res.eir_annual.values.sum() > 0,   'Expected non-zero EIR from ODE mosquitoes'

    assert (mos_res.n_susceptible.values >= 0).all(), 'Sm went negative'
    assert (mos_res.n_exposed.values     >= 0).all(), 'Em went negative'
    assert (mos_res.n_infectious.values  >= 0).all(), 'Im went negative'


@sc.timer()
def test_ode_multinode():
    """Test that ODE mosquitoes work correctly with a multi-node grid."""
    sc.heading('Testing ODE mosquitoes with multi-node grid...')
    nodes = ms.NodeGrid(
        populations = [600, 400],
        emergence   = [
            [0.008, 0.008, 0.015, 0.030, 0.045, 0.060, 0.060, 0.045, 0.030, 0.015, 0.008, 0.008],
            [0.002, 0.002, 0.005, 0.010, 0.018, 0.025, 0.025, 0.018, 0.010, 0.005, 0.002, 0.002],
        ],
    )
    sim = make_sim(human='sis', mosquito='ode', nodes=nodes, human_pars=dict(init_prev=ss.bernoulli(p=0.01)))
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected infections in multi-node ODE sim, but got {n_infected}'

    mos_mod = sim.connectors[1]
    assert (mos_mod._Sm + mos_mod._Em + mos_mod._Im > 0).all(), \
        'Expected non-zero mosquito population in every node'


@sc.timer()
def test_ode_more_emergence_more_infections():
    """Test that higher ODE mosquito emergence produces more cumulative infections."""
    sc.heading('Testing ODE: higher emergence → more infections...')

    lo_nodes = ms.NodeGrid(populations=[n_agents], emergence=[[0.002] * 12])
    hi_nodes = ms.NodeGrid(populations=[n_agents], emergence=[[0.030] * 12])

    _seed = dict(human_pars=dict(init_prev=ss.bernoulli(p=0.01)))
    sim_lo = make_sim(mosquito='ode', nodes=lo_nodes, **_seed)
    sim_hi = make_sim(mosquito='ode', nodes=hi_nodes, **_seed)
    sim_lo.run()
    sim_hi.run()

    cum_lo = sim_lo.diseases[0].results.new_infections.sum()
    cum_hi = sim_hi.diseases[0].results.new_infections.sum()
    assert cum_hi > cum_lo, (
        f'Expected more infections with high emergence than low, '
        f'but got {cum_hi} vs {cum_lo}'
    )


@sc.timer()
def test_higher_eir_more_infections():
    """Test that higher EIR produces more cumulative infections."""
    sc.heading('Testing higher EIR → more infections...')

    lo_nodes = ms.NodeGrid(populations=[n_agents], eir=[[1]*12])
    hi_nodes = ms.NodeGrid(populations=[n_agents], eir=[[20]*12])

    sim_lo = make_sim(mosquito='eir_only', nodes=lo_nodes)
    sim_hi = make_sim(mosquito='eir_only', nodes=hi_nodes)
    sim_lo.run()
    sim_hi.run()

    cum_lo = sim_lo.diseases[0].results.new_infections.sum()
    cum_hi = sim_hi.diseases[0].results.new_infections.sum()
    assert cum_hi > cum_lo, (
        f'Expected more cumulative infections with EIR=20 than EIR=1, '
        f'but got {cum_hi} vs {cum_lo}'
    )

    if do_plot:
        plt.figure()
        plt.plot(sim_lo.diseases[0].results.new_infections, label='EIR=1')
        plt.plot(sim_hi.diseases[0].results.new_infections, label='EIR=20')
        plt.legend()
        plt.title('New infections: low vs high EIR')


# ------------------------------------------------------------------
# MalariaHumanODE tests
# ------------------------------------------------------------------

AGE_BIN_EDGES     = ms.AGE_BIN_EDGES
AGE_BIN_FRACTIONS = ms.AGE_BIN_FRACTIONS


def _ode_demographics():
    """Return steady-state demographics for use with ODE human tests."""
    return ms.make_steady_state_demographics()


def _ode_age_data():
    """Return the smooth per-year age-distribution consistent with the mortality schedule."""
    return sc.dataframe(ms.make_age_data())


def make_ode_sim(mosquito='eir_only', with_age_bins=False, **kw):
    """Helper: ODE human sim, optionally with age-structured demographics."""
    extra = {}
    if with_age_bins:
        extra['age_data']       = _ode_age_data()
        extra['age_bin_edges']  = AGE_BIN_EDGES
        extra['demographics']   = _ode_demographics()
    return make_sim(human='ode', mosquito=mosquito, **extra, **kw)


@sc.timer()
def test_ode_eir():
    """ODE human + EIR-only mosquitoes: runs and produces infections."""
    sc.heading('Testing ODE human + EIR-only...')
    sim = make_ode_sim(mosquito='eir_only')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.values.sum()
    assert n_infected > 0, f'Expected infections, got {n_infected}'


@sc.timer()
def test_ode_agents():
    """ODE human + agent-based mosquitoes: runs and produces infections."""
    sc.heading('Testing ODE human + agents...')
    # init_prev=0 is the production default; seed 1% here so mosquitoes
    # can pick up infections and bootstrap transmission.
    sim = make_ode_sim(mosquito='agents', human_pars=dict(init_prev=0.01))
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.values.sum()
    assert n_infected > 0, f'Expected infections, got {n_infected}'


@sc.timer()
def test_ode_ode():
    """ODE human + ODE mosquitoes: runs and produces infections."""
    sc.heading('Testing ODE human + ODE mosquitoes...')
    sim = make_ode_sim(mosquito='ode', human_pars=dict(init_prev=0.01))
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.values.sum()
    assert n_infected > 0, f'Expected infections, got {n_infected}'


@sc.timer()
def test_ode_human_results():
    """ODE human result arrays are populated, non-negative, and prevalence is in [0, 1]."""
    sc.heading('Testing ODE human result arrays...')
    sim = make_ode_sim(mosquito='eir_only')
    sim.run()

    res = sim.diseases[0].results
    assert (res.n_susceptible.values  >= 0).all(), 'Sh went negative'
    assert (res.n_infected.values     >= 0).all(), 'Ih went negative'
    assert (res.new_infections.values >= 0).all(), 'new_infections went negative'

    N = res.n_susceptible.values + res.n_infected.values
    prev = res.n_infected.values / N
    assert (prev >= 0).all() and (prev <= 1).all(), \
        f'Prevalence out of [0,1]: min={prev.min():.3f}, max={prev.max():.3f}'


@sc.timer()
def test_ode_population_stable():
    """With age-bin demographics, ODE total population stays within 1 % over 5 years."""
    sc.heading('Testing ODE population stability with demographics...')
    sim = make_ode_sim(mosquito='eir_only', with_age_bins=True,
                       stop=2005, nodes=ms.NodeGrid(populations=[n_agents]))
    sim.run()

    d = sim.diseases[0]
    N_init  = (d._Sh + d._Ih).sum()
    N_start = n_agents
    drift_pct = abs(N_init - N_start) / N_start * 100
    assert drift_pct < 1.0, \
        f'ODE population drifted {drift_pct:.2f}% over 5 years (threshold 1%)'


@sc.timer()
def test_ode_age_distribution_stable():
    """With age-bin demographics, per-bin populations stay within 10 % of initial over 5 years."""
    sc.heading('Testing ODE age-bin distribution stability...')
    import numpy as np
    sim = make_ode_sim(mosquito='eir_only', with_age_bins=True,
                       stop=2005, nodes=ms.NodeGrid(populations=[n_agents]))
    sim.run()

    d = sim.diseases[0]
    fracs      = np.array(AGE_BIN_FRACTIONS) / sum(AGE_BIN_FRACTIONS)
    init_bins  = n_agents * fracs
    n_bins     = d._n_bins
    final_bins = (d._Sh.raw + d._Ih.raw).reshape(-1, n_bins).sum(axis=0)

    meaningful = fracs >= 0.01
    drifts     = abs((final_bins[meaningful] - init_bins[meaningful]) / init_bins[meaningful]) * 100
    max_drift  = drifts.max()
    assert max_drift < 10.0, \
        f'Largest per-bin drift was {max_drift:.1f}% over 5 years (threshold 10%)'


@sc.timer()
def test_ode_higher_eir_more_infections():
    """ODE human: higher EIR produces more cumulative infections."""
    sc.heading('Testing ODE human: higher EIR → more infections...')
    lo_nodes = ms.NodeGrid(populations=[n_agents], eir=[[1]  * 12])
    hi_nodes = ms.NodeGrid(populations=[n_agents], eir=[[20] * 12])

    sim_lo = make_ode_sim(mosquito='eir_only', nodes=lo_nodes)
    sim_hi = make_ode_sim(mosquito='eir_only', nodes=hi_nodes)
    sim_lo.run()
    sim_hi.run()

    cum_lo = sim_lo.diseases[0].results.new_infections.values.sum()
    cum_hi = sim_hi.diseases[0].results.new_infections.values.sum()
    assert cum_hi > cum_lo, \
        f'Expected more infections with EIR=20 vs EIR=1, got {cum_hi} vs {cum_lo}'


@sc.timer()
def test_ode_higher_b_more_infections():
    """ODE human: higher prob_human_infected_on_bite (b) produces more infections."""
    sc.heading('Testing ODE human: higher b → more infections...')
    sim_lo = make_ode_sim(mosquito='eir_only', human_pars=dict(prob_human_infected_on_bite=0.1))
    sim_hi = make_ode_sim(mosquito='eir_only', human_pars=dict(prob_human_infected_on_bite=0.9))
    sim_lo.run()
    sim_hi.run()

    cum_lo = sim_lo.diseases[0].results.new_infections.values.sum()
    cum_hi = sim_hi.diseases[0].results.new_infections.values.sum()
    assert cum_hi > cum_lo, \
        f'Expected more infections with b=0.9 vs b=0.1, got {cum_hi} vs {cum_lo}'


@sc.timer()
def test_ode_multinode_human():
    """ODE human works with a multi-node grid; each node accumulates infections."""
    sc.heading('Testing ODE human with multi-node grid...')
    nodes = ms.NodeGrid(
        populations = [600, 400],
        eir         = [[10] * 12, [3] * 12],
    )
    sim = make_ode_sim(mosquito='eir_only', nodes=nodes)
    sim.run()

    d      = sim.diseases[0]
    Ih_mat = d._Ih.raw.reshape(2, d._n_bins)
    assert Ih_mat[0].sum() > 0, 'Node 0 had no infections'
    assert Ih_mat[1].sum() > 0, 'Node 1 had no infections'
    assert Ih_mat[0].sum() > Ih_mat[1].sum(), \
        'Expected higher-EIR node 0 to have more infections than node 1'


@sc.timer()
def test_ode_single_bin_backward_compat():
    """ODE human with no age_bin_edges runs in single-bin mode (backward compatible)."""
    sc.heading('Testing ODE human single-bin (no age structure)...')
    sim = make_ode_sim(mosquito='eir_only')
    sim.run()

    d = sim.diseases[0]
    assert d._n_bins == 1,               f'Expected 1 bin, got {d._n_bins}'
    assert d._Sh.raw.shape == (1,),      f'Expected shape (1,), got {d._Sh.raw.shape}'
    assert float(d._Ih.raw.sum()) > 0,  'Expected infections in single-bin mode'


@sc.timer()
def test_node_population_assignment():
    """Initial node counts match NodeGrid populations exactly.

    NodeGrid(populations=[10, 6, 4]) must put exactly 10 agents in node 0,
    6 in node 1, and 4 in node 2 on the first timestep.
    """
    import numpy as np
    sc.heading('Testing node population assignment...')

    pops  = [10, 6, 4]
    nodes = ms.NodeGrid(populations=pops)
    sim   = make_sim(human='sis', mosquito='eir_only', nodes=nodes, stop=2001)
    sim.init()
    sim.run(until=sim.t.start + sim.dt)   # one step so _malaria_pre_step_state fires

    disease  = sim.diseases[0]
    node_ids = np.asarray(disease.node_id[sim.people.auids], dtype=int)

    for nid, expected in enumerate(pops):
        count = int((node_ids == nid).sum())
        assert count == expected, (
            f'Node {nid}: expected exactly {expected} agents, got {count}'
        )


@sc.timer()
def test_no_cross_node_transmission_eir_only():
    """EIR does not leak across nodes (eir_only mosquitoes).

    Node 0 has EIR=10; node 1 has EIR=0.  After running for 90 days, every
    agent assigned to node 1 must remain susceptible — the non-zero EIR in
    node 0 must not reach them.
    """
    import numpy as np
    sc.heading('Testing no cross-node transmission (eir_only)...')

    nodes = ms.NodeGrid(
        populations = [500, 500],
        eir         = [[10] * 12, [0] * 12],
    )
    # init_prev=0: disable SIS's 1% initial seeding so node 1 truly starts clean.
    sim = make_sim(human='sis', mosquito='eir_only', nodes=nodes,
                   stop=2000 + 90/365,
                   human_pars=dict(init_prev=ss.bernoulli(p=0)))
    sim.run()

    disease  = sim.diseases[0]
    alive    = sim.people.auids
    nids     = np.asarray(disease.node_id[alive], dtype=int)
    node1    = alive[nids == 1]

    n_infected_node1 = int(np.asarray(disease.infected[node1], dtype=bool).sum())
    assert n_infected_node1 == 0, (
        f'Cross-node EIR leak detected: {n_infected_node1} agents in node 1 '
        f'became infected despite EIR=0 in node 1.'
    )

    n_infected_node0 = int(np.asarray(disease.infected[alive[nids == 0]], dtype=bool).sum())
    assert n_infected_node0 > 0, (
        f'Node 0 (EIR=10) had no infections — something else is wrong.'
    )


@sc.timer()
def test_no_cross_node_transmission_agents():
    """Mosquitoes and transmission are strictly node-local (agent-based mosquitoes).

    Two equal nodes start with the same mosquito emergence rate.  An Outbreak
    seeds 50% of node 1 at t=0; node 0 starts with zero infections.
    Because node 0 mosquitoes only bite node 0 humans (who are all susceptible),
    they never become infectious and therefore cannot transmit to node 0 humans.
    After 90 days, every agent in node 0 must still be susceptible.
    """
    import numpy as np
    sc.heading('Testing no cross-node transmission (agent mosquitoes)...')

    nodes = ms.NodeGrid(
        populations = [500, 500],
        emergence   = [[1.0] * 12, [1.0] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'seed_node1',
                products    = ms.Outbreak(),
                start       = 2000,
                eligibility = ms.HumanEligibility(node_ids=[1]),
                coverage    = 0.5,
            )
        ])

    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2000 + 90/365,
                   build_campaign=build_campaign)
    sim.run()

    disease   = sim.diseases[0]
    alive_end = sim.people.auids
    nids_end  = np.asarray(disease.node_id[alive_end], dtype=int)
    node0_end = alive_end[nids_end == 0]

    n_infected_node0 = int(np.asarray(disease.infected[node0_end], dtype=bool).sum())
    assert n_infected_node0 == 0, (
        f'Cross-node mosquito transmission detected: {n_infected_node0} agents '
        f'in node 0 became infected despite only node 1 being seeded.'
    )


@sc.timer()
def test_outbreak_seeds_node_no_cross_node_spread():
    """Outbreak infects node 0 on day 10; node 1 stays clean through day 30.

    Two equal nodes, 1000 agents each, no seasonality, agent-based mosquitoes.
    An Outbreak covers 100% of node 0 on day 10.  At day 30:
      - Node 0 prevalence must be close to 100% (20 days post-infection is well
        within the ~180-day SIS recovery window, so almost no one has recovered).
      - Node 1 must still have zero infections — node 0 mosquitoes cannot reach
        node 1 humans.
    """
    import numpy as np
    sc.heading('Testing Outbreak seeds node 0, node 1 stays clean...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [[1.0] * 12, [1.0] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'seed_node0',
                products    = ms.Outbreak(),
                start       = 2000 + 10/365,
                eligibility = ms.HumanEligibility(node_ids=[0]),
                coverage    = 1.0,
            )
        ])

    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2000 + 30/365,
                   demographics=[],
                   build_campaign=build_campaign)
    sim.run()

    disease  = sim.diseases[0]
    alive    = sim.people.auids
    nids     = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    n0_infected = int(np.asarray(disease.infected[node0], dtype=bool).sum())
    n1_infected = int(np.asarray(disease.infected[node1], dtype=bool).sum())

    prev_node0 = n0_infected / len(node0)
    assert prev_node0 >= 0.95, (
        f'Node 0 prevalence at day 30 was {prev_node0:.1%} — expected ≥95% '
        f'since agents were infected on day 10 and recovery takes ~180 days.'
    )
    assert n1_infected == 0, (
        f'Cross-node transmission detected: {n1_infected} agents in node 1 '
        f'became infected despite only node 0 being seeded.'
    )


@sc.timer()
def test_outbreak_seeds_both_nodes():
    """Outbreak on day 10 infects both nodes; both show ≥95% prevalence at day 30.

    Two equal nodes, 1000 agents each, no births/deaths, no seasonality,
    agent-based mosquitoes.  An Outbreak covers 100% of both nodes on day 10.
    At day 30 (20 days post-infection, well within the ~180-day SIS recovery
    window), almost no one has recovered so both nodes must be near 100%.
    """
    import numpy as np
    sc.heading('Testing Outbreak seeds both nodes...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [[1.0] * 12, [1.0] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed_both',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 1.0,
            )
        ])

    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2000 + 730/365,
                   demographics=[],
                   build_campaign=build_campaign)
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    prev_node0 = np.asarray(disease.infected[node0], dtype=bool).mean()
    prev_node1 = np.asarray(disease.infected[node1], dtype=bool).mean()

    assert prev_node0 >= 0.95, (
        f'Node 0 prevalence at day 730 was {prev_node0:.1%} — expected ≥95% '
        f'since both nodes were seeded on day 10 with sustained transmission.'
    )
    assert prev_node1 >= 0.95, (
        f'Node 1 prevalence at day 730 was {prev_node1:.1%} — expected ≥95% '
        f'since both nodes were seeded on day 10 with sustained transmission.'
    )


@sc.timer()
def test_vaccine_clears_infection_both_nodes():
    """100% permanent vaccine distributed on day 100 drives both nodes to zero by day 730.

    Two equal nodes, 1000 agents each, no births/deaths, no seasonality,
    agent-based mosquitoes.  An Outbreak seeds 100% of both nodes on day 10.
    A vaccine with 100% initial efficacy and infinite box duration is distributed
    to 100% of both nodes on day 100.  After vaccination, existing infections run
    their course (~180-day expected recovery) and no new infections can occur
    (prob_acquire = 0).  By day 730 every infection must have cleared.
    """
    import numpy as np
    sc.heading('Testing 100% vaccine clears both nodes by day 730...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [[1.0] * 12, [1.0] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed_both',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 1.0,
            ),
            ms.InterventionDistributorScheduled(
                name     = 'vaccinate_both',
                products = ms.MalariaVaccine(waning=ms.WaningEffect()),
                start    = 2000 + 100/365,
                coverage = 1.0,
            ),
        ])

    # Use a short fixed infectious period so all infections clear well before day 730.
    # The log-normal default (mean=200d, std=200d) has a right tail that can exceed
    # 2 years, which would cause spurious failures in the clearance assertion.
    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2000 + 730/365,
                   demographics=[],
                   human_pars=dict(infection_duration_years=ss.constant(v=30/365)),
                   build_campaign=build_campaign)
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    n0_infected = int(np.asarray(disease.infected[node0], dtype=bool).sum())
    n1_infected = int(np.asarray(disease.infected[node1], dtype=bool).sum())

    assert n0_infected == 0, (
        f'Node 0 still had {n0_infected} infections at day 730 despite 100% '
        f'permanent vaccine — all infections should have cleared by now.'
    )
    assert n1_infected == 0, (
        f'Node 1 still had {n1_infected} infections at day 730 despite 100% '
        f'permanent vaccine — all infections should have cleared by now.'
    )


@sc.timer()
def test_vaccine_node0_only_clears_node0_not_node1():
    """Vaccine on node 0 only: node 0 clears, node 1 stays endemic at day 730.

    Same setup as test_vaccine_clears_infection_both_nodes, but the vaccine is
    distributed only to node 0.  Node 0 must reach zero prevalence (vaccine
    blocks new infections and existing ones clear).  Node 1 is never vaccinated
    so transmission continues unimpeded — prevalence must remain near 100%.
    """
    import numpy as np
    sc.heading('Testing vaccine on node 0 only: node 0 clears, node 1 stays endemic...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [[1.0] * 12, [1.0] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'seed_both',
                products = ms.Outbreak(),
                start    = 2000 + 10/365,
                coverage = 1.0,
            ),
            ms.InterventionDistributorScheduled(
                name        = 'vaccinate_node0',
                products    = ms.MalariaVaccine(waning=ms.WaningEffect()),
                start       = 2000 + 100/365,
                eligibility = ms.HumanEligibility(node_ids=[0]),
                coverage    = 1.0,
            ),
        ])

    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2000 + 730/365,
                   demographics=[],
                   human_pars=dict(infection_duration_years=ss.constant(v=30/365)),
                   build_campaign=build_campaign)
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    n0_infected = int(np.asarray(disease.infected[node0], dtype=bool).sum())
    n1_infected = int(np.asarray(disease.infected[node1], dtype=bool).sum())

    assert n0_infected == 0, (
        f'Node 0 still had {n0_infected} infections at day 730 despite 100% '
        f'permanent vaccine — all infections should have cleared.'
    )

    prev_node1 = n1_infected / len(node1)
    assert prev_node1 >= 0.95, (
        f'Node 1 prevalence at day 730 was {prev_node1:.1%} — expected ≥95% '
        f'since node 1 was never vaccinated and has sustained transmission.'
    )


@sc.timer()
def test_vaccine_prob_acquire_applied_and_persists():
    """prob_acquire is 0 for vaccinated agents and 1 for unvaccinated after each step.

    Vaccination fires on day 100 for node 0 only (100% coverage, infinite box).
    The execution order each step is:
      1. step_state (_do_disease_transitions) resets prob_acquire = 1.0 for all
      2. MalariaVaccineUpdater sets prob_acquire = 0 for vaccinated agents
      3. malariahumansis.step reads prob_acquire for transmission
    At the end of every step after day 100, vaccinated agents (node 0) must have
    prob_acquire = 0 and unvaccinated agents (node 1) must have prob_acquire = 1.
    """
    import numpy as np
    sc.heading('Testing prob_acquire applied and persists after vaccination...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        emergence   = [[1.0] * 12, [1.0] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'vaccinate_node0',
                products    = ms.MalariaVaccine(waning=ms.WaningEffect()),
                start       = 2000 + 100/365,
                eligibility = ms.HumanEligibility(node_ids=[0]),
                coverage    = 1.0,
            )
        ])

    # Run a few steps past the vaccine day so MalariaVaccineUpdater has re-applied.
    # Use steady-state demographics so that unvaccinated newborns enter the
    # population — this prevents the test from trivially passing simply because
    # every alive agent was vaccinated.
    # Run to year 2002 so ~2 years of births produce a meaningful number of
    # unvaccinated newborns in node 0 alongside the vaccinated original cohort.
    sim = make_sim(human='sis', mosquito='agents', nodes=nodes,
                   stop=2002,
                   demographics=ms.make_steady_state_demographics(),
                   build_campaign=build_campaign)
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    # With demographics, not all node 0 agents are vaccinated — unvaccinated
    # newborns have prob_acquire=1.  The invariant is biconditional:
    #   vaccinated (vaccine_decay_end > 0)  ↔  prob_acquire == 0
    decay_end_all = np.asarray(disease.vaccine_decay_end[alive], dtype=float)
    has_vaccine   = decay_end_all > 0
    prob_all      = np.asarray(disease._prob_acquire_indoor[alive], dtype=float)
    is_protected  = prob_all == 0.0

    # Sanity: some node 0 agents must be vaccinated and some (newborns) must not be.
    nids_all = np.asarray(disease.node_id[alive], dtype=int)
    assert has_vaccine[nids_all == 0].any(), 'No vaccinated agents found in node 0.'
    assert (~has_vaccine[nids_all == 0]).any(), 'No unvaccinated agents in node 0 — newborns missing.'
    assert not has_vaccine[nids_all == 1].any(), 'Node 1 should have no vaccinated agents.'

    n_protected_no_vacc  = int((is_protected & ~has_vaccine).sum())
    n_vacc_not_protected = int((has_vaccine  & ~is_protected).sum())

    assert n_protected_no_vacc == 0, (
        f'{n_protected_no_vacc} agents have prob_acquire=0 but no vaccine record '
        f'(vaccine_decay_end <= 0).'
    )
    assert n_vacc_not_protected == 0, (
        f'{n_vacc_not_protected} agents have a vaccine record but prob_acquire != 0 '
        f'— MalariaVaccineUpdater is not applying protection correctly.'
    )


@sc.timer()
def test_prob_acquire_zero_blocks_transmission():
    """prob_acquire=0 on line 475 of human_base.py must multiply infection probability to zero.

    Both nodes receive identical high EIR.  Node 0 is vaccinated (prob_acquire=0)
    on day 1 before any transmission occurs.  After 30 days, node 0 must have zero
    infections (p * 0 = 0) while node 1 must have many (p * 1 = p).

    Uses EIR-only mosquitoes so EIR is deterministic and both nodes have identical
    exposure — any difference in infection counts is purely due to prob_acquire.
    """
    import numpy as np
    sc.heading('Testing prob_acquire=0 blocks all transmission...')

    nodes = ms.NodeGrid(
        populations = [1000, 1000],
        eir         = [[30] * 12, [30] * 12],
    )

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name        = 'vaccinate_node0',
                products    = ms.MalariaVaccine(waning=ms.WaningEffect()),
                start       = 2000,
                eligibility = ms.HumanEligibility(node_ids=[0]),
                coverage    = 1.0,
            )
        ])

    sim = make_sim(human='sis', mosquito='eir_only', nodes=nodes,
                   stop=2000 + 30/365,
                   demographics=[],
                   build_campaign=build_campaign)
    sim.run()

    disease = sim.diseases[0]
    alive   = sim.people.auids
    nids    = np.asarray(disease.node_id[alive], dtype=int)

    node0 = alive[nids == 0]
    node1 = alive[nids == 1]

    n0_infected = int(np.asarray(disease.infected[node0], dtype=bool).sum())
    n1_infected = int(np.asarray(disease.infected[node1], dtype=bool).sum())

    assert n0_infected == 0, (
        f'Node 0 had {n0_infected} infections despite prob_acquire=0 — '
        f'line 475 of human_base.py is not zeroing infection probability.'
    )
    assert n1_infected > 0, (
        f'Node 1 had no infections despite EIR=30 — something else is wrong.'
    )


@sc.timer()
def test_ode_vaccine_blocks_new_infections():
    """100% permanent vaccine on day 1 reduces ODE new infections to ~0.

    Single node, EIR=30 (deterministic, eir_only mosquitoes). init_prev=0.01
    seeds 1% initial infections. Vaccine fires on day 1 with 100% coverage.
    After vaccine, _prob_acquire[0] = 0 so new_inf = eir_dt * b * Sh * 0 = 0
    each subsequent step. Total new_infections accumulated after day 3 must
    be ~0 (allowing 3 steps for the vaccine-firing step itself).
    """
    import numpy as np
    sc.heading('Testing ODE 100% vaccine blocks all new infections...')

    nodes = ms.NodeGrid(populations=[1000], eir=[[30.0] * 12])

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'vaccinate_all',
                products = ms.MalariaVaccine(waning=ms.WaningEffect()),
                start    = 2000 + 1/365,
                coverage = 1.0,
            ),
        ])

    sim = make_ode_sim(nodes=nodes, stop=2000 + 30/365,
                       demographics=[],
                       build_campaign=build_campaign,
                       human_pars=dict(init_prev=0.01))
    sim.run()

    disease    = sim.diseases[0]
    new_inf    = disease.results.new_infections.values
    total_after = new_inf[3:].sum()

    assert total_after < 0.1, (
        f'Expected ~0 new infections after 100% ODE vaccine; got {total_after:.3f}'
    )


@sc.timer()
def test_ode_vaccine_reduces_infections_vs_no_vaccine():
    """ODE 80% vaccine reduces total new infections versus an unvaccinated run.

    Same single-node, EIR=30 sim run twice: once with no campaign and once
    with an 80%-coverage vaccine on day 30. The vaccinated run must accumulate
    fewer total new infections over the full year.
    """
    sc.heading('Testing ODE vaccine reduces infections vs no vaccine...')

    nodes = ms.NodeGrid(populations=[1000], eir=[[30.0] * 12])
    kw    = dict(nodes=nodes, stop=2001, demographics=[],
                 rand_seed=42, human_pars=dict(init_prev=0.01))

    sim_no_vax = make_ode_sim(**kw)
    sim_no_vax.run()

    def build_campaign(sim, eir_only=False):
        return ms.Campaign(sim, [
            ms.InterventionDistributorScheduled(
                name     = 'vaccinate_80pct',
                products = ms.MalariaVaccine(waning=ms.WaningEffect()),
                start    = 2000 + 30/365,
                coverage = 0.8,
            ),
        ])

    sim_vax = make_ode_sim(**kw, build_campaign=build_campaign)
    sim_vax.run()

    inf_no_vax = sim_no_vax.diseases[0].results.new_infections.values.sum()
    inf_vax    = sim_vax.diseases[0].results.new_infections.values.sum()

    assert inf_vax < inf_no_vax, (
        f'Expected vaccine to reduce infections: vax={inf_vax:.1f}, no_vax={inf_no_vax:.1f}'
    )


def test_performance():
    """Performance smoke test — not collected by pytest by default due to no assertion."""
    sc.heading('Testing performance ...')
    n_agents_perf = 10_000
    n_days        = 730
    start         = 2000
    stop          = start + n_days / 365
    nodes         = ms.NodeGrid(populations=[n_agents_perf])
    sim           = make_sim(human='sis', mosquito='eir_only',
                             nodes=nodes, start=start, stop=stop)
    sim.run()
    sc.printcyan(f'\nRun time for {n_agents_perf} agents and {n_days} days: {sim.elapsed:n} s')


if __name__ == '__main__':
    do_plot = False
    sc.options(interactive=do_plot)
    T = sc.timer()

    test_sis_eir()
    test_seis_eir()
    test_sis_agents()
    test_sis_ode()
    test_seis_ode()
    test_ode_mosquito_results()
    test_ode_multinode()
    test_ode_more_emergence_more_infections()
    test_higher_eir_more_infections()

    test_ode_eir()
    test_ode_agents()
    test_ode_ode()
    test_ode_human_results()
    test_ode_population_stable()
    test_ode_age_distribution_stable()
    test_ode_higher_eir_more_infections()
    test_ode_higher_b_more_infections()
    test_ode_multinode_human()
    test_ode_single_bin_backward_compat()
    test_ode_vaccine_blocks_new_infections()
    test_ode_vaccine_reduces_infections_vs_no_vaccine()
    test_node_population_assignment()
    test_no_cross_node_transmission_eir_only()
    test_no_cross_node_transmission_agents()

    test_performance()

    T.toc()

    if do_plot:
        plt.show()
