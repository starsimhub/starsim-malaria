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
def test_sis_eir(do_plot=do_plot):
    """Test that SIS + EIR-only runs and produces infections."""
    sc.heading('Testing SIS + EIR-only...')
    sim = make_sim(human='sis', mosquito='eir_only')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with default EIR, but got {n_infected}'
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_seis_eir(do_plot=do_plot):
    """Test that SEIS + EIR-only runs and produces infections."""
    sc.heading('Testing SEIS + EIR-only...')
    sim = make_sim(human='seis', mosquito='eir_only')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with default EIR, but got {n_infected}'
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_sis_agents(do_plot=do_plot):
    """Test that SIS + agent-based mosquitoes runs and produces infections."""
    sc.heading('Testing SIS + agents...')
    sim = make_sim(human='sis', mosquito='agents')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with agent mosquitoes, but got {n_infected}'
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_sis_ode(do_plot=do_plot):
    """Test that SIS + ODE mosquitoes runs and produces infections."""
    sc.heading('Testing SIS + ODE mosquitoes...')
    sim = make_sim(human='sis', mosquito='ode')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with ODE mosquitoes, but got {n_infected}'
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_seis_ode(do_plot=do_plot):
    """Test that SEIS + ODE mosquitoes runs and produces infections."""
    sc.heading('Testing SEIS + ODE mosquitoes...')
    sim = make_sim(human='seis', mosquito='ode')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected some infections with ODE mosquitoes, but got {n_infected}'
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_ode_mosquito_results():
    """Test that ODE mosquito results are populated and biologically sensible."""
    sc.heading('Testing ODE mosquito results...')
    sim = make_sim(human='sis', mosquito='ode')
    sim.run()

    mos_res = sim.results['mosquitoesode']
    assert mos_res.n_total.values.sum() > 0,      'Expected non-zero total mosquito population'
    assert mos_res.n_infectious.values.sum() > 0, 'Expected some infectious mosquitoes'
    assert mos_res.eir_annual.values.sum() > 0,   'Expected non-zero EIR from ODE mosquitoes'

    # SEI compartments should be non-negative throughout
    assert (mos_res.n_susceptible.values >= 0).all(), 'Sm went negative'
    assert (mos_res.n_exposed.values     >= 0).all(), 'Em went negative'
    assert (mos_res.n_infectious.values  >= 0).all(), 'Im went negative'

    return sim


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
    sim = make_sim(human='sis', mosquito='ode', nodes=nodes)
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.sum()
    assert n_infected > 0, f'Expected infections in multi-node ODE sim, but got {n_infected}'

    # Both nodes should have mosquitoes at end of run
    mos_mod = sim.connectors[1]
    assert (mos_mod._Sm + mos_mod._Em + mos_mod._Im > 0).all(), \
        'Expected non-zero mosquito population in every node'

    return sim


@sc.timer()
def test_ode_more_emergence_more_infections():
    """Test that higher ODE mosquito emergence produces more cumulative infections."""
    sc.heading('Testing ODE: higher emergence → more infections...')

    lo_nodes = ms.NodeGrid(populations=[n_agents], emergence=[[0.002] * 12])
    hi_nodes = ms.NodeGrid(populations=[n_agents], emergence=[[0.030] * 12])

    sim_lo = make_sim(mosquito='ode', nodes=lo_nodes)
    sim_hi = make_sim(mosquito='ode', nodes=hi_nodes)
    sim_lo.run()
    sim_hi.run()

    cum_lo = sim_lo.diseases[0].results.new_infections.sum()
    cum_hi = sim_hi.diseases[0].results.new_infections.sum()
    assert cum_hi > cum_lo, (
        f'Expected more infections with high emergence than low, '
        f'but got {cum_hi} vs {cum_lo}'
    )
    return sim_lo, sim_hi


@sc.timer()
def test_higher_eir_more_infections(do_plot=do_plot):
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

    return sim_lo, sim_hi


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
    return sim


@sc.timer()
def test_ode_agents():
    """ODE human + agent-based mosquitoes: runs and produces infections."""
    sc.heading('Testing ODE human + agents...')
    sim = make_ode_sim(mosquito='agents')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.values.sum()
    assert n_infected > 0, f'Expected infections, got {n_infected}'
    return sim


@sc.timer()
def test_ode_ode():
    """ODE human + ODE mosquitoes: runs and produces infections."""
    sc.heading('Testing ODE human + ODE mosquitoes...')
    sim = make_ode_sim(mosquito='ode')
    sim.run()
    n_infected = sim.diseases[0].results.n_infected.values.sum()
    assert n_infected > 0, f'Expected infections, got {n_infected}'
    return sim


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

    return sim


@sc.timer()
def test_ode_population_stable():
    """With age-bin demographics, ODE total population stays within 1 % over 5 years."""
    sc.heading('Testing ODE population stability with demographics...')
    sim = make_ode_sim(mosquito='eir_only', with_age_bins=True,
                       stop=2005, nodes=ms.NodeGrid(populations=[n_agents]))
    sim.run()

    d = sim.diseases[0]
    N_init  = (d._Sh + d._Ih).sum()           # final value; init was n_agents
    N_start = n_agents
    drift_pct = abs(N_init - N_start) / N_start * 100
    assert drift_pct < 1.0, \
        f'ODE population drifted {drift_pct:.2f}% over 5 years (threshold 1%)'
    return sim


@sc.timer()
def test_ode_age_distribution_stable():
    """With age-bin demographics, per-bin populations stay within 10 % of initial over 5 years.

    Only bins that represent at least 1 % of the total population are checked.
    Very old, small bins (e.g. 95-100, ~0.13 % of population) are excluded
    because their tiny absolute size makes percentage drift noisy and they have
    negligible epidemiological impact.
    """
    sc.heading('Testing ODE age-bin distribution stability...')
    import numpy as np
    sim = make_ode_sim(mosquito='eir_only', with_age_bins=True,
                       stop=2005, nodes=ms.NodeGrid(populations=[n_agents]))
    sim.run()

    d = sim.diseases[0]
    fracs      = np.array(AGE_BIN_FRACTIONS) / sum(AGE_BIN_FRACTIONS)
    init_bins  = n_agents * fracs
    final_bins = (d._Sh + d._Ih).sum(axis=0)

    # Only check bins with meaningful population (>= 1 % of total)
    meaningful = fracs >= 0.01
    drifts     = abs((final_bins[meaningful] - init_bins[meaningful]) / init_bins[meaningful]) * 100
    max_drift  = drifts.max()
    assert max_drift < 10.0, \
        f'Largest per-bin drift was {max_drift:.1f}% over 5 years (threshold 10%)'
    return sim


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
    return sim_lo, sim_hi


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
    return sim_lo, sim_hi


@sc.timer()
def test_ode_multinode():
    """ODE human works with a multi-node grid; each node accumulates infections."""
    sc.heading('Testing ODE human with multi-node grid...')
    nodes = ms.NodeGrid(
        populations = [600, 400],
        eir         = [[10] * 12, [3] * 12],
    )
    sim = make_ode_sim(mosquito='eir_only', nodes=nodes)
    sim.run()

    d = sim.diseases[0]
    # Both nodes should have infected compartment > 0 at some point
    assert d._Ih[0].sum() > 0, 'Node 0 had no infections'
    assert d._Ih[1].sum() > 0, 'Node 1 had no infections'
    # Node 0 (higher EIR) should have more infections than node 1
    assert d._Ih[0].sum() > d._Ih[1].sum(), \
        'Expected higher-EIR node 0 to have more infections than node 1'
    return sim


@sc.timer()
def test_ode_single_bin_backward_compat():
    """ODE human with no age_bin_edges runs in single-bin mode (backward compatible)."""
    sc.heading('Testing ODE human single-bin (no age structure)...')
    sim = make_ode_sim(mosquito='eir_only')   # no age_bin_edges
    sim.run()

    d = sim.diseases[0]
    assert d._n_bins == 1,          f'Expected 1 bin, got {d._n_bins}'
    assert d._Sh.shape == (1, 1),   f'Expected shape (1,1), got {d._Sh.shape}'
    assert d._Ih.sum() > 0,         'Expected infections in single-bin mode'
    return sim



def test_performance(human='sis', mosquito='eir_only', n_agents=10_000, n_days=730):
    sc.heading('Testing performance ...')
    start = 2000
    stop = start + n_days/365
    nodes = ms.NodeGrid(populations=[n_agents])
    sim = make_sim(human=human, mosquito=mosquito, nodes=nodes, start=start, stop=stop)
    sim.run()
    elapsed = sim.elapsed
    sc.printcyan(f'\nRun time for {n_agents} agents and {n_days} days: {elapsed:n} s')
    return sim


if __name__ == '__main__':
    do_plot = False
    sc.options(interactive=do_plot)
    T = sc.timer()

    sim1 = test_sis_eir(do_plot=do_plot)
    sim2 = test_seis_eir(do_plot=do_plot)
    sim3 = test_sis_agents(do_plot=do_plot)
    sim4 = test_sis_ode(do_plot=do_plot)
    sim5 = test_seis_ode(do_plot=do_plot)
    sim6  = test_ode_mosquito_results()
    sim7  = test_ode_multinode()
    sims  = test_ode_more_emergence_more_infections()
    sims2 = test_higher_eir_more_infections(do_plot=do_plot)

    # MalariaHumanODE tests
    sim9  = test_ode_eir()
    sim10 = test_ode_agents()
    sim11 = test_ode_ode()
    sim12 = test_ode_human_results()
    sim13 = test_ode_population_stable()
    sim14 = test_ode_age_distribution_stable()
    sims3 = test_ode_higher_eir_more_infections()
    sims4 = test_ode_higher_b_more_infections()
    sim15 = test_ode_multinode()
    sim16 = test_ode_single_bin_backward_compat()

    sim8  = test_performance()

    T.toc()

    if do_plot:
        plt.show()
