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
    sims = test_higher_eir_more_infections(do_plot=do_plot)
    sim4 = test_performance()

    T.toc()

    if do_plot:
        plt.show()
