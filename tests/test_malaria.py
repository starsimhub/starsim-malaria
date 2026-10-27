"""
Test malaria simulation configurations: the human and mosquito models, both solvers, nodes, and populations
"""

import numpy as np
import sciris as sc
import starsim as ss
import malariasim as ms

n_agents = 1_000
sc.options(interactive=False)


def make_sim(human='sis', mosquito='eir_only', **kw):
    kw = sc.mergedicts(dict(start=2000, stop=2001, nodes=[n_agents], dt=ss.days(1), rand_seed=0, verbose=0), kw)
    return ms.Sim(human=human, mosquito=mosquito, **kw)


def seed(at='2000-01-01', coverage=1.0, where=None):
    """ An outbreak: a share of the susceptible infected at a time """
    return lambda mal: ss.Delivery(move=mal.S >> mal.I, at=at, coverage=coverage, where=where, name='seed')


def infected_by_node(sim):
    """ The infectious people in each node at the end """
    return sim.results.malaria.I.by('patch')[-1]


@sc.timer()
def test_sis_eir():
    sim = make_sim(human='sis', mosquito='eir_only').run()
    assert sim.results.malaria.I.sum() > 0, 'Expected some infections with the default EIR'


@sc.timer()
def test_seis_eir():
    sim = make_sim(human='seis', mosquito='eir_only').run()
    res = sim.results.malaria
    assert res.I.sum() > 0 and res.E.sum() > 0, 'Expected some exposed and infectious people with the default EIR'


@sc.timer()
def test_sis_agents():
    sim = make_sim(human='sis', mosquito='agents', init_prev=0.01).run()
    assert sim.results.malaria.I.sum() > 0, 'Expected some infections with agent mosquitoes'
    assert sim.results.mosquitoes.Iv.sum() > 0, 'Expected some infectious mosquitoes'


@sc.timer()
def test_sis_ode():
    sim = make_sim(human='sis', mosquito='ode', init_prev=0.01).run()
    assert sim.results.malaria.I.sum() > 0, 'Expected some infections with ODE mosquitoes'


@sc.timer()
def test_seis_ode():
    sim = make_sim(human='seis', mosquito='ode', init_prev=0.01).run()
    assert sim.results.malaria.I.sum() > 0, 'Expected some infections with ODE mosquitoes'


@sc.timer()
def test_ode_mosquito_results():
    sim = make_sim(human='sis', mosquito='ode', init_prev=0.01).run()
    res = sim.results.malaria
    assert res.mosquitoes.sum() > 0, 'Expected a mosquito population'
    assert res.Iv.sum() > 0, 'Expected some infectious mosquitoes'
    assert res.eir.sum() > 0, 'Expected a non-zero EIR from the mosquitoes'
    for key in ['Sv', 'Ev', 'Iv']:
        assert (res[key] >= 0).all(), f'{key} went negative'


@sc.timer()
def test_ode_multinode():
    nodes = ss.geo.geography(names=['Urban', 'Rural'], pop=[600, 400])
    sim = make_sim(mosquito='ode', nodes=nodes, init_prev=0.01, emergence=[ms.URBAN_EMERGENCE, ms.RURAL_EMERGENCE]).run()
    res = sim.results.malaria
    assert res.I.sum() > 0, 'Expected infections in a multi-node sim'
    assert (res.mosquitoes.by('patch')[-1] > 0).all(), 'Expected mosquitoes in every node'


@sc.timer()
def test_ode_more_emergence_more_infections():
    sim_lo = make_sim(mosquito='ode', emergence=[0.002]*12, init_prev=0.01).run()
    sim_hi = make_sim(mosquito='ode', emergence=[0.030]*12, init_prev=0.01).run()
    lo, hi = sim_lo.results.malaria.infections.sum(), sim_hi.results.malaria.infections.sum()
    assert hi > lo, f'Expected more infections with high emergence than low, but got {hi} vs {lo}'


@sc.timer()
def test_higher_eir_more_infections():
    sim_lo = make_sim(eir=1).run()
    sim_hi = make_sim(eir=20).run()
    lo, hi = sim_lo.results.malaria.infections.sum(), sim_hi.results.malaria.infections.sum()
    assert hi > lo, f'Expected more infections with EIR=20 than EIR=1, but got {hi} vs {lo}'


# The same models as an ODE (solver='ode'), with age groups and steady-state demography

def make_ode_sim(mosquito='eir_only', with_age_bins=False, **kw):
    if with_age_bins:
        kw.update(age_data=ms.make_age_data(), age_bin_edges=ms.AGE_BIN_EDGES, demographics=ms.make_steady_state_demographics())
    return make_sim(mosquito=mosquito, solver='ode', **kw)


@sc.timer()
def test_ode_eir():
    sim = make_ode_sim(mosquito='eir_only').run()
    assert sim.results.malaria.I.sum() > 0, 'Expected infections'


@sc.timer()
def test_ode_ode():
    sim = make_ode_sim(mosquito='ode', init_prev=0.01).run()
    assert sim.results.malaria.I.sum() > 0, 'Expected infections'


@sc.timer()
def test_ode_seis():
    sim = make_ode_sim(mosquito='ode', human='seis', init_prev=0.01).run()
    assert sim.results.malaria.I.sum() > 0, 'Expected infections'


@sc.timer()
def test_ode_human_results():
    sim = make_ode_sim(mosquito='eir_only').run()
    res = sim.results.malaria
    for key in ['S', 'I', 'infections']:
        assert (res[key] >= 0).all(), f'{key} went negative'
    assert ((res.prevalence >= 0) & (res.prevalence <= 1)).all(), 'Prevalence out of [0, 1]'


@sc.timer()
def test_ode_population_stable():
    sim = make_ode_sim(with_age_bins=True, stop=2005).run()
    res = sim.results.malaria
    people = res.S + res.I
    drift = abs(people[-1] - n_agents)/n_agents*100
    assert drift < 1.0, f'ODE population drifted {drift:.2f}% over 5 years (threshold 1%)'


@sc.timer()
def test_ode_age_distribution_stable():
    sim = make_ode_sim(with_age_bins=True, stop=2005).run()
    res = sim.results.malaria
    ages = res.S.by('age') + res.I.by('age')
    fracs = np.array(ms.AGE_BIN_FRACTIONS)/sum(ms.AGE_BIN_FRACTIONS)
    meaningful = fracs >= 0.01
    drifts = abs(ages[-1][meaningful] - ages[0][meaningful])/ages[0][meaningful]*100
    assert drifts.max() < 10.0, f'Largest drift of an age group was {drifts.max():.1f}% over 5 years (threshold 10%)'


@sc.timer()
def test_ode_higher_eir_more_infections():
    lo = make_ode_sim(eir=1).run().results.malaria.infections.sum()
    hi = make_ode_sim(eir=20).run().results.malaria.infections.sum()
    assert hi > lo, f'Expected more infections with EIR=20 than EIR=1, got {hi} vs {lo}'


@sc.timer()
def test_ode_higher_b_more_infections():
    lo = make_ode_sim(prob_human_infected_on_bite=0.1).run().results.malaria.infections.sum()
    hi = make_ode_sim(prob_human_infected_on_bite=0.9).run().results.malaria.infections.sum()
    assert hi > lo, f'Expected more infections with b=0.9 than b=0.1, got {hi} vs {lo}'


@sc.timer()
def test_ode_multinode_human():
    nodes = ss.geo.geography(names=['a', 'b'], pop=[600, 400])
    sim = make_ode_sim(nodes=nodes, eir=[10, 3]).run()
    inf = sim.results.malaria.I.by('patch').sum(axis=0)
    assert (inf > 0).all(), 'Each node should have infections'
    assert inf[0]/600 > inf[1]/400, 'Expected the higher-EIR node 0 to have a higher prevalence'


@sc.timer()
def test_agents_and_ode_agree():
    """ One declared model, as agents and as an ODE: prevalence after a year within 0.03 (with an exponential infectious period, which is what the ODE takes) """
    kw = dict(mosquito='ode', init_prev=0.1, nodes=[5000], infection_duration=ss.expon(ss.days(200)))
    prev = [make_sim(solver=solver, **kw).run().results.malaria.prevalence[-1] for solver in ['abm', 'ode']]
    assert abs(prev[0] - prev[1]) < 0.03, f'Agents and ODE disagree: {prev}'


# Nodes

@sc.timer()
def test_node_population_assignment():
    """ People start in each node in proportion to its population """
    pops = [500, 300, 200]
    nodes = ss.geo.geography(names=['a', 'b', 'c'], pop=pops)
    sim = make_sim(nodes=nodes)
    sim.init()
    counts = np.bincount(np.asarray(sim.people.patch[sim.people.auids], dtype=int), minlength=3)
    assert np.allclose(counts, pops, atol=50), f'Expected about {pops} people in the nodes, got {counts}'


@sc.timer()
def test_node_population_kept_with_demography():
    """ Newborns join the nodes in proportion, so the nodes keep their shares """
    pops = [500, 300, 200]
    nodes = ss.geo.geography(names=['a', 'b', 'c'], pop=pops)
    sim = make_sim(nodes=nodes, stop=2005, age_data=ms.make_age_data(), demographics=ms.make_steady_state_demographics()).run()
    counts = np.bincount(np.asarray(sim.people.patch[sim.people.auids], dtype=int), minlength=3)
    assert np.allclose(counts/counts.sum(), np.array(pops)/sum(pops), atol=0.05), f'Node shares drifted: {counts}'


@sc.timer()
def test_no_cross_node_transmission_eir_only():
    """ The EIR of one node does not reach people in another """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[500, 500])
    sim = make_sim(nodes=nodes, eir=[10, 0], stop=2000 + 90/365).run()
    inf = sim.results.malaria.infections.by('patch').sum(axis=0)
    assert inf[1] == 0, f'Cross-node EIR leak: {inf[1]} infections in node b, whose EIR is 0'
    assert inf[0] > 0, 'Node a (EIR=10) had no infections'


@sc.timer()
def test_no_cross_node_transmission_agents():
    """ Mosquitoes and transmission are node-local (mosquito agents): seeding node b leaves node a uninfected """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[500, 500])
    sim = make_sim(mosquito='agents', nodes=nodes, emergence=[[1.0]*12, [1.0]*12], stop=2000 + 90/365,
                   interventions=seed(coverage=0.5, where=ss.ppl.patch == 'b')).run()
    inf = sim.results.malaria.I.by('patch').sum(axis=0)
    assert inf[0] == 0, f'Cross-node transmission: {inf[0]} person-days infected in node a'
    assert inf[1] > 0, 'Node b was not seeded'


@sc.timer()
def test_no_cross_node_transmission_ode():
    """ The same with mosquitoes as counts """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[500, 500])
    sim = make_sim(mosquito='ode', nodes=nodes, emergence=[[1.0]*12, [1.0]*12], stop=2000 + 90/365,
                   interventions=seed(coverage=0.5, where=ss.ppl.patch == 'b')).run()
    inf = sim.results.malaria.I.by('patch').sum(axis=0)
    assert inf[0] == 0, f'Cross-node transmission: {inf[0]} person-days infected in node a'


@sc.timer()
def test_outbreak_seeds_node_no_cross_node_spread():
    """ An outbreak in node a on day 10: nearly all of node a is infected at day 30, and none of node b """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    sim = make_sim(mosquito='agents', nodes=nodes, emergence=[[1.0]*12, [1.0]*12], stop=2000 + 30/365,
                   interventions=seed(at='2000-01-11', where=ss.ppl.patch == 'a')).run()
    prev = sim.results.malaria.prevalence.by('patch')[-1]
    assert prev[0] >= 0.95, f'Node a prevalence at day 30 was {prev[0]:.1%}, expected >= 95%'
    assert prev[1] == 0, f'Cross-node transmission: node b prevalence {prev[1]:.1%}'


@sc.timer()
def test_outbreak_seeds_both_nodes():
    """ An outbreak in both nodes, with sustained transmission: both near 100% at day 730 """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    sim = make_sim(mosquito='agents', nodes=nodes, emergence=[[1.0]*12, [1.0]*12], stop=2002,
                   interventions=seed(at='2000-01-11')).run()
    prev = sim.results.malaria.prevalence.by('patch')[-1]
    assert (prev >= 0.95).all(), f'Prevalence at day 730 was {prev}, expected >= 95% in both nodes'


# Vaccines and nodes

def vaccinate(where=None, at='2000-04-10', coverage=1.0):
    return ss.vaccinate(ss.Vaccine(efficacy=1.0), at=at, coverage=coverage, where=where)


@sc.timer()
def test_vaccine_clears_infection_both_nodes():
    """ A permanent vaccine for everyone on day 100: no new infections, so every infection has cleared by day 730 """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    sim = make_sim(mosquito='agents', nodes=nodes, emergence=[[1.0]*12, [1.0]*12], stop=2002, infection_duration=ss.constant(ss.days(30)),
                   interventions=lambda mal: [seed(at='2000-01-11')(mal), vaccinate()]).run()
    inf = infected_by_node(sim)
    assert (inf == 0).all(), f'Infections left at day 730 despite a permanent vaccine: {inf}'


@sc.timer()
def test_vaccine_node0_only_clears_node0_not_node1():
    """ The vaccine in node a only: node a clears, node b stays endemic """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    sim = make_sim(mosquito='agents', nodes=nodes, emergence=[[1.0]*12, [1.0]*12], stop=2002, infection_duration=ss.constant(ss.days(30)),
                   interventions=lambda mal: [seed(at='2000-01-11')(mal), vaccinate(where=ss.ppl.patch == 'a')]).run()
    prev = sim.results.malaria.prevalence.by('patch')[-1]
    assert prev[0] == 0, f'Node a still had prevalence {prev[0]:.1%} despite a permanent vaccine'
    assert prev[1] >= 0.95, f'Node b prevalence was {prev[1]:.1%}, expected >= 95% without the vaccine'


@sc.timer()
def test_vaccine_held_by_those_vaccinated():
    """ Only people in node a who were alive on day 100 hold the vaccine, and none of them is infected afterwards """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    vx = ss.Vaccine(efficacy=1.0)
    sim = make_sim(mosquito='eir_only', nodes=nodes, eir=[30, 30], stop=2002, age_data=ms.make_age_data(), demographics=ms.make_steady_state_demographics(),
                   infection_duration=ss.constant(ss.days(30)), interventions=ss.vaccinate(vx, at='2000-04-10', coverage=1.0, where=ss.ppl.patch == 'a')).run()
    people = sim.people
    uids = people.auids
    holds = np.asarray(people[vx.dim][uids], dtype=bool)
    patch = np.asarray(people.patch[uids], dtype=int)
    assert holds[patch == 0].any(), 'No vaccinated people in node a'
    assert (~holds[patch == 0]).any(), 'Everyone in node a holds the vaccine: newborns should not'
    assert not holds[patch == 1].any(), 'Node b should have no vaccinated people'
    infected = np.asarray(sim.diseases.malaria.I.raw[uids], dtype=bool)
    assert not (holds & infected).any(), 'Someone with a permanent vaccine of full efficacy was infected after it'


@sc.timer()
def test_full_protection_blocks_transmission():
    """ A vaccine of full efficacy at the start: no infections in node a, many in node b, at the same EIR """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    sim = make_sim(nodes=nodes, eir=[30, 30], stop=2000 + 30/365, interventions=vaccinate(where=ss.ppl.patch == 'a', at='2000-01-01')).run()
    inf = sim.results.malaria.infections.by('patch').sum(axis=0)
    assert inf[0] == 0, f'{inf[0]} infections in node a despite full protection'
    assert inf[1] > 0, 'No infections in node b'


@sc.timer()
def test_ode_vaccine_blocks_new_infections():
    """ As an ODE: a permanent vaccine for everyone on day 1 stops new infections """
    sim = make_ode_sim(eir=30, stop=2000 + 30/365, init_prev=0.01, interventions=vaccinate(at='2000-01-02')).run()
    after = sim.results.malaria.infections[3:].sum()
    assert after < 0.1, f'Expected no new infections after the vaccine, got {after:.3f}'


@sc.timer()
def test_ode_vaccine_reduces_infections_vs_no_vaccine():
    """ As an ODE: a vaccine for 80% on day 30 gives fewer infections over the year """
    kw = dict(eir=30, init_prev=0.01)
    no_vax = make_ode_sim(**kw).run().results.malaria.infections.sum()
    vax = make_ode_sim(**kw, interventions=vaccinate(at='2000-01-31', coverage=0.8)).run().results.malaria.infections.sum()
    assert vax < no_vax, f'Expected fewer infections with the vaccine: {vax:.0f} vs {no_vax:.0f}'


@sc.timer()
def test_performance():
    """ A typical run: 10,000 agents for two years, daily """
    T = sc.timer()
    make_sim(nodes=[10_000], stop=2002).run()
    sc.printcyan(f'Run time for 10,000 agents and 730 days: {T.toc(output=True):n} s')


if __name__ == '__main__':
    T = sc.timer()
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            test()
    T.toc()
