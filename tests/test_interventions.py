"""
Intervention tests: outbreaks, drugs, vaccines, and how their protection wanes

The interventions are Starsim's deliveries: an outbreak is `ss.Delivery(move=mal.S >> mal.I)`, a drug
`ss.treat(ss.Treatment(cure=dict(I=1.0)))`, and a vaccine `ss.vaccinate(ss.Vaccine(...))`, waning as a box then a decay (`ss.box()`).
"""

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import malariasim as ms

_, EMERGENCE_10 = ms.eir_to_params(10.0, common_pars=ms.CommonParams(bites_per_female_per_day=0.3, prob_human_infected_on_bite=1.0, prob_mosquito_infected_on_bite=1.0)) # The model's defaults


def make_sim(interventions=None, nodes=None, **kw):
    kw = sc.mergedicts(dict(mosquito='ode', emergence=EMERGENCE_10, start=2000, stop=2001, rand_seed=0, verbose=0), kw)
    return ms.Sim(nodes=[2000] if nodes is None else nodes, interventions=interventions, **kw).run()


def seed(at='2000-01-11', coverage=0.15, where=None):
    return lambda mal: ss.Delivery(move=mal.S >> mal.I, at=at, coverage=coverage, where=where, name='seed')


def drug(at='2000-07-01', coverage=1.0, where=None):
    return ss.treat(ss.Treatment(cure=dict(I=1.0)), at=at, coverage=coverage, where=where, name='drug')


# Outbreaks

@pytest.mark.parametrize('solver', ['abm', 'ode'])
@sc.timer()
def test_outbreak_seeds_infections(solver):
    """ With mosquitoes uninfected, nothing happens unless an outbreak seeds infections """
    none = make_sim(solver=solver)
    seeded = make_sim(seed(), solver=solver)
    assert none.results.malaria.I.sum() == 0, 'Expected no infections without seeding'
    assert seeded.results.malaria.I[11] >= 0.1*2000, f'Expected about 15% infected the day after seeding, not {seeded.results.malaria.I[11]:.0f}'
    assert seeded.results.malaria.I[-1] > seeded.results.malaria.I[11], 'Expected transmission after seeding'


# Drugs

@pytest.mark.parametrize('solver', ['abm', 'ode'])
@sc.timer()
def test_drug_clears_infections(solver):
    """ A drug for everyone clears the infected """
    sim = make_sim(lambda mal: [seed()(mal), drug()], solver=solver)
    I = sim.results.malaria.I
    before, after = I[181], I[183] # The drug is given on day 182 (1 July)
    assert after < 0.1*before, f'Expected the drug to clear the infected: {before:.0f} before, {after:.0f} after'


@sc.timer()
def test_drug_half_coverage_clears_half():
    """ A drug for half the people clears about half the infected (as agents and as an ODE) """
    for solver in ['abm', 'ode']:
        sim = make_sim(lambda mal: [seed()(mal), drug(coverage=0.5)], solver=solver)
        I = np.asarray(sim.results.malaria.I)
        cleared = 1 - I[183]/I[181]
        assert 0.35 < cleared < 0.65, f'[{solver}] Expected about half the infected cleared, not {cleared:.2f}'


@sc.timer()
def test_drug_treats_infectious_only():
    """ In the SEIS model, the drug clears the infectious, not those still incubating """
    sim = make_sim(lambda mal: [seed()(mal), drug()], human='seis')
    res = sim.results.malaria
    assert res.I[183] < 0.1*res.I[181], 'Expected the drug to clear the infectious'
    assert res.E[182] > 0, 'Expected the exposed to be left'


@sc.timer()
def test_drug_node_targeted():
    """ A drug in node a only clears infections there, not in node b """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    sim = make_sim(lambda mal: [seed()(mal), drug(where=ss.ppl.patch == 'a')], nodes=nodes, emergence=[EMERGENCE_10, EMERGENCE_10])
    I = sim.results.malaria.I.by('patch')
    assert I[183, 0] < 0.1*I[181, 0], f'Expected node a cleared: {I[181, 0]:.0f} to {I[183, 0]:.0f}'
    assert I[183, 1] > 0.8*I[181, 1], f'Expected node b untouched: {I[181, 1]:.0f} to {I[183, 1]:.0f}'


# Vaccines

def rtss():
    """ An RTS,S-like vaccine: 80% for 180 days, then declining with a mean of 730 days """
    return ss.Vaccine(efficacy=0.8, waning=ss.box(dur=ss.days(180), decay=ss.days(730)))


@sc.timer()
def test_vaccine_reduces_infections_vs_baseline():
    """ An RTS,S-like vaccine for 80% of people reduces infections """
    vx = rtss()
    base = make_sim(seed(), rand_seed=42)
    vax = make_sim(lambda mal: [seed()(mal), ss.vaccinate(vx, at='2000-01-31', coverage=0.8)], rand_seed=42)
    assert vax.results.malaria.infections.sum() < base.results.malaria.infections.sum(), 'Expected fewer infections with the vaccine'


@sc.timer()
def test_vaccine_second_round_redoses():
    """ A second round reaches those vaccinated in the first, and restarts their protection (malariasim's rule, one dose at a time) """
    vx = ss.Vaccine(efficacy=1.0, waning=ss.box(dur=ss.days(30)))
    rounds = ss.vaccinate(vx, at=['2000-02-01', '2000-06-01'], coverage=0.8, name='rounds')
    sim = make_sim(lambda mal: [seed()(mal), rounds])
    delivered = np.asarray(sim.results.rounds.delivered)
    first, second = delivered[delivered > 0][:2]
    assert second > 0.9*first, f'Expected the second round to reach about as many as the first: {first:.0f} then {second:.0f}'


@sc.timer()
def test_vaccine_waning_ode():
    """ The same vaccine as an ODE (the box as levels of holding it) reduces infections """
    base = make_sim(seed(), solver='ode').results.malaria.infections.sum()
    vax = make_sim(lambda mal: [seed()(mal), ss.vaccinate(rtss(), at='2000-01-31', coverage=0.8)], solver='ode').results.malaria.infections.sum()
    assert vax < base, f'Expected fewer infections with the vaccine: {vax:.0f} vs {base:.0f}'


@sc.timer()
def test_vaccine_ode_permanent():
    """ A permanent vaccine (no waning) runs as an ODE, and reduces infections """
    vx = ss.Vaccine(efficacy=0.8)
    base = make_sim(seed(), solver='ode').results.malaria.infections.sum()
    vax = make_sim(lambda mal: [seed()(mal), ss.vaccinate(vx, at='2000-01-31', coverage=0.8)], solver='ode').results.malaria.infections.sum()
    assert vax < base, f'Expected fewer infections with the vaccine: {vax:.0f} vs {base:.0f}'


if __name__ == '__main__':
    T = sc.timer()
    test_outbreak_seeds_infections('abm')
    test_drug_clears_infections('abm')
    test_drug_half_coverage_clears_half()
    test_drug_treats_infectious_only()
    test_drug_node_targeted()
    test_vaccine_reduces_infections_vs_baseline()
    test_vaccine_second_round_redoses()
    test_vaccine_waning_ode()
    test_vaccine_ode_permanent()
    T.toc()
