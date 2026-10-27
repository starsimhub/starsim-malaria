"""
Bednet (ITN) tests: protection of the holder from indoor bites, and the blocking and killing of mosquitoes

The mosquitoes feel the nets through their coverage (`itn.coverage`): as counts, under every solver
when the nets wane by one box (`ss.box()`), and as agents, from the holders in each node.
"""

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import malariasim as ms

_, EMERGENCE_10 = ms.eir_to_params(10.0, common_pars=ms.CommonParams())


def bednet(repelling=0.0, blocking=1.0, killing=1.0, usage=1.0):
    """ A bednet whose effects last the year """
    return ms.Bednet(repelling=repelling, blocking=blocking, killing=killing, usage=usage, waning=ss.box(dur=ss.days(365)))


def make_sim(net=None, at='2000-01-31', coverage=1.0, where=None, seed=True, mosquito='ode', nodes=None, **kw):
    """ One node of 2,000 people, mosquitoes at an EIR of about 10, seeded on day 10, with a bednet on day 30 """
    def interventions(mal):
        out = [ss.Delivery(move=mal.S >> mal.I, at='2000-01-11', coverage=0.15, name='seed')] if seed else []
        if net is not None:
            out.append(ss.vaccinate(net, at=at, coverage=coverage, where=where, name='itn'))
        return out
    nodes = [2000] if nodes is None else nodes
    kw = sc.mergedicts(dict(emergence=EMERGENCE_10, start=2000, stop=2001, rand_seed=0, verbose=0), kw)
    return ms.Sim(mosquito=mosquito, nodes=nodes, interventions=interventions, **kw).run()


def infections(sim):
    return sim.results.malaria.infections.sum()


def mosquitoes(sim):
    res = sim.results.malaria if 'mosquitoes' in sim.results.malaria else sim.results.mosquitoes
    return np.mean(res.mosquitoes)


@sc.timer()
def test_bednet_reduces_infections_vs_baseline():
    """ A good net for 80% of people reduces infections over the year """
    base = make_sim(rand_seed=42)
    itn = make_sim(bednet(repelling=0.8, blocking=0.9, killing=0.6, usage=0.8), coverage=0.8, rand_seed=42)
    assert infections(itn) < infections(base), f'Expected the bednet to reduce infections: {infections(base):.0f} vs {infections(itn):.0f}'


@sc.timer()
def test_bednet_killing_reduces_mosquitoes():
    """ Blocking and killing for everyone kills mosquitoes: the extra death rate a*f*1 = 0.21 a day more than triples 0.1 """
    base = make_sim(seed=False, at='2000-01-01')
    kill = make_sim(bednet(), seed=False, at='2000-01-01')
    assert mosquitoes(kill) < 0.5*mosquitoes(base), f'Expected the nets to halve the mosquitoes: {mosquitoes(base):.0f} vs {mosquitoes(kill):.0f}'


@sc.timer()
def test_bednet_killing_scales_with_coverage():
    """ The nets of half the people kill substantially fewer mosquitoes than everyone's """
    base, half, full = [mosquitoes(make_sim(bednet() if cov else None, coverage=cov or 1.0, seed=False, at='2000-01-01')) for cov in [0, 0.5, 1.0]]
    assert full < half < base, f'Expected more killing at higher coverage: {base:.0f}, {half:.0f}, {full:.0f}'
    ratio = (base - half)/(base - full)
    assert 0.5 <= ratio <= 0.9, f'The effect at 50% coverage should be about 75% of that at 100%, not {ratio:.2f}'


@sc.timer()
def test_bednet_killing_scales_with_coverage_ode():
    """ As above, as an ODE """
    base, half, full = [mosquitoes(make_sim(bednet() if cov else None, coverage=cov or 1.0, seed=False, at='2000-01-01', solver='ode')) for cov in [0, 0.5, 1.0]]
    assert full < half < base, f'Expected more killing at higher coverage: {base:.0f}, {half:.0f}, {full:.0f}'
    ratio = (base - half)/(base - full)
    assert 0.5 <= ratio <= 0.9, f'The effect at 50% coverage should be about 75% of that at 100%, not {ratio:.2f}'


@sc.timer()
def test_bednet_killing_requires_blocking():
    """ With no blocking, a net neither protects nor kills """
    base = make_sim()
    null = make_sim(bednet(blocking=0.0))
    assert infections(null) >= 0.9*infections(base), f'A net with no blocking had an effect: {infections(base):.0f} vs {infections(null):.0f}'


@sc.timer()
def test_bednet_full_repelling_prevents_mosquito_killing():
    """ A mosquito that is repelled is not killed """
    base = make_sim(seed=False, at='2000-01-01')
    rep = make_sim(bednet(repelling=1.0), seed=False, at='2000-01-01')
    assert mosquitoes(rep) == pytest.approx(mosquitoes(base), rel=1e-6), f'Repelled mosquitoes were killed: {mosquitoes(base):.1f} vs {mosquitoes(rep):.1f}'


@sc.timer()
def test_bednet_repelling_scales_mosquito_killing():
    """ The killing is in proportion to the share not repelled """
    n = {rep:mosquitoes(make_sim(bednet(repelling=rep), seed=False, at='2000-01-01')) for rep in [1.0, 0.5, 0.0]}
    assert n[1.0] > n[0.5] > n[0.0], f'Expected more mosquitoes with more repelling: {n}'
    ratio = (n[1.0] - n[0.5])/(n[1.0] - n[0.0])
    assert 0.5 <= ratio <= 0.9, f'The kill effect at 50% repelling should be about 2/3 of that at 0%, not {ratio:.2f}'


@sc.timer()
def test_bednet_usage_zero_has_no_effect():
    """ A net that is not used does nothing """
    base = make_sim()
    unused = make_sim(bednet(repelling=1.0, usage=0.0))
    assert infections(unused) >= 0.9*infections(base), f'An unused net had an effect: {infections(base):.0f} vs {infections(unused):.0f}'


@sc.timer()
def test_bednet_node_targeted():
    """ Nets in node a only: lower prevalence there than in node b """
    nodes = ss.geo.geography(names=['a', 'b'], pop=[1000, 1000])
    net = bednet(repelling=0.8, blocking=0.9, killing=0.6, usage=0.8)
    sim = make_sim(net, where=ss.ppl.patch == 'a', nodes=nodes, emergence=[EMERGENCE_10, EMERGENCE_10])
    prev = sim.results.malaria.prevalence.by('patch')[-1]
    assert prev[0] < prev[1], f'Expected lower prevalence in node a, with nets: {prev}'


@pytest.mark.parametrize('mosquito', ['ode', 'agents'])
@sc.timer()
def test_bednet_outdoor_only_has_no_effect(mosquito):
    """ With every bite outdoors, a perfect net does nothing """
    base = make_sim(mosquito=mosquito, f_indoor=0.0)
    itn = make_sim(bednet(), mosquito=mosquito, f_indoor=0.0)
    assert infections(itn) >= 0.9*infections(base), f'A net had an effect with no indoor bites: {infections(base):.0f} vs {infections(itn):.0f}'


@pytest.mark.parametrize('mosquito', ['ode', 'agents'])
@sc.timer()
def test_bednet_indoor_only_full_effect(mosquito):
    """ With every bite indoors, a perfect net for everyone more than halves infections """
    base = make_sim(mosquito=mosquito, f_indoor=1.0)
    itn = make_sim(bednet(), mosquito=mosquito, f_indoor=1.0)
    assert infections(itn) < 0.5*infections(base), f'A perfect net should more than halve infections: {infections(base):.0f} vs {infections(itn):.0f}'


@pytest.mark.parametrize('mosquito', ['ode', 'agents'])
@sc.timer()
def test_bednet_f_indoor_scales_mosquito_killing(mosquito):
    """ Nets kill more mosquitoes where more bites are indoors """
    n = {f:mosquitoes(make_sim(bednet(), mosquito=mosquito, f_indoor=f, seed=False, at='2000-01-01')) for f in [0.0, 0.5, 1.0]}
    assert n[0.0] > n[0.5] > n[1.0], f'Expected fewer mosquitoes with more indoor biting: {n}'


@sc.timer()
def test_bednet_no_double_count():
    """ The EIR people receive is the mosquitoes' bites; only the holder's own net protects them (malariasim bug 9) """
    sim = make_sim(bednet(blocking=0.5, killing=0.0), coverage=0.5, f_indoor=1.0, seed=False)
    res = sim.results.malaria
    assert np.allclose(res.eir, 0.3*res.Iv/2000, rtol=1e-6, atol=1e-12), 'The EIR is not the mosquitoes\' bites per person'


def test_bednet_waning_by_effect():
    """ Each effect wanes by its own function; the holder's protection combines them, and wanes as one box only if every effect does """
    blocking, usage = ss.box(dur=ss.days(100)), ss.box(dur=ss.days(200), decay=ss.days(100))
    net = ms.Bednet(repelling=0.5, blocking=0.9, killing=0.6, usage=0.8, waning=dict(blocking=blocking, usage=usage))
    t = ss.days(np.array([0.0, 150, 300]))
    u = 0.8*usage(t)
    assert np.allclose(net.protection(t), 1 - (1 - 0.5*u)*(1 - 0.9*blocking(t)*u)), 'Expected the protection from the effects at each time'
    assert np.allclose(net.kill(t), (1 - 0.5*u)*0.9*blocking(t)*u*0.6*u), 'Expected the killing from the effects at each time'
    assert net.waning == net.relative_protection, 'Expected effects that wane differently to wane by the protection'
    assert bednet().waning == ss.box(dur=ss.days(365)), 'Expected one box for every effect to be the waning'
    with pytest.raises(ValueError):
        ms.Bednet(waning=dict(holes=blocking))
    with pytest.raises(ValueError):
        ms.Bednet(blocking=1.5)


if __name__ == '__main__':
    T = sc.timer()
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            if 'mosquito' in test.__code__.co_varnames[:1]:
                for mosquito in ['ode', 'agents']:
                    test(mosquito)
            else:
                test()
    T.toc()
