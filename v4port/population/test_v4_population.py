""" malariasim's population tests (tests/test_malaria.py), ported to the declared v4 model, plus the node shares of newborns """
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m

def sim_for(pops, eir=None, solver='abm', demog=None, years=1, age=None, seed=0):
    eir = np.tile(np.array([4.0]*len(pops))[:, None], 12) if eir is None else np.asarray(eir, float)
    mal = m.make_malaria(eir=eir, f_indoor=[0.7]*len(pops), labels=[f'node{i}' for i in range(len(pops))], pops=pops)
    return ss.Sim(mal, demographics=m.steady_state(demog) if demog else None, stratify=dict(age=list(age)) if age is not None else None, solver=solver,
                  people=ss.People(int(sum(pops)), age_data=sc.dataframe(m.age_data())), start=START, stop=START+years, dt=ss.days(DT_DAYS), rand_seed=seed, verbose=0)

def test_node_population_assignment():
    """ Initial node counts match the populations exactly (NodeGrid(populations=[10, 6, 4])) """
    sim = sim_for([10, 6, 4]).init()
    patch = np.asarray(sim.people.patch.values).astype(int)
    assert list(np.bincount(patch, minlength=3)) == [10, 6, 4]

def test_no_cross_node_transmission_eir_only():
    """ Node 0 has EIR 10, node 1 has EIR 0: nobody in node 1 is infected after 90 days """
    sim = sim_for([500, 500], eir=[[10]*12, [0]*12], years=90/365).run()
    I = sim.results.malaria.n_I.by('patch').values[-1]
    assert I[1] == 0 and I[0] > 0, I

def test_ode_population_stable():
    """ With age groups and births and deaths, the ODE population stays within 1% over 5 years """
    sim = sim_for([5000], solver='ode', demog='declared', years=5, age=AGE_BIN_EDGES[:-1]).run()
    N = sim.results.malaria.people.values
    assert abs(N[-1]/N[0] - 1) < 0.01, N[-1]/N[0]

def test_ode_age_distribution_stable():
    """ Every age group with at least 1% of the people stays within 10% of its start over 5 years """
    sim = sim_for([5000], solver='ode', demog='declared', years=5, age=AGE_BIN_EDGES[:-1]).run()
    a = sim.results.malaria.people.by('age').values
    keep = a[0]/a[0].sum() >= 0.01
    assert np.max(np.abs(a[-1][keep]/a[0][keep] - 1)) < 0.10

def test_ode_multinode_human():
    """ Each node accumulates infections, more in the node with the higher EIR """
    sim = sim_for([600, 400], eir=[[10]*12, [3]*12], solver='ode').run()
    I = sim.results.malaria.n_I.by('patch').values[-1]
    assert I[0] > 0 and I[1] > 0 and I[0] > I[1]

def test_newborns_keep_node_shares():
    """ New: with births and deaths, each node keeps its share of the people (malariasim assigns newborns by the initial shares) """
    for demog in ['declared', 'modules']:
        sim = sim_for([2000, 1500, 1500], demog=demog, years=10, age=AGE_BIN_EDGES[:-1]).run()
        n = sim.results.malaria.people.by('patch').values[-1]
        assert np.allclose(n/n.sum(), [0.4, 0.3, 0.3], atol=0.03), f'{demog}: node shares after 10 years {np.round(n/n.sum(), 3)}'

if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            try:
                f(); print(f'PASS {name}')
            except AssertionError as e:
                print(f'FAIL {name}: {e}')
