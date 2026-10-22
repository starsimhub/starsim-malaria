""" Probe: results by age and patch from stratify(): two-way selection, a declared ratio by patch, flows by patch, an EIR-like result """
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m
for solver in ['abm', 'ode']:
    sim = m.make_sim(seed=1, solver=solver, demog='declared', years=3)
    sim.run()
    res = sim.results.malaria
    S_p, I_p = res.n_S.by('patch').values, res.n_I.by('patch').values
    want = I_p/(S_p + I_p)
    got = res.prevalence.by('patch').values
    tot_want = res.n_I.values/(res.n_S.values + res.n_I.values)
    print(f'{solver}: prevalence.by(patch) at the end {np.round(got[-1], 4)}, from the counts {np.round(want[-1], 4)}; total {res.prevalence.values[-1]:.4f} vs {tot_want[-1]:.4f}')
    two = res.n_I.select(patch='rural').by('age').values[-1]
    print(f'   n_I rural by age (first 4 groups) {np.round(two[:4], 1)}; sum over ages {two.sum():.1f} vs n_I rural {I_p[-1, 2]:.1f}')
    for name in ['births', 'deaths', 'bite_indoor']:
        try:
            print(f'   {name}.by(patch) total {np.round(res[name].by("patch").values.sum(0), 1)}')
        except Exception as e:
            print(f'   {name}.by(patch): {type(e).__name__}: {e}')
    try:
        print('   prevalence.select(patch=rural, age=0) end', np.round(res.prevalence.select(patch='rural', age=0).values[-1], 4), 'from counts', np.round(res.n_I.select(patch='rural', age=0).values[-1]/(res.n_I.select(patch='rural', age=0).values[-1] + res.n_S.select(patch='rural', age=0).values[-1]), 4))
    except Exception as e:
        print('   two-way ratio:', type(e).__name__, e)
# Can a result be an expression of flows over a state (the force of infection, i.e. the EIR v3 records per node)?
S, I = ss.states('S I')
try:
    mal = ss.Disease(bite=S >> I @ 0.4*ss.peryear(6), recovery=I >> S @ ss.days(200), foi=None, name='malaria')
except Exception as e:
    print('foi=None:', type(e).__name__, e)
for spelling in ['flow/state']:
    S, I = ss.states('S I')
    try:
        bite = S >> I @ 0.4*ss.peryear(6)
        mal = ss.Disease(bite=bite, recovery=I >> S @ ss.days(200), eir_seen=bite / S, name='malaria')
        sim = ss.Sim(mal, n_agents=2000, start=2000, stop=2001, dt=ss.days(7), verbose=0).run()
        print(f'result flow/state: mean {np.mean(sim.results.malaria.eir_seen.values[1:]):.4f} per step (expect 1-exp(-2.4*7/365) = {1-np.exp(-2.4*7/365):.4f})')
    except Exception as e:
        print(f'result {spelling}:', type(e).__name__, str(e)[:400])
