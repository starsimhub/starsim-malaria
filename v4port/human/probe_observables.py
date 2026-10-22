""" Observables of the infection state (malariasim's infectiousness, hrp2_pg, fever_F) and the biting-weighted infectiousness the mosquitoes need (p_inf per node), as declared results """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
from human_v4 import gamma_levels, eir_series

w = list(gamma_levels(1.0, 10))
c = 0.3
tries = {
    'prevalence = I / (S + I)':                   lambda S, I: I / (S + I),
    'fever = 98.6 + 4.0 * I.frac (a constant shift while infected)': lambda S, I: 98.6 + 4.0 * I.frac,
    'p_inf = c * I.frac':                          lambda S, I: c * I.frac,
    'p_inf = c * I / (S + I)':                     lambda S, I: c * I / (S + I),
    'fever = 98.6 + 4.0 * I / (S + I)':            lambda S, I: 98.6 + 4.0 * I / (S + I),
    'p_inf = c * (w-weighted I) via sum of selects': None,
    'p_inf = c * ss.by(biting=w) * I / (S + I)  (biting-weighted)': lambda S, I: c * ss.by(biting=w) * I / (S + I),
}
for label, f in tries.items():
    for solver in ['abm', 'ode']:
        if f is None: continue
        try:
            S = ss.State(); I = ss.State(rel_trans=1.0)
            S, I = ss.states('S I')
            res = f(S, I)
            d = ss.Disease(infection=S >> I @ 0.4 * eir_series([10]*12), recovery=I >> S @ ss.days(200), obs=res, init=dict(I=0.0), name='malaria')
            d.stratify(patch=['n0', 'n1'], split=[0.5, 0.5])
            d.stratify(biting=[f'b{k}' for k in range(10)], split=[0.1]*10, adjust={'infection': w})
            kw = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
            sim = ss.Sim(d, start=2000, stop=2002, dt=ss.days(1), solver=solver, verbose=0, **kw).run()
            r = sim.results.malaria.obs
            print(f'OK   {solver:3s} {label:62s} final {float(r[-1]):.4f}; by patch', np.round(r.by('patch').values[-1], 4))
        except Exception as e:
            print(f'FAIL {solver:3s} {label:62s} {type(e).__name__}: {str(e)[:200]}')

# What the mosquitoes should see: the biting-weighted infectious fraction, computed by hand from the stratified result
S, I = ss.states('S I')
d = ss.Disease(infection=S >> I @ 0.4 * ss.by(patch=[eir_series([5]*12), eir_series([50]*12)]), recovery=I >> S @ ss.days(200), init=dict(I=0.0), name='malaria')
d.stratify(patch=['n0', 'n1'], split=[0.5, 0.5])
d.stratify(biting=[f'b{k}' for k in range(10)], split=[0.1]*10, adjust={'infection': w})
sim = ss.Sim(d, start=2000, stop=2003, dt=ss.days(1), solver='ode', verbose=0, total_pop=10_000).run()
r = sim.results.malaria
for p in ['n0', 'n1']:
    nI = np.array([float(r.n_I.select(patch=p, biting=f'b{k}')[-1]) for k in range(10)])
    nN = np.array([float(r.n_I.select(patch=p, biting=f'b{k}')[-1] + r.n_S.select(patch=p, biting=f'b{k}')[-1]) for k in range(10)])
    print(f'{p}: unweighted infectious fraction {nI.sum()/nN.sum():.4f}; biting-weighted (what malariasim gives the mosquitoes) {(np.array(w)*nI).sum()/nN.sum():.4f}')
