""" Biting heterogeneity in a declared Ross-Macdonald model: the human side takes it as adjust= on infection, but the mosquitoes' infection
rate counts Ih.n unweighted. Is there a spelling for the biting-weighted count (sum over levels of w_k * n_Ih[k]) that malariasim gives the mosquitoes? """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from human_v4 import gamma_levels

w = list(gamma_levels(1.0, 10)) # Mean-1 biting weights, Gamma(shape 1) in 10 equal-probability bins
a, b, c = 0.3, 0.5, 0.5
def make(weighted):
    Sh, Ih = ss.states('Sh Ih'); Sv, Ev, Iv = ss.states('Sv Ev Iv')
    Nh = (Sh + Ih).n
    H = ss.by(biting=w) * Ih.n if weighted else Ih.n
    d = ss.Disease(
        infection  = Sh >> Ih @ ss.perday(a*b) * Iv.n / Nh,
        recovery   = Ih >> Sh @ ss.days(100),
        emergence  = ss.null >> Sv @ ss.perday(1_000),
        biting     = Sv >> Ev @ ss.perday(a*c) * H / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
        init       = dict(Ih=0.05, Sv=10_000), name='malaria')
    d.stratify(biting=[f'b{k}' for k in range(10)], split=[0.1]*10, adjust={'infection': w}, states=[Sh, Ih])
    return d
for weighted in [False, True]:
    for solver in ['abm', 'ode']:
        kw = dict(n_agents=5_000) if solver == 'abm' else dict(total_pop=5_000)
        try:
            sim = ss.Sim(make(weighted), unit='day', stop=730, solver=solver, verbose=0, **kw).run()
            r = sim.results.malaria
            print(f'OK   weighted={weighted!s:5s} {solver:3s} human prevalence {r.n_Ih[-1]/5000:.3f}, infectious mosquitoes {r.n_Iv[-1]:.0f}')
        except Exception as e:
            print(f'FAIL weighted={weighted!s:5s} {solver:3s} {type(e).__name__}: {str(e)[:250]}')

# Which value does ss.by(biting=w) take for the mosquito states, which have no biting level? Put the levels in the reverse order
print('levels reversed (first level has w =', round(w[::-1][0], 2), ')')
w = w[::-1]
for weighted in [False, True]:
    sim = ss.Sim(make(weighted), unit='day', stop=730, solver='ode', verbose=0, total_pop=5_000).run()
    r = sim.results.malaria
    print(f'OK   weighted={weighted!s:5s} ode human prevalence {r.n_Ih[-1]/5000:.3f}, infectious mosquitoes {r.n_Iv[-1]:.0f}')
# By hand: the right weighted count, sum_k w_k n_Ih[k], against the unweighted one, from the unweighted run's groups
r = ss.Sim(make(False), unit='day', stop=730, solver='ode', verbose=0, total_pop=5_000).run().results.malaria
nI = np.array([float(r.n_Ih.select(biting=f'b{k}')[-1]) for k in range(10)])
print(f'unweighted Ih.n {nI.sum():.0f}; biting-weighted sum_k w_k n_Ih[k] {(np.array(w)*nI).sum():.0f}')
