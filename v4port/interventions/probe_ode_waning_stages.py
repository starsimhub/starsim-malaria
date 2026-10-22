""" Probe: how close the ODE's levels of time since delivery get to agents for malariasim's box + exponential vaccine waning """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from scen import *
from malaria_v4 import box_exp

def run(solver, n_stages=10, seed=1):
    ss.products.Held.n_stages = n_stages
    S, I = ss.states(2)
    mal = ss.Disease(infection=S >> I @ ss.peryear(3.0), recovery=I >> S @ ss.days(DUR_INF), init=dict(I=0.6), name='mal')
    vax = ss.Vaccine(efficacy=dict(infection=VAX['eff']), waning=box_exp(VAX['eff'], VAX['box'], VAX['tau']))
    kw = dict(n_agents=20000) if solver == 'abm' else dict(total_pop=20000)
    sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(DAYS), dt=ss.days(1), solver=solver, interventions=ss.vaccinate(vax, coverage=0.8, at=[ss.days(T_INTV)]), verbose=0, rand_seed=seed, **kw).run()
    return sim.results.mal.n_I.values[np.array([395, 455, 545, 729]) + 1]/20000

abm = np.mean([run('abm', seed=s) for s in range(1, 4)], axis=0)
print('abm (3 seeds, 20,000 agents) prevalence at days 395, 455, 545, 729:', np.round(abm, 4))
for n in [10, 40, 160]:
    ode = run('ode', n)
    print(f'ode, n_stages={n:3d}:', np.round(ode, 4), ' relative difference from agents:', np.round(ode/abm - 1, 3))
ss.products.Held.n_stages = 10
