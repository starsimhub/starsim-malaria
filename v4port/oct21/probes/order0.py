import starsim as ss, numpy as np
def make(init):
    Sh, Ih = ss.states(2)
    Ev, Iv = ss.states(2, people=False)
    N = (Sh + Ih).n
    return ss.Disease(infection=Sh >> Ih @ ss.perday(1e-4) * Iv.n / N, incubation=Ev >> Iv @ ss.perday(50), init=init, name='mal')
for init in [dict(Sh=1.0, Ev=1_000_000), dict(Ev=1_000_000), dict(Sh=1.0, Iv=1_000_000), dict(Iv=1_000_000)]:
    for solver in ['abm', 'ode']:
        pop = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        sim = ss.Sim(make(init), solver=solver, unit='day', start=0, stop=3, dt=1, verbose=0, **pop).run()
        r = sim.results.mal
        print(f'{str(init):32s} {solver}: infections {np.round(r.infection.values,1)}; n_Sh {r.n_Sh.values}; n_Ih {r.n_Ih.values}; n_Iv {r.n_Iv.values}')
sim = ss.Sim(make(dict(Sh=1.0, Ev=1_000_000)), solver='ode', unit='day', start=0, stop=3, dt=1, total_pop=10_000, verbose=0).init()
print(sim.describe() if False else '')
sim.describe()
