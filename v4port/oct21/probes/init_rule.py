import starsim as ss, numpy as np
def make(init, pf=True):
    Sh, Ih = ss.states(2)
    Sv, Iv = ss.states(2, people=False)
    N = (Sh + Ih).n
    return ss.Disease(infection=Sh >> Ih @ ss.perday(1e-6) * Iv.n / N, recovery=Ih >> Sh @ ss.days(100), biting=Sv >> Iv @ ss.perday(0.01) * Ih.n / N,
                      init=init, name='mal')
for init in [dict(Sh=1.0), dict(Sh=1.0, Sv=5000), dict(Sh=0.9, Ih=0.1, Sv=5000), dict(Sh=0.9, Sv=5000), dict(Ih=0.1, Sv=5000), dict(Sh=1, Sv=5000.0), dict(Sh=1.0, Sv=0.5)]:
    for solver in ['abm', 'ode']:
        pop = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        sim = ss.Sim(make(init), solver=solver, unit='day', start=0, stop=1, dt=1, verbose=0, **pop).run()
        r = sim.results.mal
        print(f'{str(init):40s} {solver}: n_Sh[0] {r.n_Sh[0]:8.1f}  n_Ih[0] {r.n_Ih[0]:8.1f}  n_Sv[0] {r.n_Sv[0]:8.1f}  notes: {[n for n in sim.notes if "initial" in str(n)][:1]}')
