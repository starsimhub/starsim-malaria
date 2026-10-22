"""
Can a declared model make the mosquitoes agents? The spec says a state that is not people is always a count;
ss.State(people=True) on the mosquito states is the only declared route. What happens?
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from ms_mosquito_v4 import emergence
Sh, Ih = ss.states(2)
Sv, Ev, Iv = [ss.State(people=True, name=n) for n in ['Sv', 'Ev', 'Iv']]
N = (Sh + Ih).n
mal = ss.Disease(infection=Sh >> Ih @ ss.perday(0.15) * Iv.n / N, recovery=Ih >> Sh @ ss.perday(1/200),
                 emergence=ss.null >> Sv @ ss.perday(5) * N / N, biting=Sv >> Ev @ ss.perday(0.15) * Ih.n / N,
                 incubation=Ev >> Iv @ ss.perday(1/12), mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1),
                 init=dict(Ih=50, Sv=500), name='malaria')
try:
    sim = ss.Sim(mal, solver='abm', n_agents=1000, unit='day', start=0, stop=200, dt=1, verbose=1)
    sim.run()
    r = sim.results.malaria
    print('n_alive (agents) at start, end:', sim.results.n_alive[0], sim.results.n_alive[-1])
    print('Sh, Ih, Sv, Ev, Iv at end:', [round(float(r[f"n_{s}"][-1])) for s in ['Sh', 'Ih', 'Sv', 'Ev', 'Iv']])
    print('mean age of agents in Sv at the end:', float(np.mean(sim.people.age[mal.Sv.uids])) if hasattr(mal.Sv, 'uids') else None)
except Exception as e:
    print(type(e).__name__, str(e)[:800])
