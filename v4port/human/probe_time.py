""" How many steps does start=2000, stop=2005, dt=ss.days(1) give, and what does describe() say """
import starsim as ss; print(ss.__version__, ss.__file__)
S, I = ss.states('S I')
mal = ss.Disease(infection = S >> I @ 0.4 * ss.peryear(10), recovery = I >> S @ ss.days(200), init=dict(I=0.0), name='malaria')
sim = ss.Sim(mal, start=2000, stop=2005, dt=ss.days(1), n_agents=1000, verbose=0).run()
print('steps', len(sim.results.malaria.timevec), sim.results.malaria.timevec[:3], sim.results.malaria.timevec[-2:])
print('t.yearvec', sim.t.yearvec[:3], sim.t.yearvec[-2:])
