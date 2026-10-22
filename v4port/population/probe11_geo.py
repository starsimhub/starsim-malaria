""" Probe: NodeGrid(populations=, labels=) as ss.geo(names=, pop=): exact counts, total population, and agents standing for more people """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
for kw in [dict(), dict(n_agents=1000)]:
    S, I = ss.states('S I')
    mal = ss.Disease(bite=S >> I @ 0.4*ss.by(patch=[ss.peryear(4), ss.peryear(6), ss.peryear(8)]), recovery=I >> S @ ss.days(200), name='malaria')
    mal.stratify(patch=ss.geo(names=['urban', 'peri', 'rural'], pop=[2000, 1500, 1500]))
    sim = ss.Sim(mal, start=2000, stop=2001, dt=ss.days(7), verbose=0, **kw).init()
    sim.run()
    r = sim.results.malaria
    print(f'{kw}: solver {sim.pars.solver if "solver" in sim.pars else "?"}; people by patch at the start {np.round(r.n_S.by("patch").values[0] + r.n_I.by("patch").values[0], 1)}; total_pop {sim.pars.get("total_pop")}')
