""" Probe: per-node mosquito emergence (NodeGrid.emergence) as a patch dimension of a count inflow.
(1) a total per patch, ss.by(patch=[100, 0, 0]) per day; (2) per human, e × Nh with e by patch. Steady state Sv = inflow × 10-day life """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
pops = [2000, 1500, 1500]
for label, make_rate in [
        ('total by patch, by(patch=[100,0,0])/day', lambda S, I: ss.by(patch=[ss.perday(100), ss.perday(0), ss.perday(0)])),
        ('fixed total 300/day, no by()', lambda S, I: ss.perday(300)),
        ('per human, by(patch=[0.05,0.02,0.01]) x Nh', lambda S, I: ss.by(patch=[ss.perday(0.05), ss.perday(0.02), ss.perday(0.01)]) * (S + I).n),
    ]:
    for solver in ['ode', 'abm']:
        S, I, Sv = ss.states('S I Sv')
        mal = ss.Disease(bite=S >> I @ ss.peryear(1.0), recovery=I >> S @ ss.days(200), emergence=ss.null >> Sv @ make_rate(S, I), mortality=Sv >> ss.null @ ss.days(10), name='malaria')
        mal.stratify(patch=['a', 'b', 'c'], kind='space', split=np.array(pops)/sum(pops))
        try:
            sim = ss.Sim(mal, solver=solver, total_pop=sum(pops), start=2000, stop=2000.5, dt=ss.days(1), verbose=0, **(dict(n_agents=sum(pops)) if solver == 'abm' else {})).run()
            print(f'{label} [{solver}]: Sv by patch at the end {np.round(sim.results.malaria.n_Sv.by("patch").values[-1], 1)}')
        except Exception as e:
            print(f'{label} [{solver}]: {type(e).__name__}: {str(e)[:300]}')
print('expected: (1) [1000, 0, 0]; (2) 3000 shared by split = [1200, 900, 900]; (3) [0.05*2000, 0.02*1500, 0.01*1500]*10 = [1000, 300, 150]')
