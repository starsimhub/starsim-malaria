""" Minimal repro: what makes a 3-patch declared SIS slow to step? A time series inside ss.by(patch=), age groups, and a death rate by age.
5,000 people, 5 years daily; time per step and the time spent converting typed values (TimePar.to), as ODE and as agents """
import time, cProfile, pstats
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
def seasonal(v):
    return ss.timeseries({0: ss.peryear(v), 0.5: ss.peryear(2*v), 1.0: ss.peryear(v)}, mode='linear', period=ss.years(1))
def make(solver, ts, age, death):
    S, I = ss.states('S I')
    e = ss.by(patch=[seasonal(v) for v in [4.0, 6.0, 8.0]]) if ts else ss.by(patch=[ss.peryear(4.0), ss.peryear(6.0), ss.peryear(8.0)])
    mal = ss.Disease(bite=S >> I @ 0.4*e, recovery=I >> S @ ss.days(200), name='malaria')
    mal.stratify(patch=['a', 'b', 'c'], kind='space')
    demog = [ss.birth(rate=ss.peryear(0.03)), ss.death(rate=ss.by(age=[ss.peryear(0.01 + 0.002*k) for k in range(20)]))] if death else None
    return ss.Sim(mal, solver=solver, n_agents=5000, demographics=demog, stratify=dict(age=list(range(0, 100, 5))) if age else None, start=2000, stop=2005, dt=ss.days(1), verbose=0)
cases = [(False, False, False), (True, False, False), (True, True, False), (False, True, True), (True, True, True)]
for solver in ['ode', 'abm']:
    for ts, age, death in cases:
        times = []
        for rep in range(3):
            sim = make(solver, ts, age, death); T = time.perf_counter(); sim.run(); times.append(time.perf_counter() - T)
        pr = cProfile.Profile(); sim = make(solver, ts, age, death); pr.enable(); sim.run(); pr.disable()
        st = pstats.Stats(pr); conv = sum(v[3] for k, v in st.stats.items() if k[2] == 'to' and k[0].endswith('time.py')); tot = sum(v[2] for v in st.stats.values())
        print(f'{solver}: {"seasonal" if ts else "constant"} by(patch){", 20 age groups" if age else ""}{", birth + death by age" if death else ""}: {1e3*min(times[1:])/1826:.3f} ms/step; TimePar.to {100*conv/tot:.0f}% of profiled time')
