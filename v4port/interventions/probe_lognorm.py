""" Probe: a lognormal dwell, as malariasim's infection duration, under agents and ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
for spelling in ['lognorm_ex(mean=ss.days(200), std=ss.days(200))', 'lognorm_ex(mean=200, std=200, unit=day)']:
    for solver in ['abm', 'ode']:
        try:
            S, I = ss.states(2)
            dur = ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)) if 'ss.days' in spelling else ss.lognorm_ex(mean=200, std=200, unit='day')
            mal = ss.Disease(recovery=I >> S @ dur, init=dict(I=0.5), name='m')
            sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(400), dt=ss.days(1), solver=solver, verbose=0, **(dict(n_agents=4000) if solver=='abm' else dict(total_pop=4000)))
            sim.run()
            print(f'{spelling} {solver}: n_I at day 334 {sim.results.m.n_I.values[335]:.0f} (lognormal survival 0.151 x 2000 = 302; exponential 0.188 x 2000 = 376)')
        except Exception as e:
            print(f'{spelling} {solver}: {type(e).__name__}: {e}')
print(sim.describe())
