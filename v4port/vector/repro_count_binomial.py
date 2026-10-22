"""The same count as repro_count_rk4.py (emergence 5,000/day, lifespan 10 days, start 10,000) under solver='binomial', by step size"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from repro_count_rk4 import make
for dt in [1, 7, 21, 30]:
    sim = ss.Sim(make(), solver='binomial', unit='day', start=0, stop=360, dt=dt, verbose=0, total_pop=2000).run()
    print(f'binomial, dt={dt}: Sv by step', np.round(np.asarray(sim.results.mal.n_Sv)[:6]).tolist(), f'end {sim.results.mal.n_Sv[-1]:,.0f}')
