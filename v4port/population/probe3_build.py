""" Probe: which demography spellings build and run with the declared patch model """
import sys, traceback, time
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
import mal_v4 as m
for solver in ['abm', 'ode']:
    for demog in ['modules', 'declared', 'lifetable']:
        try:
            T = time.perf_counter()
            sim = m.make_sim(seed=1, solver=solver, demog=demog, years=2)
            sim.run()
            res = sim.results.malaria
            n = res.n_S.values + res.n_I.values
            print(f'{solver:4s} {demog:9s}: OK  N {n[0]:.0f} -> {n[-1]:.0f}  by patch end {np.round(res.n_S.by("patch").values[-1] + res.n_I.by("patch").values[-1])}  {time.perf_counter()-T:.1f}s')
        except Exception as e:
            print(f'{solver:4s} {demog:9s}: {type(e).__name__}: {str(e)[:600]}')
