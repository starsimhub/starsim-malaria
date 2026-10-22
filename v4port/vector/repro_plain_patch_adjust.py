"""
With a plain dimension named patch (no kind='space'), the mosquito counts are not split, and an adjustment of their
emergence by patch is accepted and has no effect (the species case raises InvalidAdjustment instead).
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from ms_mosquito_v4 import make_malaria
for adjust in [None, [1.0, 1/3], [0.0, 0.0]]:
    mal = make_malaria()
    mal.stratify(patch=['a', 'b'], split=[0.5, 0.5], **({} if adjust is None else dict(adjust={'emergence': adjust})))
    r = ss.Sim(mal, solver='ode', unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, total_pop=5000).run().results.malaria
    M = np.array(r.n_Sv) + np.array(r.n_Ev) + np.array(r.n_Iv)
    print(f'adjust emergence by patch = {adjust}: mosquitoes y3 mean {M[731:].mean():.1f}; emergence y3 {np.sum(r.emergence[731:]):.0f}')
for adjust in [[1/3, 1.0], [1.0, 1/3]]:
    for solver in ['ode', 'abm']:
        mal = make_malaria()
        mal.stratify(patch=['a', 'b'], split=[0.5, 0.5], adjust={'emergence': adjust})
        pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        r = ss.Sim(mal, solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop).run().results.malaria
        M = np.array(r.n_Sv) + np.array(r.n_Ev) + np.array(r.n_Iv)
        print(f'{solver}, adjust emergence by patch = {np.round(adjust, 3).tolist()}: mosquitoes y3 mean {M[731:].mean():.1f}')
