"""
malariasim's nodes for the vector part: mosquitoes per patch, each patch's humans bitten by their own patch's
mosquitoes. Two patches of 2,500; patch "b" has a third of the emergence. Checks that Iv.n and N are per patch.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from ms_mosquito_v4 import make_malaria
for kind in [None, 'space']:
    for solver in ['ode', 'abm']:
        mal = make_malaria()
        kw = {} if kind is None else dict(kind=kind)
        mal.stratify(patch=['a', 'b'], split=[0.5, 0.5], adjust={'emergence': [1.0, 1/3]}, **kw)
        pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        sim = ss.Sim(mal, solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=1 if solver == 'ode' else 0, **pop).run()
        r = sim.results.malaria
        Ih = r.n_Ih.by('patch').values; Sh = r.n_Sh.by('patch').values
        y3 = slice(731, None)
        try:
            Sv = np.round(r.n_Sv.by('patch').values[y3].mean(axis=0)).tolist()
        except Exception as e:
            Sv = f'not by patch ({e}); total {np.mean(r.n_Sv[731:]):.0f}'
        print(f'kind={kind}, {solver}: prevalence y3 by patch {np.round((Ih/(Ih+Sh))[y3].mean(axis=0), 4).tolist()}; Sv by patch: {Sv}')
# The same with the patches as two separate one-patch sims (each 2,500 people)
for em in [1.0, 1/3]:
    import ms_mosquito_v4 as m
    old = list(m.emergence); m.emergence[:] = [e*em for e in old]
    mal = m.make_malaria(); m.emergence[:] = old
    mal.pars if False else None
    sim = ss.Sim(mal, solver='ode', unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, total_pop=2500)
    sim.diseases.malaria if False else None
    sim.run(); r = sim.results.malaria
    print(f'one patch alone, emergence x{em:.3f} (ode): prevalence y3 {(np.array(r.n_Ih)/(np.array(r.n_Ih)+np.array(r.n_Sh)))[731:].mean():.4f}')
