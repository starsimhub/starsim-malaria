""" Repro: a declared ratio (prevalence = I/(S+I)) by patch, from single sims and from ss.parallel's results """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
import mal_v4 as m
sim = m.make_sim(seed=0, solver='abm', demog='declared', years=2)
msim = ss.parallel(sim, n_runs=2, n_cpus=2)
pr = msim.results.malaria.prevalence
one = [s.results.malaria.prevalence.select(patch='rural').values[-1] for s in msim.sims] if getattr(msim, 'sims', None) else None
print('from the MultiSim, select(patch=rural), last time, per run:', np.round(np.asarray(pr.select(patch='rural').values)[:, -1], 4))
print('from the MultiSim, by(patch)[rural], last time, per run:   ', np.round(np.asarray(pr.by('patch')['rural'].values)[:, -1], 4))
print('total prevalence from the MultiSim, last time:            ', np.round(np.asarray(pr.values)[:, -1], 4))
I = msim.results.malaria.n_I.select(patch='rural').values; N = msim.results.malaria.people.select(patch='rural').values
print('from the counts n_I/people (rural), last time:             ', np.round(np.asarray(I)[:, -1]/np.asarray(N)[:, -1], 4))
print('single sims (if kept):', one)
