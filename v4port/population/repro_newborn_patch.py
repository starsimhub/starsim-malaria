""" Minimal repro: which patch do newborns join? A declared SIS stratified by patch (split 0.4/0.3/0.3), with ss.Births/ss.Deaths modules or with ss.birth/ss.death """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
for kind in ['modules', 'declared']:
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(0.5)), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.1, name='sis')
    sis.stratify(patch=['a', 'b', 'c'], kind='space', split=[0.4, 0.3, 0.3])
    demog = [ss.Births(birth_rate=ss.peryear(40)), ss.Deaths(death_rate=ss.peryear(40))] if kind == 'modules' else [ss.birth(rate=ss.peryear(0.04)), ss.death(rate=ss.peryear(0.04))]
    sim = ss.Sim(sis, n_agents=10_000, demographics=demog, start=2000, stop=2020, dt=ss.years(1/12), rand_seed=2, verbose=0).run()
    ppl = sim.people
    patch = np.asarray(sim.people.patch.values) if hasattr(ppl, 'patch') else np.asarray(sim.diseases.sis.patch.values)
    born = np.asarray(ppl.age.values) < 20
    res = sim.results.sis
    n0 = res.n_S.by('patch').values[0] + res.n_I.by('patch').values[0]; n1 = res.n_S.by('patch').values[-1] + res.n_I.by('patch').values[-1]
    print('patch dtype', patch.dtype, 'values', np.unique(patch)); print(f'{kind:8s}: patch shares at start {np.round(n0/n0.sum(), 3)}, after 20 y {np.round(n1/n1.sum(), 3)}; newborns (age < 20) by patch {np.bincount(patch[born].astype(int), minlength=3)}, of {born.sum()}')
