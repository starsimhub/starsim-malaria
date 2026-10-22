""" Speed, like for like: the population part (human SIS driven by per-node seasonal EIR, 3 nodes, steady-state demography),
20,000 people, 5 years, daily steps (malariasim's default). Best of 3 after a warm-up, configurations interleaved """
import sys, time
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
N, YEARS, REPS = 20_000, 5, 4
pops = [int(N*p/sum(POPS)) for p in POPS]
if ss.__version__.startswith('3'):
    import malariasim as ms
    def make(cfg):
        nodes = ms.NodeGrid(populations=pops, labels=LABELS, eir=node_eir_profiles(), f_indoor=F_INDOOR)
        if cfg == 'fast':
            import v3_fast_demog as fd; demog = fd.make_steady_state_demographics()
        else:
            demog = ms.make_steady_state_demographics()
        return ms.Sim(human='sis', mosquito='eir_only', common_pars=ms.CommonParams(), age_data=sc.dataframe(ms.make_age_data()), age_bin_edges=ms.AGE_BIN_EDGES,
                      start=START, stop=START+YEARS, dt=ss.days(1), nodes=nodes, demographics=demog, analyzers=ms.ResultsAnalyzer(), rand_seed=0, verbose=0)
    configs = ['plain', 'fast']
else:
    import mal_v4 as m
    def make(cfg):
        solver, demog = cfg.split('/')
        mal = m.make_malaria(pops=pops)
        return ss.Sim(mal, demographics=m.steady_state(demog), stratify=dict(age=AGE_BIN_EDGES[:-1]), solver=solver, people=ss.People(N, age_data=sc.dataframe(m.age_data())),
                      start=START, stop=START+YEARS, dt=ss.days(1), rand_seed=0, verbose=0)
    configs = ['abm/declared', 'abm/modules', 'abm/fast', 'ode/declared']
times = {c: [] for c in configs}
for rep in range(REPS):
    for c in configs:
        sim = make(c); T = time.perf_counter(); sim.run(); times[c].append(time.perf_counter() - T)
for c in configs:
    print(f'{c:14s}: best of 3 after warm-up {min(times[c][1:]):.2f}s  (all {np.round(times[c], 2)})')
