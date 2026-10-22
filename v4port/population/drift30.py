""" Is the declared population stable? 30 seeds, 5,000 people, 20 years, weekly: N at the end and births minus deaths, for each demography spelling """
import sys
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m
for demog in sys.argv[1:]:
    rows = []
    for seed in range(30):
        sim = m.make_sim(seed=seed, solver='abm', demog=demog); sim.run()
        res = sim.results.malaria
        b = res.births.values.sum() if 'births' in res.keys() else sum(mm.results.new.sum() for mm in sim.demographics.values() if isinstance(mm, ss.Births))
        d = res.deaths.values.sum() if 'deaths' in res.keys() else sum(mm.results.new.sum() for mm in sim.demographics.values() if isinstance(mm, ss.Deaths))
        rows.append([res.people.values[-1], b, d, sim.people.alive.sum()])
    r = np.array(rows, float)
    print(f'{demog:9s}: N end {r[:,0].mean():.1f} +/- {r[:,0].std()/np.sqrt(30):.1f} (SE); births {r[:,1].mean():.0f}; deaths {r[:,2].mean():.0f}; alive agents end {r[:,3].mean():.1f}', flush=True)
