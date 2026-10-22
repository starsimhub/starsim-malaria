""" v4 version of run_population_stability.py: many seeds, population in total, by 5-year age group and by node, with bands, no analyzer """
import numpy as np, sciris as sc, matplotlib.pyplot as plt
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m

if __name__ == '__main__':
    sim = m.make_sim(seed=0, solver='abm', demog='declared')
    msim = ss.parallel(sim, n_runs=6, n_cpus=2)
    people = msim.results.malaria.people
    for dim in ['age', 'patch']:
        levels = people.by(dim) # Workaround: on a MultiSim, by() gives a dict of results, one per level, not one result with a column per level
        fig, axes = plt.subplots(*sc.getrowscols(len(levels)), figsize=(12, 8), sharex=True)
        for ax, (label, res) in zip(np.ravel(axes), levels.items()):
            res.plot(ax=ax); ax.set_title(label)
            v = np.asarray(res.values); print(f'{dim} {label:6s}: mean over runs {v[:, 0].mean():7.1f} -> {v[:, -1].mean():7.1f} ({100*(v[:, -1].mean()/v[:, 0].mean() - 1):+5.1f}%)')
        plt.savefig(f'v4_stability_{dim}.png', dpi=50); plt.close('all')
