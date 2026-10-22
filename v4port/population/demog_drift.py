""" Is v4's ss.Births + ss.Deaths(table) population stable? 30 seeds, 5,000 agents, 20 years weekly; births and deaths against the expected count """
import sys
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import demog_only as d
mu, n, br, _ = compute_mortality_schedule()
mode = sys.argv[1] if len(sys.argv) > 1 else 'plain'
rows = []
for seed in range(30):
    sim = d.make(seed, mode); sim.run()
    b = sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Births))
    dd = sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Deaths))
    rows.append([sim.results.n_alive[-1], b, dd, len(sim.results.timevec), sim.results.n_alive.values.mean()])
r = np.array(rows, float)
print(f'mode {mode}: n_end {r[:,0].mean():.1f} +/- {r[:,0].std()/np.sqrt(len(r)):.1f} (SE); births {r[:,1].mean():.1f}; deaths {r[:,2].mean():.1f}; births-deaths {np.mean(r[:,1]-r[:,2]):.1f} +/- {np.std(r[:,1]-r[:,2])/np.sqrt(len(r)):.1f}; n points {r[0,3]:.0f}')
print(f'expected births over 20 y at the calibrated rate {br:.5f}/y and mean pop {r[:,4].mean():.0f}: {br*20*r[:,4].mean():.0f}')
