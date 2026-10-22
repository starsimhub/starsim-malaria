""" malariasim (v3): with no age bins and no demographics, MalariaHumanODE's single 100-year bin still ages people out (rate dt/100) with no births,
so the population falls by about 1% a year. Reads out_v3.npz (run_v3.py) and also runs one sim directly """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc, malariasim as ms
from scenarios import load
for key in ['A/ode/0', 'B/ode/0']:
    N = np.array(load('out_v3.npz')[key]['N']).sum(1)
    print(f'{key}: total humans at start {N[0]:.0f}, after 1 year {N[365]:.0f}, after 5 years {N[-1]:.0f} ({N[-1]/N[0] - 1:+.1%})')
sim = ms.Sim(human='ode', mosquito='eir_only', start=2000, stop=2010, dt=ss.days(1), verbose=0).run()
r = sim.diseases[0].results
N = r.n_susceptible.values + r.n_infected.values
print(f'10 years, default ms.Sim(human="ode"): {N[0]:.0f} -> {N[-1]:.0f} ({N[-1]/N[0] - 1:+.1%}); NodeState n_humans at the end {sim.connectors[0].n_humans}')
