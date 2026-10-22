""" Probe: does v3's excess prevalence over the SIS equilibrium (0.568) shrink with the step? (constant EIR 6/y, b=0.4, 200-day infections) """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
import malariasim as ms
for dt in [7, 1]:
    vals = []
    for seed in range(3):
        sim = ms.Sim(human='sis', mosquito='eir_only', common_pars=ms.CommonParams(), mosquito_pars=dict(eir=6.0), start=2000, stop=2012, dt=ss.days(dt), nodes=ms.NodeGrid(populations=[5000]), rand_seed=seed, verbose=0).run()
        res = sim.results.malariahumansis
        k = int(5*365/dt)
        vals.append((res.n_infected.values/sim.results.n_alive.values)[-k:].mean())
    print(f'v3 dt={dt} d: prev {np.mean(vals):.4f}')
