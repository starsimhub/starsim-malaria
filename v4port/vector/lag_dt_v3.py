"""malariasim on Starsim 3 at dt = 1 and 7 days: mean human prevalence over the third year (3 seeds)"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, malariasim as ms
for dt in [1, 7]:
    out = {}
    for mos in ['ode', 'agents']:
        vals = []
        for seed in range(3):
            sim = ms.Sim(human='sis', mosquito=mos, start=2000, stop=2003, dt=ss.days(dt), nodes=ms.NodeGrid(populations=[5000]), rand_seed=seed, verbose=0,
                         human_pars=dict(init_prev=ss.bernoulli(p=0.01), prob_human_infected_on_bite=0.5, prob_mosquito_infected_on_bite=0.5)).run()
            p = np.array(sim.diseases[0].results.n_infected)/np.array(sim.results.n_alive)
            vals.append(p[-int(365/dt):].mean())
        out[mos] = f'{np.mean(vals):.4f}±{np.std(vals):.4f}'
    print(f'dt={dt}:', out)
