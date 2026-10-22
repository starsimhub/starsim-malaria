"""
How v4's within-step order behaves as the step grows: the declared model as agents (both directions read the other
population at the start of the step), as an ODE (no order), and the coded mosquito agents (mosquitoes first, then
humans). Mean human prevalence over the third year, dt = 1 and 7 days, 3 seeds for agents.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
sys.path.insert(0, sc.thispath(__file__))
import ms_mosquito_v4 as msv4, mosquito_agents_v4 as mav4
def prev_y3(sim):
    r = sim.results.malaria; p = np.array(r.n_Ih)/(np.array(r.n_Ih) + np.array(r.n_Sh))
    return p[-int(365/sim.pars.dt.days if hasattr(sim.pars.dt, 'days') else 365/sim.pars.dt):].mean()
for dt in [1, 7]:
    out = {}
    for label in ['declared ode', 'declared abm', 'coded mosquitoes, abm']:
        vals = []
        for seed in ([0] if 'ode' in label else range(3)):
            if label == 'coded mosquitoes, abm':
                sim = ss.Sim(mav4.make_humans(), connectors=mav4.MosquitoAgents(), solver='abm', unit='day', start='2000-01-01', stop='2003-01-01', dt=dt, rand_seed=seed, verbose=0, n_agents=5000)
            else:
                solver = label.split()[-1]; pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
                sim = ss.Sim(msv4.make_malaria(), solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=dt, rand_seed=seed, verbose=0, **pop)
            sim.run(); vals.append(prev_y3(sim))
        out[label] = f'{np.mean(vals):.4f}±{np.std(vals):.4f}'
    print(f'dt={dt}:', out)
