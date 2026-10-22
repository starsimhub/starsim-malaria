""" Probe: why is v4's prevalence ~3% below v3's? Constant EIR 6/y, no demography, one node: v4 ODE, v4 agents, against the SIS equilibrium """
import sys
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
b, E, dur = 0.4, 6.0, 200
lam = b*E; r = 365/dur
print(f'continuous-time equilibrium lam/(lam+r) = {lam/(lam+r):.4f}')
if ss.__version__.startswith('4'):
    for solver, kw in [('ode', dict(total_pop=5000)), ('abm', dict(n_agents=5000))]:
        for dwell in ['exp', 'lognorm']:
            S, I = ss.states('S I')
            rec = I >> S @ (ss.days(dur) if dwell == 'exp' else ss.lognorm_ex(mean=ss.days(dur), std=ss.days(dur)))
            mal = ss.Disease(bite=S >> I @ b * ss.peryear(E), recovery=rec, name='malaria')
            vals = []
            for seed in range(1 if solver == 'ode' else 4):
                sim = ss.Sim(mal, solver=solver, start=2000, stop=2015, dt=ss.days(7), rand_seed=seed, verbose=0, **kw).run()
                res = sim.results.malaria; n = res.n_S.values + res.n_I.values
                vals.append((res.n_I.values/n)[-260:].mean())
            print(f'v4 {solver} {dwell}: prev {np.mean(vals):.4f}')
else:
    import malariasim as ms
    for heading, cp in [('b=0.4, lognorm 200/200', ms.CommonParams())]:
        vals = []
        for seed in range(4):
            sim = ms.Sim(human='sis', mosquito='eir_only', common_pars=cp, mosquito_pars=dict(eir=E), start=2000, stop=2015, dt=ss.days(7), nodes=ms.NodeGrid(populations=[5000]), rand_seed=seed, verbose=0).run()
            res = sim.results.malariahumansis
            vals.append((res.n_infected.values/sim.results.n_alive.values)[-260:].mean())
        print(f'v3 {heading}: prev {np.mean(vals):.4f}')
    # The dwell v3 gives: mean recovery time in steps
    sim = ms.Sim(human='sis', mosquito='eir_only', common_pars=ms.CommonParams(), mosquito_pars=dict(eir=E), start=2000, stop=2003, dt=ss.days(7), nodes=ms.NodeGrid(populations=[5000]), rand_seed=0, verbose=0).run()
    d = sim.diseases[0]
    inf = d.infected.uids
    print('v3 scheduled stay in steps (ti_recovered - ti_infected), mean:', np.mean(d.ti_recovered[inf] - d.ti_infected[inf]), 'vs 200/7 =', 200/7)
