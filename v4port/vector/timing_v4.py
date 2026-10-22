"""Timings on Starsim v4 (best of 3 after a warm-up): the declared hybrid (ODE, agents), the hybrid's v3 MosquitoABM with declared humans, the declared malariasim mosquitoes (ODE, agents), and the coded mosquito agents"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
sys.path.insert(0, sc.thispath(__file__))
import hybrid_v4, hybrid_v4_coded_mosquito as hc, ms_mosquito_v4 as msv4, mosquito_agents_v4 as mav4
def hyb(solver):
    pop = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    return ss.Sim(hybrid_v4.make_malaria(), solver=solver, unit='day', start=0, stop=365*5, dt=1, verbose=0, **pop)
def hyb_coded(solver):
    np.random.seed(0); pop = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    sim = ss.Sim(hc.make_humans(), connectors=hc.MosquitoABM(c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100), analyzers=hc.MosqResults(), solver=solver, unit='day', start=0, stop=365*5, dt=1, verbose=0, **pop)
    sim.pars.update(dict(a=0.24, Nh=10_000), create=True); return sim
def ms(solver):
    pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
    return ss.Sim(msv4.make_malaria(), solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop)
def ms_coded():
    return ss.Sim(mav4.make_humans(), connectors=mav4.MosquitoAgents(), solver='abm', unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, n_agents=5000)
cases = dict(hybrid_ode=lambda: hyb('ode'), hybrid_abm=lambda: hyb('abm'), hybrid_abm_coded_mosq=lambda: hyb_coded('abm'),
             ms_declared_ode=lambda: ms('ode'), ms_declared_abm=lambda: ms('abm'), ms_coded_mosq_agents=ms_coded)
out = {}
for name, make in cases.items():
    make().run()
    times = []
    for i in range(3):
        sim = make(); T = sc.tic(); sim.run(); times.append(sc.toc(T, output=True))
    out[name] = min(times)
    print(f'{name}: best of 3 {min(times):.3f} s ({np.round(times, 3).tolist()})')
sc.savejson(sc.thispath(__file__)/f'timing_v4_{sys.argv[1] if len(sys.argv) > 1 else 0}.json', out)
