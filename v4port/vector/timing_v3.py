"""Timings on Starsim 3 (best of 3 after a warm-up): malaria_hybrid HumanABM x MosquitoODE and ABM x ABM; malariasim SIS with ODE and agent mosquitoes"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
sys.path.insert(0, sc.thispath(__file__))
import hybrid_v3 as mh
import ms_v3_ref
def hybrid(H, M):
    np.random.seed(0)
    return mh.Sim(pars=dict(a=0.24, Nh=10_000, start=0, stop=365*5, dt=1, verbose=0), human=H(b=0.15, r=1/120, Ih0=10), mosquito=M(c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100))
cases = dict(hybrid_ode_ode=lambda: hybrid(mh.HumanODE, mh.MosquitoODE), hybrid_abm_ode=lambda: hybrid(mh.HumanABM, mh.MosquitoODE), hybrid_abm_abm=lambda: hybrid(mh.HumanABM, mh.MosquitoABM),
             ms_sis_mosq_ode=lambda: ms_v3_ref.make('ode', 0), ms_sis_mosq_agents=lambda: ms_v3_ref.make('agents', 0))
out = {}
for name, make in cases.items():
    make().run()
    times = []
    for i in range(3):
        sim = make(); T = sc.tic(); sim.run(); times.append(sc.toc(T, output=True))
    out[name] = min(times)
    print(f'{name}: best of 3 {min(times):.3f} s ({np.round(times, 3).tolist()})')
sc.savejson(sc.thispath(__file__)/f'timing_v3_{sys.argv[1] if len(sys.argv) > 1 else 0}.json', out)
