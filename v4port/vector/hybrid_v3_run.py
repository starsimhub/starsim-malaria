"""
Run malaria_hybrid (removed from malariasim in 3aac4c4; copy in ./hybrid_v3) on Starsim 3:
the four human x mosquito combinations, with the parameters of its own run_malaria_hybrid.py,
over several seeds. Writes hybrid_v3_summary.json and hybrid_v3_traces.csv.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, json, time, numpy as np, sciris as sc
sys.path.insert(0, sc.thispath(__file__))
import hybrid_v3 as mh

Nh = 10_000
hum_pars = dict(b=0.15, r=1/120, Ih0=10)
mos_pars = dict(c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100)
combos = [('ode', mh.HumanODE, 'ode', mh.MosquitoODE), ('abm', mh.HumanABM, 'ode', mh.MosquitoODE),
          ('ode', mh.HumanODE, 'abm', mh.MosquitoABM), ('abm', mh.HumanABM, 'abm', mh.MosquitoABM)]
seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

def summarize(sim):
    ph = np.array(sim.human.results.prev_h); pm = np.array(sim.mosquito.results.prev_m); m = np.array(sim.mosquito.results.m)
    yr5 = slice(4*365, 5*365)
    return dict(prev_h_y1=float(ph[365]), prev_h_y5_mean=float(ph[yr5].mean()), prev_h_y5_max=float(ph[yr5].max()), prev_h_y5_min=float(ph[yr5].min()),
                prev_m_y5_mean=float(pm[yr5].mean()), m_y5_mean=float(m[yr5].mean()), m_y5_max=float(m[yr5].max()), argmax_m_y5=int(np.argmax(m[yr5])))

out = {}
traces = {}
for hl, H, ml, M in combos:
    key = f'h_{hl}_m_{ml}'
    res = []; times = []
    for seed in (seeds if 'abm' in (hl, ml) else [0]):
        np.random.seed(seed)
        sim = mh.Sim(pars=dict(a=0.24, Nh=Nh, start=0, stop=365*5, dt=1, rand_seed=seed, verbose=0), human=H(**hum_pars), mosquito=M(**mos_pars))
        T = sc.tic(); sim.run(); times.append(sc.toc(T, output=True))
        res.append(summarize(sim))
        if seed == 0:
            traces[key+'_prev_h'] = np.array(sim.human.results.prev_h)[::5]
            traces[key+'_prev_m'] = np.array(sim.mosquito.results.prev_m)[::5]
    agg = {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res]))) for k in res[0]}
    out[key] = dict(n=len(res), mean_sd=agg, time_s=times)
    print(key, 'n=%d' % len(res), {k: '%.4f±%.4f' % v for k, v in agg.items()}, 'time %.2f s (min of %d)' % (min(times), len(times)))
sc.savejson(sc.thispath(__file__)/'hybrid_v3_summary.json', out)
import pandas as pd; pd.DataFrame(traces).to_csv(sc.thispath(__file__)/'hybrid_v3_traces.csv', index=False, float_format='%.5f')
