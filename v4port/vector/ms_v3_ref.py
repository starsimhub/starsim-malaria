"""
Reference runs of unchanged malariasim on Starsim 3: SIS humans (5,000 agents, one node) with the ODE and the
agent-based mosquito modules, default seasonal emergence, b = c = 0.5, 2000-2003, daily steps, 5 seeds.
Writes ms_v3_ref.json (summaries) and ms_v3_ref_traces.csv (every 10 days, seed 0).
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc, pandas as pd, malariasim as ms

def make(mosquito, seed, n=5000):
    return ms.Sim(human='sis', mosquito=mosquito, start=2000, stop=2003, dt=ss.days(1), nodes=ms.NodeGrid(populations=[n]), rand_seed=seed, verbose=0,
                  human_pars=dict(init_prev=ss.bernoulli(p=0.01), prob_human_infected_on_bite=0.5, prob_mosquito_infected_on_bite=0.5))

def summarize(sim):
    h = sim.diseases[0].results; m = sim.connectors[-1].results
    prev = np.array(h.n_infected)/np.array(sim.results.n_alive)
    y3 = slice(2*365, 3*365)
    return dict(prev_y1_mean=float(prev[:365].mean()), prev_y3_mean=float(prev[y3].mean()), prev_y3_max=float(prev[y3].max()), prev_y3_min=float(prev[y3].min()),
                M_y3_mean=float(np.mean(m.n_total[y3])), M_y3_max=float(np.max(m.n_total[y3])), Iv_y3_mean=float(np.mean(m.n_infectious[y3])),
                eir_y3_mean=float(np.mean(m.eir_annual[y3])), M_day0=float(m.n_total[0]), M_day30=float(m.n_total[30]))

if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out, traces = {}, {}
    make('ode', 99).run() # Warm-up
    for mosquito in ['ode', 'agents']:
        res, times = [], []
        for seed in range(n_seeds):
            sim = make(mosquito, seed)
            T = sc.tic(); sim.run(); times.append(sc.toc(T, output=True))
            res.append(summarize(sim))
            if seed == 0:
                traces[f'{mosquito}_prev'] = (np.array(sim.diseases[0].results.n_infected)/np.array(sim.results.n_alive))[::10]
                traces[f'{mosquito}_M'] = np.array(sim.connectors[-1].results.n_total)[::10]
        agg = {k: (float(np.mean([x[k] for x in res])), float(np.std([x[k] for x in res]))) for k in res[0]}
        out[mosquito] = dict(n=len(res), mean_sd=agg, time_s=times)
        print(mosquito, {k: '%.4g±%.2g' % v for k, v in agg.items()}, 'times', np.round(times, 2).tolist())
    sc.savejson(sc.thispath(__file__)/'ms_v3_ref.json', out)
    pd.DataFrame(traces).to_csv(sc.thispath(__file__)/'ms_v3_ref_traces.csv', index=False, float_format='%.5g')
