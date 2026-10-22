"""
malariasim's mosquito module (MosquitoesODE / MosquitoesAdultFemaleAgentsOnly, one node, no interventions),
declared in v4 as states that are not people, coupled to a minimal SIS human (MalariaHumanSIS's defaults:
lognormal infection, mean and sd 200 days). Same scenario as ms_v3_ref.py.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc, pandas as pd

emergence = [0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005] # New females per human per day, by month (malariasim's default)

def make_malaria(a=0.3, b=0.5, c=0.5, life=10, eip=12, init_prev=0.01, eip_fixed=False):
    Sh, Ih     = ss.states(2)
    Sv, Ev, Iv = ss.states(3)
    N = (Sh + Ih).n
    em = ss.timeseries({k*365.25/12: v for k, v in enumerate(emergence + emergence[:1])}, mode='linear', period=ss.years(1)) # Linear between month starts, wrapping
    malaria = ss.Disease(
        infection  = Sh >> Ih @ ss.perday(a*b) * Iv.n / N,                       # EIR x b
        recovery   = Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
        emergence  = ss.null >> Sv @ ss.perday(1) * em * N,                       # per human per day
        biting     = Sv >> Ev @ ss.perday(a*c) * Ih.n / N,
        incubation = Ev >> Iv @ (ss.constant(ss.days(eip)) if eip_fixed else ss.perday(1/eip)),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.perday(1/life),
        init       = dict(Ih=init_prev, Sv=emergence[0]*life*5000),               # Disease-free equilibrium in January
        name       = 'malaria',
    )
    return malaria

def run(solver, seed=0, n=5000, **kw):
    pop = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    sim = ss.Sim(make_malaria(**kw), solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, rand_seed=seed, verbose=0, **pop)
    T = sc.tic(); sim.run(); t = sc.toc(T, output=True)
    return sim, t

def summarize(sim):
    r = sim.results.malaria
    prev = (np.array(r.n_Ih)/(np.array(r.n_Sh) + np.array(r.n_Ih)))[1:] # v4's index k is the end of step k; drop the start so that index k is v3's
    M = (np.array(r.n_Sv) + np.array(r.n_Ev) + np.array(r.n_Iv))[1:]; Iv = np.array(r.n_Iv)[1:]
    N = (np.array(r.n_Sh) + np.array(r.n_Ih))[1:]
    eir = 0.3*Iv/N*365.25
    y3 = slice(2*365, 3*365)
    return dict(prev_y1_mean=float(prev[:365].mean()), prev_y3_mean=float(prev[y3].mean()), prev_y3_max=float(prev[y3].max()), prev_y3_min=float(prev[y3].min()),
                M_y3_mean=float(M[y3].mean()), M_y3_max=float(M[y3].max()), Iv_y3_mean=float(Iv[y3].mean()), eir_y3_mean=float(eir[y3].mean()), M_day0=float(M[0]), M_day30=float(M[30]))

if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out, traces = {}, {}
    run('abm', seed=99); run('ode') # Warm-up
    for label, solver, kw in [('ode', 'ode', {}), ('abm', 'abm', {}), ('ode_fixed_eip', 'ode', dict(eip_fixed=True)), ('abm_fixed_eip', 'abm', dict(eip_fixed=True))]:
        res, times = [], []
        for seed in (range(n_seeds) if solver == 'abm' else [0]):
            sim, t = run(solver, seed=seed, **kw)
            res.append(summarize(sim)); times.append(t)
            if seed == 0 and label in ['ode', 'abm']:
                r = sim.results.malaria
                traces[f'{label}_prev'] = (np.array(r.n_Ih)/(np.array(r.n_Sh) + np.array(r.n_Ih)))[1::10]
                traces[f'{label}_M'] = (np.array(r.n_Sv) + np.array(r.n_Ev) + np.array(r.n_Iv))[1::10]
        agg = {k: (float(np.mean([x[k] for x in res])), float(np.std([x[k] for x in res]))) for k in res[0]}
        out[label] = dict(n=len(res), mean_sd=agg, time_s=times)
        print(label, {k: '%.4g±%.2g' % v for k, v in agg.items()}, 'times', np.round(times, 2).tolist())
    sc.savejson(sc.thispath(__file__)/'ms_mosquito_v4.json', out)
    pd.DataFrame(traces).to_csv(sc.thispath(__file__)/'ms_mosquito_v4_traces.csv', index=False, float_format='%.5g')
