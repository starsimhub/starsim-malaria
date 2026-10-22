"""
malaria_hybrid's Ross-Macdonald model declared in Starsim v4: SIS humans, SEI mosquitoes as
states that are not people, with malaria_hybrid's seasonal target and relaxation. One model,
run as an ODE (humans and mosquitoes counts) and as agents (humans agents, mosquitoes counts).
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc

eps = 1e-8
def seasonality(t, sn_amp=0.65, sn_phase=120, sn_peak=1.5, year_len=365):
    """ malaria_hybrid's seasonal multiplier, verbatim """
    s = np.maximum(1 + sn_amp*np.sin(2*np.pi*(t - sn_phase)/year_len), eps)**sn_peak
    tt = np.linspace(0, year_len, 2000)
    s_ref = np.maximum(1 + sn_amp*np.sin(2*np.pi*(tt - sn_phase)/year_len), eps)**sn_peak
    return s/s_ref.mean()

def make_malaria(Nh=10_000, a=0.24, b=0.15, r=1/120, Ih0=10, c=0.15, mu_m=0.10, sigma_m=1/12, m0=5, Im0=100, sn_relax=0.15):
    Sh, Ih     = ss.states(2)
    Sv, Ev, Iv = ss.states(3)
    N = (Sh + Ih).n
    M = (Sv + Ev + Iv).n
    days = np.arange(366)
    target = ss.timeseries(dict(zip(days.tolist(), (Nh*m0*seasonality(days)).tolist())), mode='linear', period=ss.days(365)) # M*(t)
    malaria = ss.Disease(
        infection  = Sh >> Ih @ ss.perday(a*b) * Iv.n / N,
        recovery   = Ih >> Sh @ ss.perday(r),
        emergence  = ss.null >> Sv @ ss.perday(sn_relax) * np.maximum(target - M, 0),           # births top up toward the target
        biting     = Sv >> Ev @ ss.perday(a*c) * Ih.n / N,
        incubation = Ev >> Iv @ ss.perday(sigma_m),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.perday(mu_m),
        culling    = Sv + Ev + Iv >> ss.null @ ss.perday(sn_relax) * np.maximum(M - target, 0) / np.maximum(M, eps), # extra loss above the target
        init       = dict(Ih=Ih0, Sv=Nh*m0 - Im0, Iv=Im0),
        name       = 'malaria',
    )
    return malaria

def run(solver, seed=0, Nh=10_000, **kw):
    pop = dict(n_agents=Nh) if solver == 'abm' else dict(total_pop=Nh)
    sim = ss.Sim(make_malaria(Nh=Nh), solver=solver, unit='day', start=0, stop=365*5, dt=1, rand_seed=seed, verbose=0, **pop, **kw)
    T = sc.tic(); sim.run(); t = sc.toc(T, output=True)
    return sim, t

def summarize(sim):
    r = sim.results.malaria
    Nh = np.array(r.n_Sh) + np.array(r.n_Ih); M = np.array(r.n_Sv) + np.array(r.n_Ev) + np.array(r.n_Iv)
    ph = np.array(r.n_Ih)/Nh; pm = np.array(r.n_Iv)/M; m = M/Nh
    yr5 = slice(4*365+1, 5*365+1) # v4 index k is the end of step k, so v3's index k is v4's k+1
    return dict(prev_h_y1=float(ph[366]), prev_h_y5_mean=float(ph[yr5].mean()), prev_h_y5_max=float(ph[yr5].max()), prev_h_y5_min=float(ph[yr5].min()),
                prev_m_y5_mean=float(pm[yr5].mean()), m_y5_mean=float(m[yr5].mean()), m_y5_max=float(m[yr5].max()), argmax_m_y5=int(np.argmax(m[yr5])))

if __name__ == '__main__':
    import sys
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {}; traces = {}
    for label, solver, kw in [('ode_rk4', 'ode', {}), ('ode_euler', 'ode', dict(integrator='euler')), ('abm', 'abm', {}), ('binomial', 'binomial', {})]:
        res, times = [], []
        for seed in (range(n_seeds) if solver in ['abm', 'binomial'] else [0]):
            sim, t = run(solver, seed=seed, **kw)
            res.append(summarize(sim)); times.append(t)
            if seed == 0:
                r = sim.results.malaria
                traces[label+'_prev_h'] = (np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih)))[1::5]
        agg = {k: (float(np.mean([x[k] for x in res])), float(np.std([x[k] for x in res]))) for k in res[0]}
        out[label] = dict(n=len(res), mean_sd=agg, time_s=times)
        print(label, 'n=%d' % len(res), {k: '%.4f±%.4f' % v for k, v in agg.items()}, 'times', np.round(times, 2).tolist())
    sc.savejson(sc.thispath(__file__)/'hybrid_v4_summary.json', out)
    import pandas as pd; pd.DataFrame({k: v[:min(len(x) for x in traces.values())] for k, v in traces.items()}).to_csv(sc.thispath(__file__)/'hybrid_v4_traces.csv', index=False, float_format='%.5f')
