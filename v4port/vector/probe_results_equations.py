"""Declared derived results for the vector model (EIR, mosquito prevalence, m), equations(), and the count notes in describe()"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from ms_mosquito_v4 import emergence
Sh, Ih = ss.states(2); Sv, Ev, Iv = ss.states(3)
N = (Sh + Ih).n
em = ss.timeseries({k*365.25/12: v for k, v in enumerate(emergence + emergence[:1])}, mode='linear', period=ss.years(1))
mal = ss.Disease(infection=Sh >> Ih @ ss.perday(0.15) * Iv.n / N, recovery=Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
                 emergence=ss.null >> Sv @ ss.perday(1) * em * N, biting=Sv >> Ev @ ss.perday(0.15) * Ih.n / N,
                 incubation=Ev >> Iv @ ss.perday(1/12), mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(Ih=0.01, Sv=250),
                 eir=365.25 * 0.3 * Iv.n / N, prev_m=Iv / (Sv + Ev + Iv), m=(Sv + Ev + Iv) / (Sh + Ih), name='malaria')
for solver in ['ode', 'abm']:
    pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
    try:
        sim = ss.Sim(mal, solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop).run()
        r = sim.results.malaria
        manual = 365.25*0.3*np.array(r.n_Iv)/(np.array(r.n_Sh) + np.array(r.n_Ih))
        print(f'{solver}: eir y3 {np.mean(r.eir[731:]):.3f} (by hand {manual[731:].mean():.3f}); prev_m {np.mean(r.prev_m[731:]):.4f}; m {np.mean(r.m[731:]):.4f}; annual eir resampled {np.round(r.eir.resample("year").values, 2).tolist() if hasattr(r.eir, "resample") else None}')
    except Exception as e:
        print(solver, type(e).__name__, str(e)[:400])
try: print(mal.equations())
except Exception as e: print('equations()', type(e).__name__, str(e)[:300])
try: print('R0()', mal.R0())
except Exception as e: print('R0()', type(e).__name__, str(e)[:300])
