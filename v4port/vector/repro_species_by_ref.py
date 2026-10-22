"""References for repro_species_weighted_count.py: the explicit per-species weighting, and plain a = 0.15, 0.2, 0.225, 0.25"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from ms_mosquito_v4 import emergence
def make(a_inf):
    Sh, Ih = ss.states(2); S1, E1, I1, S2, E2, I2 = ss.states(6)
    N = (Sh + Ih).n
    em = ss.timeseries({k*365.25/12: v for k, v in enumerate(emergence + emergence[:1])}, mode='linear', period=ss.years(1))
    return ss.Disease(infection=Sh >> Ih @ ss.perday(0.5) * (a_inf[0]*I1.n + a_inf[1]*I2.n) / N, recovery=Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
        emergence1=ss.null >> S1 @ ss.perday(1.0) * em * N, emergence2=ss.null >> S2 @ ss.perday(0.5) * em * N,
        biting1=S1 >> E1 @ ss.perday(0.15) * Ih.n / N, biting2=S2 >> E2 @ ss.perday(0.075) * Ih.n / N,
        incubation1=E1 >> I1 @ ss.perday(1/12), incubation2=E2 >> I2 @ ss.perday(1/12),
        mortality1=S1 + E1 + I1 >> ss.null @ ss.perday(0.1), mortality2=S2 + E2 + I2 >> ss.null @ ss.perday(0.2), init=dict(Ih=0.01, S1=250, S2=62.5), name='malaria')
for a_inf in [(0.15, 0.3), (0.15, 0.15), (0.2, 0.2), (0.225, 0.225), (0.25, 0.25)]:
    for solver in ['ode', 'abm']:
        pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        r = ss.Sim(make(a_inf), solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop).run().results.malaria
        prev = np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih))
        print(f'explicit species, EIR weights {a_inf}, {solver}: prev y3 {prev[731:].mean():.4f}')
