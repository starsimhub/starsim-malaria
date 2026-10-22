"""
Cost of a time series in a count's rate as agents: ss.timeseries.at() converts every value at every call (8 calls a step
for RK4), so the same seasonal emergence given at 13 points or at 366 daily points (same values) costs very differently.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
import ms_mosquito_v4 as m
em13 = {k*365.25/12: v for k, v in enumerate(m.emergence + m.emergence[:1])}
days = np.arange(366.0)
em366 = dict(zip(days.tolist(), np.interp(days, list(em13.keys()), list(em13.values())).tolist()))
def make(points):
    Sh, Ih = ss.states(2); Sv, Ev, Iv = ss.states(3); N = (Sh + Ih).n
    em = ss.timeseries(points, mode='linear', period=ss.years(1))
    return ss.Disease(infection=Sh >> Ih @ ss.perday(0.15) * Iv.n / N, recovery=Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
                      emergence=ss.null >> Sv @ ss.perday(1) * em * N, biting=Sv >> Ev @ ss.perday(0.15) * Ih.n / N,
                      incubation=Ev >> Iv @ ss.perday(1/12), mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(Ih=0.01, Sv=250), name='malaria')
for label, pts in [('13 points', em13), ('366 points', em366)] * 2:
    for solver in ['abm', 'ode']:
        pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        sim = ss.Sim(make(pts), solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop)
        T = sc.tic(); sim.run(); t = sc.toc(T, output=True)
        print(f'{label}, {solver}: {t:.2f} s; prev at end {sim.results.malaria.n_Ih[-1]/5000:.4f}')
