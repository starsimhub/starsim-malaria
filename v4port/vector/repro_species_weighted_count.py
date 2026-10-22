"""
Can the EIR weight each species' infectious mosquitoes by its own biting rate, a_s * Iv_s, when the mosquitoes are
stratified by species? Tries ss.by(species=...) on the human infection rate. Reference (two sets of states): prev 0.4962 (ode).
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from ms_mosquito_v4 import emergence
def make(weighted):
    Sh, Ih = ss.states(2); Sv, Ev, Iv = ss.states(3)
    N = (Sh + Ih).n
    em = ss.timeseries({k*365.25/12: v for k, v in enumerate(emergence + emergence[:1])}, mode='linear', period=ss.years(1))
    a = ss.by(species=[0.3, 0.15]) if weighted else 0.3
    mal = ss.Disease(infection=Sh >> Ih @ ss.perday(0.5) * a * Iv.n / N, recovery=Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
                     emergence=ss.null >> Sv @ ss.perday(1) * em * N, biting=Sv >> Ev @ ss.perday(0.15) * Ih.n / N,
                     incubation=Ev >> Iv @ ss.perday(1/12), mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(Ih=0.01, Sv=250), name='malaria')
    mal.stratify(species=['gambiae', 'funestus'], split=[2/3, 1/3], states=[Sv, Ev, Iv], adjust={'emergence': [1.5, 1.5], 'mortality': [1, 2], 'biting': [1, 0.5]})
    return mal
for weighted in [False, True]:
    try:
        sim = ss.Sim(make(weighted), solver='ode', unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, total_pop=5000).run()
        r = sim.results.malaria; prev = np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih))
        print(f'a = {"ss.by(species=[0.3, 0.15])" if weighted else "0.3"}: prev y3 {prev[731:].mean():.4f}')
    except Exception as e:
        print(f'a = ss.by(species=...): {type(e).__name__}: {str(e)[:500]}')
# What the model says it does, and the same with a = ss.by(species=[0.15, 0.3]) (the order reversed)
mal = make(True)
sim = ss.Sim(mal, solver='ode', unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, total_pop=5000).run()
print([l for l in str(mal.describe(output=True) if 'output' in mal.describe.__code__.co_varnames else '').splitlines() if 'infection' in l or 'species' in l][:8])
def make_rev():
    m = make(True); return m
import ms_mosquito_v4
for vals in [[0.15, 0.3], [3.0, 3.0], [0.0, 0.0]]:
    Sh, Ih = ss.states(2); Sv, Ev, Iv = ss.states(3)
    N = (Sh + Ih).n
    em = ss.timeseries({k*365.25/12: v for k, v in enumerate(emergence + emergence[:1])}, mode='linear', period=ss.years(1))
    mal = ss.Disease(infection=Sh >> Ih @ ss.perday(0.5) * ss.by(species=vals) * Iv.n / N, recovery=Ih >> Sh @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
                     emergence=ss.null >> Sv @ ss.perday(1) * em * N, biting=Sv >> Ev @ ss.perday(0.15) * Ih.n / N,
                     incubation=Ev >> Iv @ ss.perday(1/12), mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(Ih=0.01, Sv=250), name='malaria')
    mal.stratify(species=['gambiae', 'funestus'], split=[2/3, 1/3], states=[Sv, Ev, Iv], adjust={'emergence': [1.5, 1.5], 'mortality': [1, 2], 'biting': [1, 0.5]})
    for solver in ['ode', 'abm']:
        pop = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
        r = ss.Sim(mal, solver=solver, unit='day', start='2000-01-01', stop='2003-01-01', dt=1, verbose=0, **pop).run().results.malaria
        prev = np.array(r.n_Ih)/(np.array(r.n_Sh)+np.array(r.n_Ih))
        print(f'a = ss.by(species={vals}), {solver}: prev y3 {prev[731:].mean():.4f}')
