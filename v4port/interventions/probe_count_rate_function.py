""" Probe: ways to make mosquito mortality depend on bednet coverage, as agents, given that ss.modify on a count transition is ignored there """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

def make(spelling, solver='abm'):
    S, I = ss.states(2)
    Sv, Iv = [ss.State(n, people=False) for n in ['Sv', 'Iv']]
    a = ss.perday(0.3)
    extra = 1.0 # Doubling the death rate from day 50
    if spelling == 'rate function':
        mort = Sv + Iv >> ss.null @ (lambda sim: ss.perday(0.1*(2.0 if sim.ti >= 50 else 1.0)))
    elif spelling == 'par function * rate':
        mort = Sv + Iv >> ss.null @ ss.perday(0.1) * ss.par(lambda sim: 2.0 if sim.ti >= 50 else 1.0)
    elif spelling == 'timeseries * rate':
        mort = Sv + Iv >> ss.null @ ss.perday(0.1) * ss.timeseries({'2000-01-01': 1.0, '2000-02-20': 2.0}, mode='constant')
    elif spelling == 'count of holders':
        mort = Sv + Iv >> ss.null @ ss.perday(0.1) * (1 + I.frac)   # a rate in terms of a fraction of people: works if coverage were a state
    mal = ss.Disease(infection=S >> I @ a * Iv.n / (S + I).n, recovery=I >> S @ ss.days(100), emergence=ss.null >> Sv @ ss.perday(400),
                     biting=Sv >> Iv @ a * I.n / (S + I).n, mortality=mort, init=dict(I=0.2, Sv=4000), name='mal')
    kw = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
    sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(100), dt=ss.days(1), solver=solver, verbose=0, rand_seed=1, **kw).run()
    r = sim.results.mal
    return r.n_Sv[-1] + r.n_Iv[-1]

for spelling in ['rate function', 'par function * rate', 'timeseries * rate', 'count of holders']:
    for solver in ['abm', 'ode']:
        try:
            print(f'{spelling:20s} {solver}: mosquitoes at day 100 {make(spelling, solver):7.0f} (4000 unchanged; about 2000 if doubled)')
        except Exception as e:
            print(f'{spelling:20s} {solver}: {type(e).__name__}: {str(e)[:300]}')
