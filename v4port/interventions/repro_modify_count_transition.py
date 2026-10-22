""" Repro: ss.modify on a transition of states that are not people (mosquito mortality, mosquito infection) """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

def make(mult, target, solver, start=50):
    S, I = ss.states(2)
    Sv, Iv = [ss.State(n, people=False) for n in ['Sv', 'Iv']]
    a = ss.perday(0.3)
    mal = ss.Disease(
        infection = S >> I @ a * Iv.n / (S + I).n,
        recovery  = I >> S @ ss.days(100),
        emergence = ss.null >> Sv @ ss.perday(400),
        biting    = Sv >> Iv @ a * I.n / (S + I).n,
        mortality = Sv + Iv >> ss.null @ ss.days(10),
        init = dict(I=0.2, Sv=4000), name='mal')
    intv = None if mult is None else ss.modify(getattr(mal, target), multiply=mult, start=ss.days(start), name='mod')
    kw = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
    sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(100), dt=ss.days(1), solver=solver, interventions=intv, verbose=0, rand_seed=1, **kw).run()
    r = sim.results.mal
    return r.n_Sv[-1] + r.n_Iv[-1], r.n_Iv[-1], sim

for solver in ['abm', 'ode']:
    for target in ['mortality', 'biting']:
        for label, mult in [('none', None), ('2.0', 2.0), ('lambda: 2.0', lambda sim: 2.0)]:
            mos, iv, sim = make(mult, target, solver)
            print(f'{solver} multiply {target} by {label:12s}: mosquitoes at day 100 {mos:7.0f}, infectious {iv:6.0f}  (no change expected only for none)')
print(sim.what_modifies('mal.mortality'))
mos, iv, sim = make(2.0, 'mortality', 'abm')
print('abm:', sim.what_modifies('mal.mortality'))
