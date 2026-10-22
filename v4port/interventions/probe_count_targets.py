""" Probe: more ways to act on the mosquito counts: ss.change on the emergence inflow, on a named parameter in a count rate, and a product's efficacy keyed by a count transition """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

def make(kind, solver):
    S, I = ss.states(2)
    Sv, Iv = [ss.State(n, people=False) for n in ['Sv', 'Iv']]
    a = ss.perday(0.3)
    em = ss.par(ss.perday(400), name='em')
    mal = ss.Disease(infection=S >> I @ a * Iv.n / (S + I).n, recovery=I >> S @ ss.days(100), emergence=ss.null >> Sv @ em,
                     biting=Sv >> Iv @ a * I.n / (S + I).n, mortality=Sv + Iv >> ss.null @ ss.days(10), init=dict(I=0.2, Sv=4000), name='mal')
    if kind == 'none':
        intv = None
    elif kind == 'change emergence':
        intv = ss.change(mal.emergence, at=[ss.days(50)], by=[0.5])
    elif kind == 'change pars.em':
        intv = ss.change('mal.pars.em', at=[ss.days(50)], by=[0.5])
    elif kind == 'product on mortality':
        intv = ss.vaccinate(ss.Vaccine(efficacy=dict(mortality=-1.0)), coverage=1.0, at=[ss.days(50)])  # 'mortality' means people's deaths
    elif kind == 'product on biting':
        intv = ss.vaccinate(ss.Vaccine(efficacy=dict(biting=1.0)), coverage=1.0, at=[ss.days(50)])      # holders give no infections to mosquitoes?
    kw = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
    sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(100), dt=ss.days(1), solver=solver, interventions=intv, verbose=0, rand_seed=1, **kw).run()
    r = sim.results.mal
    return r.n_Sv[-1] + r.n_Iv[-1], r.n_Iv[-1]

for kind in ['none', 'change emergence', 'change pars.em', 'product on mortality', 'product on biting']:
    for solver in ['abm', 'ode']:
        try:
            mos, iv = make(kind, solver)
            print(f'{kind:22s} {solver}: mosquitoes {mos:6.0f}, infectious {iv:6.0f}')
        except Exception as e:
            print(f'{kind:22s} {solver}: {type(e).__name__}: {str(e)[:300]}')
