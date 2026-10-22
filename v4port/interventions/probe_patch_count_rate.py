""" Probe: mosquito mortality per patch from a function of the sim (node-targeted killing), as agents and as an ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
for spelling in ['by(patch) par', 'function returning one value per patch']:
    for solver in ['abm', 'ode']:
        try:
            S, I = ss.states(2)
            Sv, Iv = [ss.State(n, people=False) for n in ['Sv', 'Iv']]
            a = ss.perday(0.3)
            if spelling == 'by(patch) par':
                mult = ss.by(patch=[2.0, 1.0])
            else:
                mult = ss.par(lambda sim: np.array([2.0, 1.0]), name='kill')
            mal = ss.Disease(infection=S >> I @ a * Iv.n / (S + I).n, recovery=I >> S @ ss.days(100), emergence=ss.null >> Sv @ ss.perday(400),
                             biting=Sv >> Iv @ a * I.n / (S + I).n, mortality=Sv + Iv >> ss.null @ ss.perday(0.1) * mult, init=dict(I=0.2, Sv=4000), name='mal')
            mal.stratify(patch=['n0', 'n1'], split=[0.5, 0.5], kind='space')
            kw = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
            sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(100), dt=ss.days(1), solver=solver, verbose=0, rand_seed=1, **kw).run()
            r = sim.results.mal
            print(f'{spelling:40s} {solver}: mosquitoes by patch at day 100 {np.round(r.n_Sv.by("patch").values[-1] + r.n_Iv.by("patch").values[-1]).tolist()} (expected [1000, 2000])')
        except Exception as e:
            print(f'{spelling:40s} {solver}: {type(e).__name__}: {str(e)[:300]}')
