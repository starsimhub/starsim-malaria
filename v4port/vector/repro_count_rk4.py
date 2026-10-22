"""
States that are not people are advanced in an agent-based run by one RK4 step per sim step, with no substeps,
so a count whose turnover in a step exceeds about 2.8 grows instead of settling. The spec's own mosquito
population (emergence 5,000 a day, lifespan 10 days) should settle at 50,000 whatever the step; it starts at 10,000.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
def make():
    Sh, Ih = ss.states(2)
    Sv = ss.State(people=False)
    N = (Sh + Ih).n
    return ss.Disease(infection=Sh >> Ih @ ss.perday(1e-6) * Sv.n / N, emergence=ss.null >> Sv @ ss.perday(5000),
                      mortality=Sv >> ss.null @ ss.days(10), init=dict(Sh=1.0, Sv=10_000), name='mal')
for dt in [1, 7, 14, 30]:
    out = []
    for solver in ['ode', 'abm']:
        pop = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        sim = ss.Sim(make(), solver=solver, unit='day', start=0, stop=360, dt=dt, verbose=0, **pop).run()
        Sv = np.asarray(sim.results.mal.n_Sv)
        out.append(f'{solver}: Sv at the end {Sv[-1]:,.0f} (max {Sv.max():,.3g})')
    print(f'dt={dt:>2} days:', '; '.join(out))
for dt in [21, 28, 30, 365/12]:
    sim = ss.Sim(make(), solver='abm', unit='day', start=0, stop=360, dt=dt, verbose=0, n_agents=2000).run()
    print(f'abm, dt={dt:.4g}: Sv by step', np.round(np.asarray(sim.results.mal.n_Sv)[:8]).tolist())
