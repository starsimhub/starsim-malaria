import starsim as ss, sys
def make(pf, init):
    a, b, c = 0.3, 0.5, 0.5
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv = ss.states(3, people=False) if pf else ss.states(3)
    Nh = (Sh + Eh + Ih + Rh).n
    return ss.Disease(infection=Sh >> Eh @ a*b*Iv.n/Nh, latency=Eh >> Ih @ ss.days(12), recovery=Ih >> Rh @ ss.days(100),
        waning=Rh >> Sh @ ss.days(365), emergence=ss.null >> Sv @ ss.perday(5_000), biting=Sv >> Ev @ a*c*Ih.n/Nh,
        incubation=Ev >> Iv @ ss.days(10), mortality=Sv + Ev + Iv >> ss.null @ ss.days(10), init=init, name='malaria')
for pf in [False, True]:
    for init in [dict(Ih=0.01, Sv=50_000), dict(Sv=3000), dict(Sh=0.99, Ih=0.01, Sv=50_000)]:
        for solver in ['abm', 'ode']:
            kw = dict(n_agents=5000) if solver == 'abm' else dict(total_pop=5000)
            try:
                sim = ss.Sim(make(pf, init), unit='day', stop=365, solver=solver, verbose=0, rand_seed=2, **kw).run()
                r = sim.results.malaria
                print(f'people=False:{pf!s:5} {str(init):38s} {solver}: infections {r.infection[:].sum():7.0f}  n_Sh[-1] {r.n_Sh[-1]:7.0f}  n_Sv[-1] {r.n_Sv[-1]:8.0f}')
            except Exception as e:
                print(f'people=False:{pf!s:5} {str(init):38s} {solver}: {type(e).__name__}: {str(e)[:200]}')
