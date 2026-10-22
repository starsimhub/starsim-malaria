""" First probe: a declared SIS driven by a constant external EIR, as agents and as an ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc

b = 0.4
def make():
    eir = ss.peryear(10) # infectious bites per person per year
    S, I = ss.states('S I')
    return ss.Disease(
        infection = S >> I @ b * eir,
        recovery  = I >> S @ ss.days(200),
        init = dict(I=0.0),
        name = 'malaria',
    )
for solver in ['abm', 'ode']:
    kw = dict(n_agents=10_000) if solver == 'abm' else {}
    sim = ss.Sim(make(), start=2000, stop=2005, dt=ss.days(1), total_pop=10_000, solver=solver, verbose=0, **kw)
    T = sc.timer()
    sim.run()
    r = sim.results.malaria
    print(solver, T.toc(output=True), list(r.keys())[:20])
    print('  prev end', r.n_I[-1]/10_000, 'eq', b*10/(b*10 + 365/200))
    if solver == 'abm': sim.describe()
