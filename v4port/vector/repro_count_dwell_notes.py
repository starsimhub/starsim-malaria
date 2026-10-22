"""What each solver says about a fixed EIP on a count (verbose=1)"""
import starsim as ss; print(ss.__version__, ss.__file__)
from repro_count_dwell import make
for solver in ['ode', 'binomial', 'abm']:
    print(f'--- {solver}')
    pop = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
    sim = ss.Sim(make(ss.constant(ss.days(12))), solver=solver, unit='day', start=0, stop=30, dt=1, verbose=1, **pop).run()
    print('Iv by day', [round(float(x)) for x in sim.results.mal.n_Iv[:31:3]])
