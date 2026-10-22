""" Probe: ss.treat on the EIR model: default eligibility, with transition=, with where=mal.I; and lognormal dwell syntax """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from scen import *
import malaria_v4 as mv

for label, kw in [('default', dict()), ('where=I', dict(where='I'))]:
    for solver in ['abm', 'ode']:
        mal = mv.make_disease()
        drug = ss.Treatment(cure=1.0, transition='recovery')
        where = mal.I if kw.get('where') else None
        intv = ss.treat(drug, coverage=DRUG['cov'], at=[ss.days(T_INTV)], where=where, name='mda')
        sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(DAYS), dt=ss.days(1), solver=solver, rand_seed=1, interventions=intv, verbose=0, **(dict(n_agents=4000) if solver=='abm' else dict(total_pop=4000)))
        sim.run()
        r = sim.results
        print(f'{label:8s} {solver}: delivered {r.mda.delivered.values.sum():.0f}, eligible {r.mda.eligible.values.sum():.0f}; n_I at days 364, 366, 372: {np.round(r.malaria.n_I.values[[365, 367, 373]]).tolist()}; recovery flow on campaign step {r.malaria.recovery.values[366]:.0f}')
print('recovery flow at points 364..368 (last sim):', r.malaria.recovery.values[364:369], 'n_I', r.malaria.n_I.values[364:369])
