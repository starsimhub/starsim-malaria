import starsim as ss; print(ss.__version__, ss.__file__)
import sciris as sc, numpy as np, malariasim as ms
from human_v4 import make_human
from scenarios import scenarios
for scen in ['A', 'B', 'C']:
    s = scenarios[scen]
    age_data = sc.dataframe(ms.make_age_data())
    d = make_human('seis', eir=s['eir'], pops=s['pops'], shape=s.get('shape', 0), age_ages=s.get('age_ages'), age_vals=s.get('age_vals'), age_data=age_data)
    sim = ss.Sim(d, people=ss.People(sum(s['pops']), age_data=age_data), start=2000, stop=2005, dt=ss.days(1), verbose=0, rand_seed=0)
    T = sc.timer(); sim.run(); print(scen, 'run', T.toc(output=True))
    r = sim.results.malaria
    print(' keys', list(r.keys()))
    if scen != 'A':
        nI = r.n_I.by('patch'); print(' last-year prev by node', np.round(nI.values[-365:].sum(0)/(np.array(s['pops'])*365), 4))
    else:
        print(' last-year prev', r.n_I[-365:].sum()/(365*10000))
    if scen == 'C': sim.describe()
