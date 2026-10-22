""" Probe: deaths by single year of age while results are by 5-year group. (1) ss.by(age=) with 100 values on a 20-group age dimension;
(2) a function of age; (3) declared ss.birth with malariasim's ss.Deaths(table) module """
import numpy as np, pandas as pd, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m
mu, n, br, max_age = compute_mortality_schedule()
df = pd.DataFrame({'Time': 2000, 'AgeGrpStart': np.arange(max_age), 'mx': mu})
deaths_module = lambda: ss.Deaths(death_rate=df, rate_units=1, metadata={'data_cols': dict(year='Time', age='AgeGrpStart', value='mx')})
def mu_of_age(sim, uids):
    return mu[np.clip(sim.people.age[uids].astype(int), 0, 99)]
cases = {
    'by(age=) with 100 values': lambda: [ss.birth(rate=ss.peryear(br)), ss.death(rate=ss.by(age=[ss.peryear(x) for x in mu]))],
    'function of age (per year)': lambda: [ss.birth(rate=ss.peryear(br)), ss.death(rate=ss.peryear(mu_of_age))],
    'ss.birth + ss.Deaths(table) module': lambda: [ss.birth(rate=ss.peryear(br)), deaths_module()],
}
for label, demog in cases.items():
    rows = []
    try:
        for seed in range(6):
            sim = ss.Sim(m.make_malaria(), demographics=demog(), stratify=dict(age=AGE_BIN_EDGES[:-1]), people=ss.People(5000, age_data=sc.dataframe(m.age_data())), start=START, stop=STOP, dt=ss.days(DT_DAYS), rand_seed=seed, verbose=0).run()
            res = sim.results.malaria
            bya = res.people.by('age').values
            rows.append([res.people.values[-1], tvd(bya[0]/bya[0].sum(), bya[-1]/bya[-1].sum())] + list(res.people.by('patch').values[-1]))
        r = np.array(rows)
        print(f'{label}: N end {r[:,0].mean():.0f} +/- {r[:,0].std()/np.sqrt(len(r)):.0f}; pyramid TVD {r[:,1].mean():.3f}; patches end {r[:,2:].mean(0).round(0)}')
    except Exception as e:
        print(f'{label}: {type(e).__name__}: {str(e)[:500]}')
