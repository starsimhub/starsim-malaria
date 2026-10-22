""" Do agents keep malariasim's ages (ss.People(age_data=...)) when the disease is stratified by age, and what does describe() say? """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc, malariasim as ms
from human_v4 import make_human
age_data = sc.dataframe(ms.make_age_data())
a = np.asarray(age_data['age'], float) + 0.5; w = np.asarray(age_data['value'], float)
print('malariasim age data: mean age', round((a*w).sum()/w.sum(), 2), 'share under 5', round(w[a < 5].sum()/w.sum(), 3))
d = make_human('sis', age_ages=[0, 5, 15], age_vals=[0.5, 0.8, 1.0], age_data=age_data)
sim = ss.Sim(d, people=ss.People(10_000, age_data=age_data), start=2000, stop=2001, dt=ss.days(1), verbose=0).init()
age = sim.people.age[sim.people.auids]
print('v4 agents after init: mean age', round(age.mean(), 2), 'share under 5', round((age < 5).mean(), 3))
d2 = make_human('sis', age_ages=[0, 5, 15], age_vals=[0.5, 0.8, 1.0], age_data=age_data)
sim2 = ss.Sim(d2, n_agents=10_000, start=2000, stop=2001, dt=ss.days(1), verbose=0).init()
age2 = sim2.people.age[sim2.people.auids]
print('v4 agents with n_agents= and split= on the age dimension: mean age', round(age2.mean(), 2), 'share under 5', round((age2 < 5).mean(), 3))
with sc.capture() as txt:
    sim.describe()
print([l for l in txt.splitlines() if 'guess' in l])
