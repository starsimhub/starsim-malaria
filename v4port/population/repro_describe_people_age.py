""" Repro: ss.People(5000, age_data=...) with an age stratification. The ages are used under both solvers, but describe() calls them a guess as agents and crashes as an ODE """
import traceback
import numpy as np, pandas as pd, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
ages = pd.DataFrame({'age': np.arange(100), 'value': np.exp(-np.arange(100)/20)}) # A young population: 22% under 5
w = ages.value.values; print('given shares by age group:', np.round([w[:5].sum(), w[5:15].sum(), w[15:50].sum(), w[50:].sum()]/w.sum(), 3))
for solver in ['abm', 'ode']:
    sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(0.5)), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.1, name='sis')
    sim = ss.Sim(sis, people=ss.People(5000, age_data=sc.dataframe(ages)), stratify=dict(age=[0, 5, 15, 50]), solver=solver,
                 demographics=[ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.05))], start=2000, stop=2001, verbose=0).run()
    a = sim.results.sis.n_S.by('age').values[0] + sim.results.sis.n_I.by('age').values[0]
    print(f'{solver}: shares by age group at the start {np.round(a/a.sum(), 3)}')
    try:
        txt = str(sim.describe())
        print('   ' + '\n   '.join(l.strip() for l in txt.splitlines() if 'people' in l.lower() and ('guess' in l or 'Sim:' in l) or l.startswith('dimension') or ('agents' in l and 'guess' in l)))
    except Exception as e:
        print(f'   describe(): {type(e).__name__}: {e}  ({traceback.extract_tb(e.__traceback__)[-1].filename.split("/")[-1]}:{traceback.extract_tb(e.__traceback__)[-1].lineno})')
