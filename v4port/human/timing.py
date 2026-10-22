""" Like-for-like timing of the human side, EIR-driven, scenario B (3 nodes, seasonal EIR), 5 years at daily steps: best of 3 after a warm-up.
Run under v3 (malariasim's MalariaHumanSIS/SEIS/ODE + MosquitoesEirOnly) or v4 (the declared human_v4.py), chosen by the Starsim on the path """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc, malariasim as ms
from scenarios import scenarios, years
v4 = ss.__version__.startswith('4')
s = scenarios['B']
age_data = sc.dataframe(ms.make_age_data())

def make(model, n):
    pops = list(np.array(s['pops'])*n//10_000)
    if v4:
        from human_v4 import make_human
        d = make_human(model if model != 'ode' else 'sis', eir=s['eir'], pops=pops)
        kw = dict(total_pop=sum(pops), solver='ode') if model == 'ode' else dict(people=ss.People(sum(pops), age_data=age_data), solver='abm')
        return ss.Sim(d, start=2000, stop=2000 + years, dt=ss.days(1), verbose=0, rand_seed=1, **kw)
    kw = dict(age_data=age_data) if model != 'ode' else {}
    return ms.Sim(human=model, mosquito='eir_only', nodes=ms.NodeGrid(populations=pops, eir=s['eir']), common_pars=ms.CommonParams(),
                  start=2000, stop=2000 + years, dt=ss.days(1), rand_seed=1, verbose=0, **kw)

configs = [('sis', 10_000), ('seis', 10_000), ('ode', 10_000), ('sis', 100_000)]
for model, n in configs:
    times = []
    for rep in range(4):
        sim = make(model, n)
        T = sc.timer(); sim.run(); times.append(T.toc(output=True))
    print(f'{"v4" if v4 else "v3"} {model:4s} n={n:>7,}: warm-up {times[0]:.2f} s, best of 3 {min(times[1:]):.2f} s, all {np.round(times, 2).tolist()}', flush=True)
