""" Under the ODE, a rate fn(sim) returning one value per patch works with a patch dimension alone. What if the model also has a biting dimension? """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from human_v4 import gamma_levels
eir = np.array([5.0, 50.0])
for extra in [False, True]:
    S, I = ss.states('S I')
    d = ss.Disease(infection=S >> I @ (lambda sim: ss.peryear(0.4*eir)), recovery=I >> S @ ss.days(200), init=dict(I=0.0), name='malaria')
    d.stratify(patch=['n0', 'n1'], split=[0.5, 0.5])
    if extra: d.stratify(biting=[f'b{k}' for k in range(10)], split=[0.1]*10, adjust={'infection': list(gamma_levels(1.0, 10))})
    try:
        sim = ss.Sim(d, start=2000, stop=2003, dt=ss.days(1), solver='ode', verbose=0, total_pop=10_000).run()
        print(f'OK   biting dimension={extra}: final prevalence by patch', np.round(sim.results.malaria.n_I.by('patch').values[-1]/5000, 4))
    except Exception as e:
        print(f'FAIL biting dimension={extra}: {type(e).__name__}: {str(e)[:300]}')
