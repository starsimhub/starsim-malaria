""" Speed of the intervention scenarios, v3 malariasim against the declared v4 model, at 40,000 people: best of 3 after a warm-up """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
import scen
scen.POPS[:] = [20_000, 20_000]
scen.RM['pop'] = 40_000
v4 = ss.__version__.startswith('4')
if v4:
    import malaria_v4, malaria_rm_v4
    makers = dict(eir=lambda name: malaria_v4.make(name, 'abm', 1), eir_ode=lambda name: malaria_v4.make(name, 'ode', 1), rm=lambda name: malaria_rm_v4.make(name, 'abm', 1))
else:
    import run_v3, run_v3_rm, run_v3_ode
    makers = dict(eir=lambda name: run_v3.make(name, 1), eir_ode=lambda name: run_v3_ode.make(name), rm=lambda name: run_v3_rm.make(name, 1))
cases = [('eir', 'base'), ('eir', 'vax'), ('eir', 'itn'), ('eir', 'mda'), ('eir_ode', 'vax'), ('rm', 'base'), ('rm', 'itn')]
for kind, name in cases:
    makers[kind](name).run() # Warm-up
    times = []
    for rep in range(3):
        sim = makers[kind](name)
        t0 = sc.tic(); sim.run(); times.append(sc.toc(t0, output=True))
    print(f'{"v4" if v4 else "v3"} {kind:8s} {name:5s}: best {min(times):.2f} s (all {np.round(times, 2).tolist()})', flush=True)
