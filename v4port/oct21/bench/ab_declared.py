""" Declared malaria models on whichever v4 is on the path: build + run times after a warm-up. Usage: ab_declared.py <case> """
import sys, time, numpy as np
import starsim as ss
case = sys.argv[1]
which = ss.__file__.split('/')[-3]

def seis(n=10_000, years=2, solver='abm', seed=0, b=0.4):
    """ Human SEIS (agents or ODE) with mosquito counts, single node: as declared_seis.py """
    S, E, I = ss.states(3)
    Sv, Ev, Iv = [ss.State(nm, people=False) for nm in ['Sv', 'Ev', 'Iv']]
    Nh = (S + E + I).n
    a, c, em = ss.perday(0.3), 0.3, 0.3*n/10*0+ 3.0
    mal = ss.Disease(infection=S >> E @ a*b*Iv.n/Nh, onset=E >> I @ ss.normal(ss.days(12), ss.days(2)),
                     recovery=I >> S @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
                     emergence=ss.null >> Sv @ ss.perday(em*n/10), biting=Sv >> Ev @ a*c*I.n/Nh, incubation=Ev >> Iv @ ss.perday(1/12),
                     mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(I=0.15, Sv=em*n), name='malaria')
    kw = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    return ss.Sim(mal, start='2000-01-01', stop=ss.years(years), dt=ss.days(1), solver=solver, rand_seed=seed, verbose=0, **kw)

def timeit(make, reps):
    make().run() # Warm-up (compiles)
    builds, runs = [], []
    for i in range(reps):
        t0 = time.perf_counter(); sim = make(i); sim.init(); t1 = time.perf_counter(); sim.run(); t2 = time.perf_counter()
        builds.append(t1 - t0); runs.append(t2 - t1)
    return sim, np.array(builds), np.array(runs)

if case == 'seis_abm':
    sim, b, r = timeit(lambda i=0: seis(solver='abm', seed=i), 3)
elif case == 'seis_ode20':  # 20 repeated ODE runs with varied b, as a calibration would
    sim, b, r = timeit(lambda i=0: seis(solver='ode', b=0.3 + 0.01*i), 20)
elif case == 'pop_ode':     # 3 patches x 20 age groups, seasonal EIR per patch, declared births/deaths
    sys.path.insert(0, '.'); import mal_v4 as m
    sim, b, r = timeit(lambda i=0: m.make_sim(seed=i, solver='ode', years=5), 3)
elif case == 'pop_abm':
    sys.path.insert(0, '.'); import mal_v4 as m
    sim, b, r = timeit(lambda i=0: m.make_sim(seed=i, solver='abm', years=5), 3)
res = sim.results.malaria
prev = res.n_I[-1] / (sum(res[k][-1] for k in res.keys() if k in ['n_S', 'n_E', 'n_I']))
print(f'{which} {case}: build {np.median(b):.3f} s, run median {np.median(r):.3f} s (min {r.min():.3f}, total {r.sum():.2f}); prev end {float(prev):.4f}', flush=True)
