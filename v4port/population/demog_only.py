""" Demography alone (no disease): malariasim's steady-state ss.Births + ss.Deaths(table), and FastBirths/FastDeaths, on v3 and v4 """
import sys, time
import numpy as np, pandas as pd, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *

N = 5000
def plain_demog():
    """ malariasim.defaults.make_steady_state_demographics, verbatim """
    mu, _, birth_rate, max_age = compute_mortality_schedule()
    births = ss.Births(birth_rate=ss.peryear(birth_rate * 1000))
    df = pd.DataFrame({'Time': 2000, 'AgeGrpStart': np.arange(max_age), 'mx': mu})
    metadata = {'data_cols': dict(year='Time', age='AgeGrpStart', value='mx')}
    return [births, ss.Deaths(death_rate=df, rate_units=1, metadata=metadata)]

def age_data():
    _, n, _, max_age = compute_mortality_schedule()
    return pd.DataFrame({'age': np.arange(max_age, dtype=int), 'value': n})

def make(seed, mode, n=N, years=STOP-START):
    if mode == 'fast':
        import v3_fast_demog as fd
        demog = fd.make_steady_state_demographics()
    else:
        demog = plain_demog()
    return ss.Sim(people=ss.People(n, age_data=sc.dataframe(age_data())), demographics=demog, start=START, stop=START+years, dt=ss.days(DT_DAYS), rand_seed=seed, verbose=0)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plain'
    out = []
    for seed in SEEDS:
        sim = make(seed, mode)
        sim.init()
        pyr0 = pyramid(sim.people.age.values); a0 = sim.people.age.values.mean()
        T = time.perf_counter(); sim.run(); el = time.perf_counter() - T
        pyr1 = pyramid(sim.people.age.values)
        b = sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Births))
        d = sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Deaths))
        out.append([sim.results.n_alive[-1], a0, sim.people.age.values.mean(), tvd(pyr0, pyr1), b, d, el])
        print(f'seed {seed}: n_end {sim.results.n_alive[-1]:.0f} mean age {a0:.2f}->{sim.people.age.values.mean():.2f} tvd {tvd(pyr0, pyr1):.3f} births {b:.0f} deaths {d:.0f} {el:.2f}s')
    out = np.array(out, float)
    for k, m, s in zip(['n_end','age0','age1','tvd','births','deaths','time'], out.mean(0), out.std(0)):
        print(f'{k}: {m:.3f} +/- {s:.3f}')
    # Timing: best of 3 after warm-up, 20k agents, 20 years
    times = []
    for rep in range(4):
        sim = make(0, mode, n=20_000); T = time.perf_counter(); sim.run(); times.append(time.perf_counter() - T)
    print(f'timing 20k agents x 20 y weekly: best of 3 after warm-up {min(times[1:]):.2f}s  (all {np.round(times,2)})')
