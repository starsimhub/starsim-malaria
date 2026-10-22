"""
Unchanged malariasim (ms.Sim) on v3 and on v4 with the shim: do the numbers agree?

9 combinations (human sis/seis/ode x mosquito eir_only/ode/agents), 3 nodes (3000/2000/1000 humans,
target EIR 4/6/8, f_indoor 0.7), steady-state demographics, outbreak of 15% at day 30, 3 years, daily,
seeds 0-4. Reports the year-3 mean of human prevalence and of the annual EIR over nodes (and the
mosquito count). Usage: python compare_unchanged.py <outfile.json> [ncpus]
"""
import sys
import numpy as np
import sciris as sc
import starsim as ss
print(ss.__version__, ss.__file__)
V4 = ss.__version__.startswith('4')
if V4:
    import malshim # noqa
import malariasim as ms

START, YEARS = 2000, 3
POPS = [3000, 2000, 1000]
EIRS = [4.0, 6.0, 8.0]
CP = ms.CommonParams()
PROFILES = [ms.eir_to_params(e, common_pars=CP, seasonal_profile=None) for e in EIRS]

def campaign(sim, eir_only=False):
    c = ms.Campaign(sim)
    c.add(ms.InterventionDistributorScheduled(name='Outbreak', products=ms.Outbreak(), start=START + 30/365, coverage=0.15))
    return c

def build(human, mosquito, seed):
    nodes = ms.NodeGrid(populations=POPS, labels=['A', 'B', 'C'], eir=[list(p[0]) for p in PROFILES],
                        emergence=[list(p[1]) for p in PROFILES], f_indoor=[0.7]*3)
    age_data = sc.dataframe(ms.make_age_data(edges=ms.AGE_BIN_EDGES, fractions=ms.AGE_BIN_FRACTIONS))
    return ms.Sim(human=human, mosquito=mosquito, nodes=nodes, age_data=age_data, age_bin_edges=ms.AGE_BIN_EDGES,
                  start=START, stop=START + YEARS, dt=ss.days(1), rand_seed=seed, demographics=ms.make_steady_state_demographics(),
                  common_pars=CP, build_campaign=campaign, verbose=0)

def run_one(human, mosquito, seed):
    sim = build(human, mosquito, seed)
    T = sc.timer()
    sim.run()
    elapsed = T.toc(output=True)
    hum = sim.diseases[0].results
    ni = np.asarray(hum.n_infected.values, dtype=float)
    if human == 'ode':
        N = np.asarray(hum.n_susceptible.values, dtype=float) + ni
    else:
        N = np.asarray(sim.results.n_alive.values, dtype=float)
    last = slice(-365, None)
    with np.errstate(invalid='ignore', divide='ignore'):
        prev = np.nanmean((ni/N)[last][N[last] > 0])
    mos = [m for m in sim.connectors.values() if 'eir_annual' in m.results]
    eir = float(np.mean(mos[0].results.eir_annual.values[last])) if mos else np.nan
    ntot = float(np.mean(mos[0].results.n_total.values[last])) if mos and 'n_total' in mos[0].results else np.nan
    return dict(human=human, mosquito=mosquito, seed=seed, prev=float(prev), eir=eir, n_mos=ntot, time=elapsed, npts=len(ni))

if __name__ == '__main__':
    out = sys.argv[1]
    ncpus = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ss.options(crn=False) if False else None
    jobs = [dict(human=h, mosquito=m, seed=s) for h in ['sis', 'seis', 'ode'] for m in ['eir_only', 'ode', 'agents'] for s in range(5)]
    res = sc.parallelize(run_one, iterkwargs=jobs, ncpus=ncpus)
    sc.savejson(out, res)
    import pandas as pd
    df = pd.DataFrame(res)
    print(df.groupby(['human', 'mosquito'])[['prev', 'eir', 'n_mos', 'time']].agg(['mean', 'std']).round(4).to_string())
