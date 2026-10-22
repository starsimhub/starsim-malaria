""" v3 reference for the ODE: malariasim's MalariaHumanODE (3 nodes x 20 age bins), driven by the same per-node seasonal EIR, 5 years daily.
Prevalence by node, time (best of 3 after a warm-up), and the death and birth rates it ends up with from make_steady_state_demographics() """
import time
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
import malariasim as ms
from common import *
def make(years=5):
    nodes = ms.NodeGrid(populations=POPS, labels=LABELS, eir=node_eir_profiles(), f_indoor=F_INDOOR)
    return ms.Sim(human='ode', mosquito='eir_only', common_pars=ms.CommonParams(), age_data=sc.dataframe(ms.make_age_data()), age_bin_edges=ms.AGE_BIN_EDGES,
                  start=START, stop=START+years, dt=ss.days(1), nodes=nodes, demographics=ms.make_steady_state_demographics(), analyzers=ms.ResultsAnalyzer(), verbose=0)
ts = []
for rep in range(4):
    sim = make(); T = time.perf_counter(); sim.run(); ts.append(time.perf_counter() - T)
print(f'v3 MalariaHumanODE, 5 y daily: best of 3 after warm-up {min(ts[1:]):.2f}s ({1e3*min(ts[1:])/1826:.3f} ms/step)')
d = sim.diseases[0]
print('death_rate_per_year given to the ODE:', d.pars.death_rate_per_year, '; birth_rate_per_year:', d.pars.birth_rate_per_year)
print('per-bin death rates it solved (first 4):', np.round(d._death_rates[:4], 4), '; birth rate', round(float(d._birth_rate), 5))
sim = make(years=20); sim.run()
az = sim.analyzers[0]
nT = len(sim.results.timevec); last5 = slice(nT - 5*365, nT)
print('prev by node, last 5 of 20 years:', np.round([np.mean(az.node_infected_hist[l][last5]/az.node_hist[l][last5]) for l in LABELS], 4))
print('N start -> end', az.hist['Total'][0], az.hist['Total'][-1])
