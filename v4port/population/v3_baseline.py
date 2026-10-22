""" v3 reference: unchanged malariasim, human SIS driven by per-node seasonal EIR (eir_only), 3 nodes, steady-state demographics """
import sys, json, time
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
import malariasim as ms
from common import *

def make_sim(seed, fast=False):
    nodes = ms.NodeGrid(populations=POPS, labels=LABELS, eir=node_eir_profiles(), f_indoor=F_INDOOR)
    if fast:
        import v3_fast_demog as fd
        demog = fd.make_steady_state_demographics()
    else:
        demog = ms.make_steady_state_demographics()
    return ms.Sim(human='sis', mosquito='eir_only', common_pars=ms.CommonParams(),
                  age_data=sc.dataframe(ms.make_age_data()), age_bin_edges=ms.AGE_BIN_EDGES,
                  start=START, stop=STOP, dt=ss.days(DT_DAYS), nodes=nodes, demographics=demog,
                  analyzers=ms.ResultsAnalyzer(), rand_seed=seed, verbose=0)

def summarize(sim):
    az = sim.analyzers[0]
    d = sim.diseases[0]
    bins = az.labels[:-1]
    pyr0 = np.array([az.hist[b][0] for b in bins], float); pyr1 = np.array([az.hist[b][-1] for b in bins], float)
    nT = len(sim.results.timevec); last5 = slice(int(nT - 5*365/DT_DAYS), nT)
    out = dict(
        n0 = float(sim.results.n_alive[0]), n1 = float(sim.results.n_alive[-1]),
        node0 = [float(az.node_hist[l][0]) for l in LABELS], node1 = [float(az.node_hist[l][-1]) for l in LABELS],
        prev_node = [float(np.mean(az.node_infected_hist[l][last5]/az.node_hist[l][last5])) for l in LABELS],
        prev = float(np.mean(sim.results.malariahumansis.n_infected[last5]/sim.results.n_alive[last5])),
        pyr0 = list(pyr0/pyr0.sum()), pyr1 = list(pyr1/pyr1.sum()), tvd = float(tvd(pyr0/pyr0.sum(), pyr1/pyr1.sum())),
        mean_age1 = float(np.mean(sim.people.age.values)),
        births = float(sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Births))),
        deaths = float(sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Deaths))),
    )
    return out

if __name__ == '__main__':
    fast = 'fast' in sys.argv
    res = {}
    for seed in SEEDS:
        sim = make_sim(seed, fast=fast)
        T = time.perf_counter(); sim.run(); el = time.perf_counter() - T
        res[seed] = summarize(sim); res[seed]['time'] = el
        r = res[seed]
        print(f"seed {seed}: N {r['n0']:.0f}->{r['n1']:.0f}  nodes {r['node1']}  prev {np.round(r['prev_node'],3)}  tvd {r['tvd']:.3f}  births {r['births']} deaths {r['deaths']}  {el:.2f}s")
    tag = 'fast' if fast else 'plain'
    sc.savejson(f'out_v3_baseline_{tag}{TAG}.json', res)
    keys = ['n1', 'prev', 'tvd', 'mean_age1', 'births', 'deaths', 'time']
    for k in keys:
        v = [res[s][k] for s in SEEDS]; print(f'{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}')
    pn = np.array([res[s]['prev_node'] for s in SEEDS]); print('prev by node', pn.mean(0).round(4), pn.std(0).round(4))
    nd = np.array([res[s]['node1'] for s in SEEDS]); print('node pops end', nd.mean(0).round(1), nd.std(0).round(1))
