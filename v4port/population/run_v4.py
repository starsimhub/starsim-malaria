""" The declared population model on v4, against the v3 reference (v3_baseline.py): 3 nodes, 20 years, weekly, 6 seeds """
import sys, time
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m

def summarize(sim):
    res = sim.results.malaria
    N = res.n_S.values + res.n_I.values
    byp = res.n_S.by('patch').values + res.n_I.by('patch').values
    bya = res.n_S.by('age').values + res.n_I.by('age').values
    nT = len(N); last5 = slice(nT - int(5*365/DT_DAYS), nT)
    prevp = (res.n_I.by('patch').values[last5]/byp[last5]).mean(0)
    flows = {k: float(np.sum(res[k].values)) for k in res.keys() if k in ('births', 'deaths', 'death', 'birth')}
    return dict(n0=N[0], n1=N[-1], node1=list(byp[-1]), prev_node=list(prevp), prev=float((res.n_I.values[last5]/N[last5]).mean()),
                pyr0=list(bya[0]/bya[0].sum()), pyr1=list(bya[-1]/bya[-1].sum()), tvd=tvd(bya[0]/bya[0].sum(), bya[-1]/bya[-1].sum()), **flows)

if __name__ == '__main__':
    configs = [('abm', 'modules'), ('abm', 'declared'), ('abm', 'lifetable'), ('ode', 'declared'), ('ode', 'lifetable')]
    if len(sys.argv) > 1:
        configs = [tuple(c.split(':')) for c in sys.argv[1:]]
    allres = {}
    v3 = sc.loadjson(f'out_v3_baseline_plain{TAG}.json')
    v3n = np.array([v3[str(s)]['node1'] for s in SEEDS]); v3p = np.array([v3[str(s)]['prev_node'] for s in SEEDS])
    print(f'v3 reference: N end {np.mean([v3[str(s)]["n1"] for s in SEEDS]):.0f}; nodes end {v3n.mean(0).round(0)}; prev by node {v3p.mean(0).round(4)} +/- {v3p.std(0).round(4)}; tvd {np.mean([v3[str(s)]["tvd"] for s in SEEDS]):.3f}')
    for solver, demog in configs:
        rows = []
        seeds = SEEDS if solver == 'abm' else [0]
        for seed in seeds:
            sim = m.make_sim(seed=seed, solver=solver, demog=demog)
            T = time.perf_counter(); sim.run(); el = time.perf_counter() - T
            r = summarize(sim); r['time'] = el; rows.append(r)
        nd = np.array([r['node1'] for r in rows]); pv = np.array([r['prev_node'] for r in rows])
        print(f'{solver}/{demog}: N {rows[0]["n0"]:.0f} -> {np.mean([r["n1"] for r in rows]):.0f}; nodes end {nd.mean(0).round(0)}; prev by node {pv.mean(0).round(4)} +/- {pv.std(0).round(4)}; tvd {np.mean([r["tvd"] for r in rows]):.3f}; time {np.mean([r["time"] for r in rows]):.2f}s (first run includes compile)')
        allres[f'{solver}/{demog}'] = rows
    sc.savejson(f'out_run_v4{TAG}.json', allres)
