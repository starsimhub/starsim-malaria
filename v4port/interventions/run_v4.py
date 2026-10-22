""" The same scenarios as run_v3.py, on the declared v4 model, as agents and as an ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
from scen import *
from malaria_v4 import make

if __name__ == '__main__':
    solvers = sys.argv[1:] or ['abm', 'ode']
    out = {}
    for solver in solvers:
        out[solver] = {}
        for name in SCENARIOS:
            out[solver][name] = []
            for seed in (SEEDS if solver == 'abm' else [1]):
                sim = make(name, solver, seed)
                t0 = sc.tic(); sim.run(); el = sc.toc(t0, output=True)
                r = sim.results.malaria
                # v4's n_I[ti] is the time point ti (0 = the start); v3's index ti is the state after step ti, so v3 day d is v4 point d + 1
                idx = np.array(REPORT) + 1
                inf = r.n_I.values[idx]
                nodes = r.n_I.by('patch').values[idx]
                new = r.infection_in.values + r.infection_out.values
                out[solver][name].append(dict(inf=inf.tolist(), inf_nodes=nodes.tolist(), cum_y2=float(new[366:].sum()), cum=float(new.sum()), time=el))
            m = np.mean([o['inf'] for o in out[solver][name]], axis=0); s = np.std([o['inf'] for o in out[solver][name]], axis=0)
            mn = np.mean([o['inf_nodes'] for o in out[solver][name]], axis=0)
            print(f'{solver} {name:10s} n_inf at {REPORT}: {np.round(m).astype(int).tolist()} (sd {np.round(s).astype(int).tolist()}); nodes end {np.round(mn[-1]).astype(int).tolist()}; nodes d395 {np.round(mn[2]).astype(int).tolist()}; cum y2 {np.mean([o["cum_y2"] for o in out[solver][name]]):.0f}; t {np.mean([o["time"] for o in out[solver][name]]):.2f}s', flush=True)
    sc.savejson(f'compare_v4_{"_".join(solvers)}.json', out)
