""" The Ross-Macdonald scenarios of run_v3_rm.py, on the declared v4 model """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
from scen import *
from malaria_rm_v4 import make

if __name__ == '__main__':
    solvers = sys.argv[1:] or ['abm']
    out = {}
    for solver in solvers:
        out[solver] = {}
        for name in RM_SCENARIOS:
            out[solver][name] = []
            for seed in (SEEDS if solver == 'abm' else [1]):
                try:
                    sim = make(name, solver, seed)
                    t0 = sc.tic(); sim.run(); el = sc.toc(t0, output=True)
                except Exception as e:
                    print(f'{solver} {name}: {type(e).__name__}: {str(e)[:400]}'); break
                r = sim.results.malaria
                idx = np.array(RM_REPORT) + 1
                mos = (r.n_Sv.values + r.n_Ev.values + r.n_Iv.values)[idx]
                eir = RM['a']*r.n_Iv.values[idx]/RM['pop']*365
                new = r.infection_in.values + r.infection_out.values
                out[solver][name].append(dict(inf=r.n_I.values[idx].tolist(), mos=mos.tolist(), Iv=r.n_Iv.values[idx].tolist(), eir=eir.tolist(), cum_y2=float(new[366:].sum()), time=el))
            if not out[solver][name]: continue
            f = lambda k: np.round(np.mean([o[k] for o in out[solver][name]], axis=0), 1).tolist()
            print(f'{solver} {name:10s} n_inf {f("inf")}; mosquitoes {f("mos")}; Iv {f("Iv")}; EIR/yr {f("eir")}; cum y2 {np.mean([o["cum_y2"] for o in out[solver][name]]):.0f}; t {np.mean([o["time"] for o in out[solver][name]]):.2f}s', flush=True)
    sc.savejson(f'compare_v4_rm_{"_".join(solvers)}.json', out)
