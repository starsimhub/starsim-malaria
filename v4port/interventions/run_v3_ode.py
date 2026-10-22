""" malariasim (v3) with its ODE human (MalariaHumanODE) on the EIR-only scenarios of run_v3.py: the same interventions on the ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
import malariasim as ms
from scen import *
from run_v3 import campaign

def make(name):
    eir = [0.0, 0.0] if name == 'seed' else EIR
    nodes = ms.NodeGrid(populations=POPS, eir=eir, f_indoor=[F_INDOOR]*len(POPS))
    return ms.Sim(human='ode', mosquito='eir_only', nodes=nodes, start=2000, stop=2002, dt=ss.days(1), human_pars=dict(infection_duration=ss.days(DUR_INF)),
                  verbose=0, build_campaign=campaign(name))

if __name__ == '__main__':
    out = {}
    for name in SCENARIOS:
        try:
            sim = make(name)
            t0 = sc.tic(); sim.run(); el = sc.toc(t0, output=True)
        except Exception as e:
            print(f'{name}: {type(e).__name__}: {str(e)[:300]}'); continue
        d = sim.diseases[0]
        inf = np.asarray(d.results.n_infected.values)[REPORT]
        out[name] = dict(inf=inf.tolist(), time=el)
        print(f'{name:10s} n_inf at {REPORT}: {np.round(inf).astype(int).tolist()}; t {el:.2f}s', flush=True)
    sc.savejson('compare_v3_ode.json', out)
