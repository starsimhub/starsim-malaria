""" Probe: where does the declared ODE spend 2 ms a step? 3 patches x 20 age groups x 2 states, 5 years daily; vary the pieces """
import time, cProfile, pstats, io
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
import mal_v4 as m
import sys
SOLVER = sys.argv[1] if len(sys.argv) > 1 else 'ode'
def make(age=True, seasonal=True, demog=True):
    eir = node_eir_profiles() if seasonal else np.tile(np.array(ANNUAL)[:, None], 12)
    mal = m.make_malaria(eir=eir)
    if not seasonal: # constant EIR as plain numbers, no time series
        S, I = ss.states('S I')
        e = ss.par(ss.by(patch=[ss.peryear(a) for a in ANNUAL]), name='eir'); fin = ss.par(ss.by(patch=F_INDOOR), name='f_indoor')
        mal = ss.Disease(bite_indoor=S >> I @ 0.4*fin*e, bite_outdoor=S >> I @ 0.4*(1 - fin)*e, recovery=I >> S @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)), prevalence=I/(S + I), people=S + I, name='malaria')
        mal.stratify(patch=LABELS, kind='space', split=np.array(POPS)/sum(POPS))
    return ss.Sim(mal, demographics=m.steady_state('declared') if demog else None, stratify=dict(age=AGE_BIN_EDGES[:-1]) if age else None, solver=SOLVER,
                  people=ss.People(5000, age_data=sc.dataframe(m.age_data())), start=START, stop=START+5, dt=ss.days(1), verbose=0)
for kw in [dict(), dict(seasonal=False), dict(age=False, demog=False), dict(age=False, demog=False, seasonal=False)]:
    ts = []
    for rep in range(3):
        sim = make(**kw); T = time.perf_counter(); sim.run(); ts.append(time.perf_counter() - T)
    print(f'{str(kw):55s}: best of 2 after warm-up {min(ts[1:]):.2f}s ({1e3*min(ts[1:])/1826:.2f} ms/step)')
sim = make(); pr = cProfile.Profile(); pr.enable(); sim.run(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(25); print(s.getvalue()[:5000])
