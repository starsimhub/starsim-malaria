""" Probe: what EIR does the declared model actually apply through the year? Recover it from the flows (ODE, so no noise) """
import sys
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *
sys.path.insert(0, '.')
from interpolate import interpolate_monthly

b = 0.4
prof = node_eir_profiles()
def ts(row, keys):
    return ss.timeseries({k: ss.peryear(v) for k, v in zip(keys, list(row) + [row[0]])}, mode='linear', period=ss.years(1))

for label, keys, extra in [
        ('numbers k/12 (read in the sim unit, days)', [k/12 for k in range(13)], {}),
        ('numbers k/12 with unit="year"', [k/12 for k in range(13)], dict(unit='year')),
        ('dates (month starts)', [f'{2000 + (k//12)}-{(k%12)+1:02d}-01' for k in range(13)], {}),
    ]:
    S, I = ss.states('S I')
    eir = ss.par(ss.by(patch=[ts(r, keys) for r in prof]), name='eir')
    mal = ss.Disease(bite = S >> I @ b * eir, recovery = I >> S @ ss.days(200), name='malaria')
    mal.stratify(patch=LABELS, kind='space', split=np.array(POPS)/sum(POPS))
    try:
        sim = ss.Sim(mal, total_pop=sum(POPS), solver='ode', start=START, stop=START+2, dt=ss.days(DT_DAYS), verbose=0, **extra).run()
    except Exception as e:
        print(label, '->', type(e).__name__, e); continue
    res = sim.results.malaria
    flow = res.bite.by('patch').values[1:]; nS = res.n_S.by('patch').values[:-1]
    dt_y = DT_DAYS/365
    eir_eff = -np.log(1 - flow/nS)/(b*dt_y) # per-year EIR implied by each step
    yrs = sim.results.timevec.years if hasattr(sim.results.timevec, 'years') else np.asarray(sim.results.timevec)
    want = np.array([[interpolate_monthly(prof[p], (y % 1)) for p in range(3)] for y in np.asarray(yrs, float)[:-1]])
    print(f'{label}: mean EIR applied {eir_eff.mean(0).round(2)} (want {want.mean(0).round(2)}); max abs error {np.abs(eir_eff - want).max():.2f}; min/max applied urban {eir_eff[:,0].min():.2f}/{eir_eff[:,0].max():.2f}')
