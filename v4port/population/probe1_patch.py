""" Probe: declared human SIS with per-patch seasonal EIR, split indoor/outdoor, patch dimension; no demography """
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
from common import *

b = 0.4
prof = node_eir_profiles()
def ts(row):
    return ss.timeseries({k/12: ss.peryear(v) for k, v in enumerate(list(row) + [row[0]])}, mode='linear', period=ss.years(1))

S, I = ss.states('S I')
eir  = ss.par(ss.by(patch=[ts(r) for r in prof]), name='eir')
fin  = ss.par(ss.by(patch=F_INDOOR), name='f_indoor')
mal = ss.Disease(
    bite_indoor  = S >> I @ b * fin * eir,
    bite_outdoor = S >> I @ b * (1 - fin) * eir,
    recovery     = I >> S @ ss.lognorm_ex(mean=ss.days(200), std=ss.days(200)),
    name = 'malaria',
)
mal.stratify(patch=LABELS, kind='space', split=np.array(POPS)/sum(POPS))
sim = ss.Sim(mal, n_agents=sum(POPS), start=START, stop=START+10, dt=ss.days(DT_DAYS), rand_seed=1, verbose=0)
sim.init()
print(sim.describe() if hasattr(sim, 'describe') else '')
sim.run()
res = sim.results.malaria
print('patch counts t0', res.n_S.by('patch').values[0] + res.n_I.by('patch').values[0])
nT = len(res.n_I); last5 = slice(nT - int(5*365/DT_DAYS), nT)
prev = (res.n_I.by('patch').values / (res.n_I.by('patch').values + res.n_S.by('patch').values))[last5].mean(0)
print('prev by patch (last 5 y)', np.round(prev, 4))
