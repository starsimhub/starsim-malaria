"""
EIR calibration: malariasim's analytic eir_to_params() against v4's sim.calibrate(gof=) on an unchanged ms.Sim.

Single node, 5,000 humans, human ODE + mosquito ODE (deterministic), 3 years, target annual EIR (year-3 mean).
(1) Analytic: eir_to_params(target) gives the emergence; run it and report the EIR it achieves.
(2) calibrate(gof=) fits an emergence multiplier. The 12-month emergence array can't be an ss.par
    (calib_spellings.py), so a 6-line subclass of MosquitoesODE adds a scalar emergence_scale.
(3) calibrate(gof=) fits bites_per_female_per_day, an existing scalar, with no code change.
"""
import numpy as np
import sciris as sc
import scipy.optimize as spo
import starsim as ss; print(ss.__version__, ss.__file__)
import malshim # noqa
import malariasim as ms

class ScaledMosODE(ms.MosquitoesODE):
    """ MosquitoesODE with a scalar multiplier on emergence, so it can be fitted """
    def __init__(self, pars=None, **kwargs):
        ms.MosquitoesODE.__init__(self) # Defines the pars; then add one and apply the user's
        self.define_pars(emergence_scale=1.0)
        self.update_pars(pars, **kwargs)
    def _emergence_rate_for_node(self, grid, nid, t):
        return float(self.pars.emergence_scale) * super()._emergence_rate_for_node(grid, nid, t)
ms.sim.MosquitoesODE = ScaledMosODE # ms.Sim builds its mosquito from this name

CP = ms.CommonParams()
YEARS = 3
def mk(target, scale=None, a=None, emergence=None):
    em = emergence if emergence is not None else ms.eir_to_params(target, common_pars=CP)[1]
    mos = {}
    if scale is not None: mos['emergence_scale'] = scale
    if a is not None:     mos['bites_per_female_per_day'] = a
    return ms.Sim(human='ode', mosquito='ode', nodes=ms.NodeGrid(populations=[5000], emergence=[em]), start=2000, stop=2000+YEARS,
                  verbose=0, common_pars=CP, human_pars=dict(init_prev=0.3), mosquito_pars=mos)

def eir_y3(sim):
    v = np.asarray(sim.connectors[-1].results.eir_annual.values, dtype=float)
    return v[-366:-1].mean() # The last point is never written in v4 (results written in step() at self.ti)

rows = []
for target in [2.0, 10.0, 50.0]:
    em = ms.eir_to_params(target, common_pars=CP)[1]
    T = sc.timer()
    analytic = eir_y3(mk(target).run())
    t_an = T.toc(output=True)

    gof = lambda sim, target=target: (eir_y3(sim) - target)**2
    T = sc.timer()
    fit = mk(target, scale=ss.par(1.0, bounds=[0.2, 5.0])).calibrate(gof=gof, method='nelder-mead', verbose=0)
    t_fit = T.toc(output=True)
    scale = float(list(fit.pars.values())[0])
    achieved = eir_y3(mk(target, scale=scale).run())

    # The same with an existing parameter: bites per female per day (a), emergence from the analytic value
    T = sc.timer()
    fit_a = mk(target, a=ss.par(CP.bites_per_female_per_day, bounds=[0.005, 0.2])).calibrate(gof=gof, method='nelder-mead', verbose=0)
    t_fit_a = T.toc(output=True)
    a_fit = float(list(fit_a.pars.values())[0])

    # Brute-force check of the scale: a 1-D root find on the same model (what eir_to_params approximates analytically)
    T = sc.timer()
    root = spo.brentq(lambda s: eir_y3(mk(target, scale=s).run()) - target, 0.3, 3.0, xtol=1e-4)
    t_root = T.toc(output=True)
    rows.append(dict(target=target, emergence_analytic=em[0], eir_analytic=analytic, scale_fit=scale, eir_fit=achieved,
                     evals=getattr(fit, 'n_evals', None), t_fit=t_fit, scale_root=root, t_root=t_root, a_fit=a_fit, t_fit_a=t_fit_a, t_one_run=t_an))
    print(sc.objdict(rows[-1]))

import pandas as pd
df = pd.DataFrame(rows)
print(df.round(4).to_string())
print('fit object attributes:', [k for k in vars(fit) if not k.startswith('_')])
