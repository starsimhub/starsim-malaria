""" Why calibrate(gof=) takes 25-38 s where a root find on the same deterministic model takes 1-3 s: count runs per evaluation """
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
import malshim # noqa
import malariasim as ms
exec(open('calib_eir.py').read().split('CP = ms.CommonParams()')[0].split('import malariasim as ms')[1]) # ScaledMosODE
CP = ms.CommonParams()
n = [0]
orig = ss.Sim.run
def run(self, *a, **k):
    n[0] += 1
    return orig(self, *a, **k)
ss.Sim.run = run
def mk(scale, n_runs=None):
    em = ms.eir_to_params(10.0, common_pars=CP)[1]
    return ms.Sim(human='ode', mosquito='ode', nodes=ms.NodeGrid(populations=[5000], emergence=[em]), start=2000, stop=2003, verbose=0, common_pars=CP,
                  human_pars=dict(init_prev=0.3), mosquito_pars=dict(emergence_scale=scale))
gof = lambda sim: (np.asarray(sim.connectors[-1].results.eir_annual.values)[-366:-1].mean() - 10.0)**2
for kw in [dict(), dict(n_runs=1)]:
    n[0] = 0
    T = sc.timer()
    fit = mk(ss.par(1.0, bounds=[0.2, 5.0])).calibrate(gof=gof, method='nelder-mead', verbose=0, **kw)
    print(kw, 'sim runs:', n[0], 'trials:', len(fit.trials) if fit.trials is not None else None, 'seeds:', fit.seeds, 'time: %.1f s' % T.toc(output=True), 'scale', fit.pars)
