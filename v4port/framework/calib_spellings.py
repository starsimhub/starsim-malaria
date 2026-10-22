""" Which spellings of a fittable emergence does v4 accept on an unchanged ms.Sim (human ODE + mosquito ODE)? """
import traceback
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
import malshim # noqa
import malariasim as ms

CP = ms.CommonParams()
eir_t, em_t = ms.eir_to_params(10.0, common_pars=CP)
E0 = em_t[0]
def mk(mos_pars):
    return ms.Sim(human='ode', mosquito='ode', nodes=ms.NodeGrid(populations=[5000]), start=2000, stop=2002, verbose=0,
                  common_pars=CP, human_pars=dict(init_prev=0.3), mosquito_pars=mos_pars)
def gof(sim):
    eir = np.asarray(sim.connectors[-1].results.eir_annual.values[-366:-1])
    return abs(eir.mean() - 10.0)

tries = {
    'emergence = ss.par(array of 12, bounds=)': lambda: dict(emergence=ss.par(np.full(12, E0), bounds=[0.1*E0, 10*E0], name='emergence')),
    'emergence = [ss.par(scalar, bounds=)] * 12': lambda: dict(emergence=[ss.par(E0, bounds=[0.1*E0, 10*E0], name='em')]*12),
    'emergence = ss.par(scalar, bounds=) (module checks 12 elements)': lambda: dict(emergence=ss.par(E0, bounds=[0.1*E0, 10*E0], name='em')),
}
for label, fn in tries.items():
    print(f'\n=== {label}')
    try:
        sim = mk(fn())
        sim.init()
        print('  init OK; emergence after init:', np.round(sim.connectors[-1].pars.emergence[:3], 5))
        fit = sim.calibrate(gof=gof, method='nelder-mead', verbose=0, maxiter=30)
        print('  fit OK:', fit.pars, 'mismatch', fit.mismatch)
    except Exception as e:
        print(f'  {type(e).__name__}: {str(e)[:400]}')
        tb = traceback.format_exc().strip().splitlines()
        print('   at', tb[-3].strip()[:200])
