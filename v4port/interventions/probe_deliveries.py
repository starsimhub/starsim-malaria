""" Probe: who is eligible for and receives each delivery on the declared EIR model """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from malaria_v4 import make
for name in ['vax', 'mda', 'itn']:
    sim = make(name, 'abm', 1)
    sim.run()
    r = sim.results
    keys = ['intv']
    for k in keys:
        try:
            d = r[k]
            print(name, k, {rk: float(np.nansum(d[rk].values)) for rk in d.keys() if rk != 'timevec'})
        except Exception as e:
            print(name, k, 'err', e)
    print('n_I at the campaign', r.malaria.n_I[365], r.malaria.n_I[366])
