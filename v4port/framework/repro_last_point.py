""" Unchanged malariasim results written in step() at self.ti: first, second and last points of a few results, v3 and v4 """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
if ss.__version__.startswith('4'):
    import malshim # noqa
import malariasim as ms
for human, mos in [('sis', 'eir_only'), ('ode', 'ode')]:
    sim = ms.Sim(human=human, mosquito=mos, nodes=ms.NodeGrid(populations=[1000]), start=2000, stop=2000+30/365, verbose=0,
                 human_pars=dict(init_prev=ss.bernoulli(0.1)) if human != 'ode' else dict(init_prev=0.1))
    sim.run()
    m = sim.connectors[-1].results
    h = sim.diseases[0].results
    print(f'{human}+{mos}: npts={len(m.eir_annual)}')
    for name, r in [('mos.eir_annual', m.eir_annual), ('hum.n_infected', h.n_infected)] + ([('mos.n_total', m.n_total)] if 'n_total' in m else []):
        v = np.asarray(r.values, dtype=float)
        print(f'   {name:16s} [0]={v[0]:10.3f} [1]={v[1]:10.3f} [-2]={v[-2]:10.3f} [-1]={v[-1]:10.3f}')
