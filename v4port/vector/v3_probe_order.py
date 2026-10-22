"""v3: loop order and mosquito start-up in unchanged malariasim"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, malariasim as ms
for mos in ['agents', 'ode']:
    sim = ms.Sim(human='sis', mosquito=mos, start=2000, stop=2001, nodes=ms.NodeGrid(populations=[5000]), rand_seed=1, verbose=0)
    sim.init()
    if mos == 'agents':
        print('loop order of step 0:', [f.__qualname__ if hasattr(f,'__qualname__') else str(f) for f in sim.loop.funcs][:20])
    sim.run()
    m = sim.connectors[-1].results
    print(mos, 'n_total first 8 steps:', np.round(m.n_total[:8]).tolist(), 'at day 30:', round(float(m.n_total[30])), 'at day 180:', round(float(m.n_total[180])))
    print(mos, 'eir_annual first 5:', np.round(m.eir_annual[:5], 3).tolist())
