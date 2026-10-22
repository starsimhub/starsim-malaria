""" How much malariasim's double count of bednet blocking matters: the v4 Ross-Macdonald model with and without it """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
from scen import *
from malaria_rm_v4 import make
for dc in [True, False]:
    for name in ['itn', 'itn_nokill']:
        res = []
        for seed in SEEDS:
            r = make(name, 'abm', seed, double_count=dc).run().results.malaria
            res.append([r.n_I[730], (r.infection_in.values + r.infection_out.values)[366:].sum(), r.n_Iv[730]])
        m = np.mean(res, axis=0)
        print(f'{"as malariasim (double count)" if dc else "each bite blocked once      "} {name:10s}: n_I at day 729 {m[0]:6.0f}, new infections in year 2 {m[1]:6.0f}, Iv {m[2]:5.1f}')
