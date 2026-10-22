""" Repro: the same typed birth rate in ss.Births and in ss.birth. 0.025768 per person per year (malariasim's calibrated rate), 10,000 people, no deaths, 20 years weekly, 10 seeds """
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
r, N0, Y = 0.025768, 10_000, 20
print(f'a rate of {r}/y: N should grow to {N0*np.exp(r*Y):.0f}; as an annual probability, to {N0*(1-r)**-Y:.0f}')
for label, demog in [('ss.Births(birth_rate=ss.peryear(25.768))', lambda: [ss.Births(birth_rate=ss.peryear(r*1000))]), ('ss.birth(rate=ss.peryear(0.025768))', lambda: [ss.birth(rate=ss.peryear(r))])]:
    ends = []
    for seed in range(10):
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(0.1)), recovery=ss.progression('I -> S', dur=ss.years(1)), init_prev=0.0, name='sis')
        sim = ss.Sim(sis, n_agents=N0, demographics=demog(), start=2000, stop=2000+Y, dt=ss.days(7), rand_seed=seed, verbose=0).run()
        ends.append(sim.people.alive.sum())
    print(f'{label}: N after {Y} y {np.mean(ends):.0f} +/- {np.std(ends)/np.sqrt(len(ends)):.0f}')
