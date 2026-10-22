""" Does a per-person rel_sus (base, or via ss.modify) reach an EIR-driven ("spontaneous") infection rate? """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

def make(**kw):
    S, I = ss.states('S I')
    return ss.Disease(infection = S >> I @ 0.4 * ss.peryear(10), recovery = I >> S @ ss.days(200), init=dict(I=0.0), name='malaria', **kw)

def zero_half(sim):
    """ Set the base rel_sus of the even UIDs to 0 at the start """
    if sim.ti == 0 or sim.ti == 1:
        m = sim.diseases.malaria
        uids = sim.people.auids
        m.rel_sus[uids[::2]] = 0.0

out = {}
# A: nothing
sim = ss.Sim(make(), start=2000, stop=2001, dt=ss.days(1), n_agents=10_000, verbose=0, rand_seed=1).run()
out['none'] = sim.results.malaria.infection.cum[-1] if hasattr(sim.results.malaria.infection, 'cum') else sim.results.malaria.infection[:].sum()
# B: base rel_sus 0 for half, set by a plain function
try:
    sim = ss.Sim(make(seeding=zero_half), start=2000, stop=2001, dt=ss.days(1), n_agents=10_000, verbose=0, rand_seed=1).run()
    out['base rel_sus=0 for half'] = sim.results.malaria.infection[:].sum()
except Exception as e:
    out['base rel_sus=0 for half'] = f'{type(e).__name__}: {e}'
# C: ss.modify(rel_sus, multiply=0)
m = make()
try:
    sim = ss.Sim(m, start=2000, stop=2001, dt=ss.days(1), n_agents=10_000, verbose=0, rand_seed=1, interventions=ss.modify(m.rel_sus, multiply=0.0)).run()
    out['modify rel_sus x0'] = sim.results.malaria.infection[:].sum()
except Exception as e:
    out['modify rel_sus x0'] = f'{type(e).__name__}: {e}'
# D: ss.modify on the transition itself
m = make()
sim = ss.Sim(m, start=2000, stop=2001, dt=ss.days(1), n_agents=10_000, verbose=0, rand_seed=1, interventions=ss.modify(m.infection, multiply=0.0)).run()
out['modify infection x0'] = sim.results.malaria.infection[:].sum()
for k,v in out.items(): print(f'{k:28s} {v}')
