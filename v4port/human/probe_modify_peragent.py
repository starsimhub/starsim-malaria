""" Workarounds for per-person acquisition multipliers (continuous biting heterogeneity, a net), since rel_sus doesn't reach an EIR-driven rate:
ss.modify on the infection transition with where=, and with a per-agent amount """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
def make():
    S, I = ss.states('S I')
    return ss.Disease(infection=S >> I @ 0.4*ss.peryear(10), recovery=I >> S @ ss.days(200), init=dict(I=0.0), name='m')
W = None
def amount(sim):
    global W
    if W is None or len(W) < len(sim.people.uid): W = np.random.default_rng(1).gamma(1.0, 1.0, size=len(sim.people.uid)) # per-UID biting weights
    return W[sim.people.auids] if False else W
tries = {
    'none': lambda m: None,
    'modify(infection, multiply=0.5, where=even uids)': lambda m: ss.modify(m.infection, multiply=0.5, where=lambda sim: sim.people.uid % 2 == 0),
    'modify(infection, multiply=fn(sim) -> per-UID array)': lambda m: ss.modify(m.infection, multiply=amount),
}
for label, f in tries.items():
    m = make()
    try:
        sim = ss.Sim(m, start=2000, stop=2001, dt=ss.days(1), n_agents=10_000, verbose=0, rand_seed=1, interventions=f(m)).run()
        print(f'OK   {label:56s} infections {sim.results.m.infection[:].sum():.0f}')
    except Exception as e:
        print(f'FAIL {label:56s} {type(e).__name__}: {str(e)[:200]}')
