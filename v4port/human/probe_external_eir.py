""" How does a declared human disease take an EIR computed by another (coded) module each step, per node?
Spellings tried: a function of the sim in ss.by(patch=[...]); a function returning one value per patch; a function of (sim, uids); a parameter changed by the coded module """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc, traceback

class Bus(ss.Module):
    """ Stand-in for malariasim's NodeState + a coded mosquito module: writes a per-node EIR (per person per year) each step """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.eir = np.zeros(2)
        self.compartmental = True # Its code doesn't touch people
    def step(self):
        self.eir[:] = [5.0, 50.0] # per person per year

b = 0.4
def build(rate, solver, n=10_000):
    S, I = ss.states('S I')
    d = ss.Disease(infection=S >> I @ rate, recovery=I >> S @ ss.days(200), init=dict(I=0.0), name='malaria')
    d.stratify(patch=['n0', 'n1'], split=[0.5, 0.5])
    kw = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    return ss.Sim(d, connectors=Bus(name='bus'), start=2000, stop=2003, dt=ss.days(1), solver=solver, verbose=0, **kw)

spellings = {
    'by(patch=[fn(sim), fn(sim)]) * peryear': lambda: b*ss.by(patch=[lambda sim: sim.connectors.bus.eir[0], lambda sim: sim.connectors.bus.eir[1]])*ss.peryear(1),
    'fn(sim) returning per-patch array':      lambda: lambda sim: ss.peryear(b*sim.connectors.bus.eir),
    'fn(sim, uids) by node of each agent':    lambda: lambda sim, uids: ss.peryear(b*sim.connectors.bus.eir[np.asarray(sim.people.patch[uids], int)]),
    'fn(sim) returning a scalar (node 1)':     lambda: lambda sim: ss.peryear(b*sim.connectors.bus.eir[1]),
    'bare-number fn(sim) (node 1), x peryear(1)': lambda: (lambda sim: b*sim.connectors.bus.eir[1])*ss.peryear(1),
}
eq = [b*e/(b*e + 365/200) for e in [5, 50]]
print('SIS equilibrium prevalence by node', np.round(eq, 4))
for label, f in spellings.items():
    for solver in ['abm', 'ode']:
        try:
            sim = build(f(), solver).run()
            nI = sim.results.malaria.n_I.by('patch').values
            print(f'OK   {solver:3s} {label:42s} final prevalence by node', np.round(nI[-1]/5000, 4))
        except Exception as e:
            print(f'FAIL {solver:3s} {label:42s} {type(e).__name__}: {str(e)[:300]}')
