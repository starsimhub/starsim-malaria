""" Probe: a declared EIR-driven malaria SIS (indoor and outdoor bites as two routes), 2 patches, agents and ODE """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc

def make(solver='abm', n=4000, eir=(2.0, 4.0), f_in=0.7, seed=1, interventions=None):
    S, I = ss.states(2)
    eir_by = ss.by(patch=[ss.peryear(e) for e in eir])
    mal = ss.Disease(
        infection_in  = S >> I @ eir_by * f_in,
        infection_out = S >> I @ eir_by * (1 - f_in),
        recovery      = I >> S @ ss.days(200),
        name = 'malaria',
    )
    mal.stratify(patch=['n0', 'n1'], split=[0.5, 0.5])
    kw = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    sim = ss.Sim(mal, start='2000-01-01', stop='2002-01-01', dt=ss.days(1), solver=solver, rand_seed=seed, interventions=interventions, verbose=0, **kw)
    return sim

if __name__ == '__main__':
    for solver in ['abm', 'ode']:
        sim = make(solver)
        T = sc.timer()
        sim.run()
        res = sim.results.malaria
        print(solver, T.toc(output=True), 'n_I end', res.n_I[-1], 'by patch', res.n_I.by('patch').values[-1])
    print(sim.describe())
