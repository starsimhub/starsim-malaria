"""
Within-step order between humans (agents) and mosquito counts in v4. Humans start all susceptible; there are no
infectious mosquitoes at the start, but a million exposed ones that become infectious within hours. If human infection
saw the mosquito count after it is advanced (as 10-execution.md says), people would be infected in the first step.
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
def make():
    Sh, Ih = ss.states(2)
    Ev, Iv = ss.states(2)
    N = (Sh + Ih).n
    return ss.Disease(
        infection  = Sh >> Ih @ ss.perday(1e-4) * Iv.n / N,  # about 100 infectious bites per person per day once Iv is 1e6
        incubation = Ev >> Iv @ ss.perday(50),               # mean 30 minutes
        init       = dict(Sh=1.0, Ev=1_000_000),
        name = 'mal')
for solver in ['abm', 'ode']:
    pop = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
    sim = ss.Sim(make(), solver=solver, unit='day', start=0, stop=3, dt=1, verbose=0, **pop).run()
    r = sim.results.mal
    tr = sim.diseases.mal.graph.transitions['infection'] if solver == 'abm' else None
    print(f'{solver}: infections by step {np.round(r.infection[:4], 1).tolist()}; n_Iv {np.round(r.n_Iv[:4]).tolist()}' + (f'; infection.contact={tr.contact}' if tr is not None else ''))
