""" Repro: a vaccine's acquisition efficacy (rel_sus) does nothing when the force of infection is an external rate (EIR), silently """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

def run(rate_form, vaccinate, solver='abm'):
    S, I = ss.states(2)
    if rate_form == 'external':      # EIR given as data: a rate with no state in it
        inf = S >> I @ ss.peryear(2.0)
    else:                            # the same hazard written as transmission from I (so rel_sus applies)
        inf = S >> I @ ss.peryear(2.0) * I
    mal = ss.Disease(infection=inf, recovery=I >> S @ ss.days(200), init=dict(I=0.5), name='malaria')
    intv = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=1.0, at=[ss.days(10)], name="campaign") if vaccinate else None
    kw = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
    sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(365), dt=ss.days(1), interventions=intv, solver=solver, verbose=0, rand_seed=1, **kw)
    sim.run()
    r = sim.results
    new = r.malaria.infection.values[20:].sum()
    deliv = r.campaign.delivered.values.sum() if vaccinate else 0
    return new, deliv, sim

for solver in ['abm', 'ode']:
    for form in ['external', 'transmission']:
        base, _, _ = run(form, False, solver)
        vax, n, sim = run(form, True, solver)
        print(f'{solver} {form:12s}: new infections after day 20, no vaccine {base:8.1f}; 100% vaccinated at 90% efficacy {vax:8.1f} (delivered {n:.0f})')
print(sim.what_modifies('malaria.rel_sus'))
print(sim.describe())
