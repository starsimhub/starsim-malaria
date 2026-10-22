""" Minimal repro: on the spec's own Ross-Macdonald model (03-states-and-transitions.md), the spec's bednet idiom
ss.modify(malaria.rel_sus, multiply=..., where=...) (08-interventions.md) changes nothing, silently. So does a scalar-efficacy vaccine? """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

def make():
    a, b, c = 0.3, 0.5, 0.5
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv = ss.states(3)
    Nh = (Sh + Eh + Ih + Rh).n
    return ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(100),
        waning     = Rh >> Sh @ ss.days(365),
        emergence  = ss.null >> Sv @ ss.perday(5_000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
        init       = dict(Ih=0.01, Sv=50_000),
        name = 'malaria',
    )

def run(label, interv=None, solver='abm'):
    m = make()
    iv = interv(m) if interv else None
    kw = dict(n_agents=5_000) if solver == 'abm' else dict(total_pop=5_000)
    sim = ss.Sim(m, unit='day', stop=365, solver=solver, verbose=0, rand_seed=2, interventions=iv, **kw).run()
    print(f'{solver:4s} {label:45s} human infections in a year: {sim.results.malaria.infection[:].sum():8.0f}')

for solver in ['abm', 'ode']:
    run('none', solver=solver)
    run('modify(rel_sus, multiply=0) [spec bednet idiom]', lambda m: ss.modify(m.rel_sus, multiply=0.0), solver=solver)
    run('modify(infection, multiply=0)', lambda m: ss.modify(m.infection, multiply=0.0), solver=solver)
    run('vaccinate(Vaccine(efficacy=1), coverage=1)', lambda m: ss.vaccinate(ss.Vaccine(efficacy=1.0), coverage=1.0), solver=solver)

print('--- controls')
for solver in ['abm']:
    run('vaccinate(Vaccine(efficacy=dict(infection=1)))', lambda m: ss.vaccinate(ss.Vaccine(efficacy=dict(infection=1.0)), coverage=1.0), solver=solver)
# Control: the same idioms on an exposure (contact) transmission do work
def make_sis():
    S, I = ss.states('S I')
    return ss.Disease(infection = S >> I @ ss.perday(0.3) * I, recovery = I >> S @ ss.days(10), init=dict(I=0.05), name='sis')
for label, f in [('none', None), ('modify(rel_sus, multiply=0)', lambda m: ss.modify(m.rel_sus, multiply=0.0)), ('vaccinate(efficacy=1)', lambda m: ss.vaccinate(ss.Vaccine(efficacy=1.0), coverage=1.0))]:
    m = make_sis()
    sim = ss.Sim(m, unit='day', stop=365, n_agents=5000, verbose=0, rand_seed=2, interventions=f(m) if f else None).run()
    print(f'abm  SIS with exposure: {label:30s} infections {sim.results.sis.infection[:].sum():8.0f}')
