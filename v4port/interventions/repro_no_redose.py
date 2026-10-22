""" Repro: a second round never reaches people who hold a product whose protection fades by a function of time, even once it has faded to 0 """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np

for waning_label, waning in [('function, 0 after 100 days', lambda t: np.where(np.asarray(t) < 100, 1.0, 0.0)), ('duration (exponential loss), mean 100 days', ss.days(100))]:
    for solver in ['abm', 'ode']:
        S, I = ss.states(2)
        mal = ss.Disease(infection=S >> I @ ss.peryear(2.0), recovery=I >> S @ ss.days(200), name='mal')
        vax = ss.Vaccine(efficacy=dict(infection=0.9), waning=waning)
        rounds = ss.vaccinate(vax, coverage=0.8, at=[ss.days(100), ss.days(400)], name='rounds')
        kw = dict(n_agents=4000) if solver == 'abm' else dict(total_pop=4000)
        sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(600), dt=ss.days(1), solver=solver, interventions=rounds, verbose=0, rand_seed=1, **kw).run()
        d = sim.results.rounds.delivered.values
        print(f'waning={waning_label:45s} {solver}: delivered in round 1 {d[95:110].sum():5.0f}, in round 2 {d[395:410].sum():5.0f} (malariasim re-doses: 3200 each); new infections days 400-600 {sim.results.mal.infection.values[401:].sum():6.0f}')
