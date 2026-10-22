"""
A minimal declared v4 malaria model for the speed comparison: humans S, E, I (SEIS) as agents or groups, mosquitoes
Sv, Ev, Iv as counts (states that are not people), single node, constant emergence. Matches malariasim's
human='seis' + mosquito='ode' (scenario f, single node): b=0.4, c=0.3, a=0.03/day, mosquito life 10 days,
EIP 12 days as a rate (as MosquitoesODE), incubation normal(12, 2) days, infection lognormal(200, 200) days,
emergence from eir_to_params(6), 15% seeded at day 30.
Usage: python declared_seis.py <n_humans> <years> <reps> [demog]
"""
import sys, time
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__, flush=True)
import malshim # noqa (only for malariasim's eir_to_params and demographics)
import malariasim as ms
ss.options(crn=False)

CP = ms.CommonParams()
EM = ms.eir_to_params(6.0, common_pars=CP)[1][0] # New females per human per day

def make(n, years, solver, demog=False, seed=0):
    S, E, I = ss.states(3)
    Sv, Ev, Iv = [ss.State(name, people=False) for name in ['Sv', 'Ev', 'Iv']]
    Nh = (S + E + I).n
    a, b, c = ss.perday(CP.bites_per_female_per_day), CP.prob_human_infected_on_bite, CP.prob_mosquito_infected_on_bite
    mal = ss.Disease(
        infection  = S >> E @ a * b * Iv.n / Nh,
        onset      = E >> I @ ss.normal(ss.days(12), ss.days(2)),
        recovery   = I >> S @ ss.lognorm_ex(mean=ss.days(CP.infection_duration_mean_days), std=ss.days(CP.infection_duration_std_days)),
        emergence  = ss.null >> Sv @ ss.perday(EM * n),
        biting     = Sv >> Ev @ a * c * I.n / Nh,
        incubation = Ev >> Iv @ ss.perday(1/CP.mosquito_eip_days),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.perday(1/CP.mosquito_life_expectancy_days),
        init       = dict(S=1.0, Sv=EM * n * CP.mosquito_life_expectancy_days),
        name = 'malaria',
    )
    seed_inf = ss.Delivery(move='S -> I', coverage=0.15, at=[ss.days(30)], name='seed')
    kw = dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)
    if demog:
        kw.update(demographics=ms.make_steady_state_demographics(), use_aging=False)
    return ss.Sim(mal, start='2000-01-01', stop=ss.years(years), dt=ss.days(1), solver=solver, rand_seed=seed, interventions=seed_inf, verbose=0, **kw)

if __name__ == '__main__':
    n, years, reps = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    demog = len(sys.argv) > 4 and sys.argv[4] == 'demog'
    for solver in ['abm', 'ode']:
        try:
            make(200, 0.1, solver, demog).run() # Warm-up (Numba kernels)
            times = []
            for r in range(reps):
                sim = make(n, years, solver, demog)
                t0 = time.perf_counter(); sim.run(); times.append(time.perf_counter() - t0)
            res = sim.results.malaria
            ni = np.asarray(res.n_I.values, dtype=float); N = ni + np.asarray(res.n_S.values, dtype=float) + np.asarray(res.n_E.values, dtype=float)
            prev = (ni/N)[-365:].mean()
            print(f'declared seis {solver} n={n} demog={demog}: best {min(times):.3f} s (all {np.round(times, 3).tolist()}); year-{years:g} prevalence {prev:.4f}', flush=True)
        except Exception as e:
            print(f'declared seis {solver} n={n} demog={demog}: {type(e).__name__}: {str(e)[:300]}', flush=True)
