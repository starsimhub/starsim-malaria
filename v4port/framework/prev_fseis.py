""" Year-5 prevalence and run time of malariasim scenario fseis10 (seis + mosquito ODE, 10k, single node), with and without demographics, to set beside declared_seis.py """
import sys, time, numpy as np
sys.argv = ['bench.py', 'starsim']
exec(open('bench.py').read().split("if __name__ == '__main__':")[0])
eir, eme = epi_profiles(F_TARGET_EIR)
for demog in [True, False]:
    sim = build_starsim('seis', 'ode', [10000], ['Node-0'], [eir], [eme], 5)
    if not demog:
        sim = ms.Sim(age_data=AGE_DATA, start=START, stop=START+5, dt=ss.days(1), rand_seed=SEED, common_pars=COMMON_PARS, human='seis', mosquito='ode',
                     nodes=ms.NodeGrid(populations=[10000], labels=['Node-0'], eir=[eir], emergence=[eme], f_indoor=[F_INDOOR]), build_campaign=_starsim_campaign, verbose=0)
    t0 = time.perf_counter(); sim.run(); t = time.perf_counter() - t0
    ni = np.asarray(sim.diseases[0].results.n_infected.values, dtype=float)
    N = np.asarray(sim.results.n_alive.values, dtype=float)
    print(f'malariasim seis+ode 10k demog={demog}: {t:.3f} s; year-5 prevalence {np.mean((ni/N)[-366:-1]):.4f}', flush=True)
