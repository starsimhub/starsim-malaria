""" Time malariasim seis+ode 10k (no demography, 2 years) on whichever v4 is on the path, after a warm-up run """
import sys, time, numpy as np
sys.argv = ['bench.py', 'starsim']
exec(open('bench.py').read().split("if __name__ == '__main__':")[0])
eir, eme = epi_profiles(F_TARGET_EIR)
def make(years):
    return ms.Sim(age_data=AGE_DATA, start=START, stop=START+years, dt=ss.days(1), rand_seed=SEED, common_pars=COMMON_PARS, human='seis', mosquito='ode',
                  nodes=ms.NodeGrid(populations=[10000], labels=['Node-0'], eir=[eir], emergence=[eme], f_indoor=[F_INDOOR]), build_campaign=_starsim_campaign, verbose=0)
make(0.1).run()
ts = []
for i in range(2):
    sim = make(2); t0 = time.perf_counter(); sim.run(); ts.append(time.perf_counter() - t0)
ni = np.asarray(sim.diseases[0].results.n_infected.values, dtype=float)
print(f'{ss.__file__.split("/")[-3]}: best {min(ts):.2f} s {np.round(ts,2)}; final infected {ni[-1]:.0f}', flush=True)
