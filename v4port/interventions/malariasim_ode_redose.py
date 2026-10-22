""" malariasim (v3): a second vaccine round on the ODE human, against one round, and against the agent-based human """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
import malariasim as ms
from scen import *

def make(human, days, seed=1):
    def build(sim, eir_only=False):
        vax = ms.MalariaVaccine(waning=ms.WaningEffect(initial_efficacy=VAX['eff'], box_duration=ss.days(VAX['box']), decay_time_constant=ss.days(VAX['tau'])))
        return ms.Campaign(sim, [ms.InterventionDistributorScheduled(name=f'vax{i}', products=vax, start=2000 + d/365, coverage=VAX['cov']) for i, d in enumerate(days)])
    nodes = ms.NodeGrid(populations=POPS, eir=EIR, f_indoor=[F_INDOOR]*len(POPS))
    hp = dict(infection_duration=ss.days(DUR_INF)) if human == 'ode' else dict(infection_duration_years=ss.lognorm_ex(mean=DUR_INF/365, std=DUR_INF/365))
    sim = ms.Sim(human=human, mosquito='eir_only', nodes=nodes, start=2000, stop=2002, dt=ss.days(1), human_pars=hp, verbose=0, rand_seed=seed, build_campaign=build)
    sim.run()
    return np.asarray(sim.diseases[0].results.n_infected.values)[[499, 545, 600, 729]]

for human in ['ode', 'sis']:
    for days in [[], [200], [200, 500]]:
        print(f'{human:4s} rounds on days {str(days):10s}: n_infected at days 499, 545, 600, 729 = {np.round(make(human, days)).astype(int).tolist()}')
