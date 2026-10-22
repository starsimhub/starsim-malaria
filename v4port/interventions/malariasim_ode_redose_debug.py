""" malariasim (v3): what the ODE human's vaccine state does at a second round """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
import malariasim as ms
from scen import *
from malariasim_ode_redose import make as _m

def build(sim, eir_only=False):
    vax = ms.MalariaVaccine(waning=ms.WaningEffect(initial_efficacy=VAX['eff'], box_duration=ss.days(VAX['box']), decay_time_constant=ss.days(VAX['tau'])))
    return ms.Campaign(sim, [ms.InterventionDistributorScheduled(name=f'vax{i}', products=vax, start=2000 + d/365, coverage=VAX['cov']) for i, d in enumerate([200, 500])])
nodes = ms.NodeGrid(populations=POPS, eir=EIR, f_indoor=[F_INDOOR]*len(POPS))
sim = ms.Sim(human='ode', mosquito='eir_only', nodes=nodes, start=2000, stop=2002, dt=ss.days(1), human_pars=dict(infection_duration=ss.days(DUR_INF)), verbose=0, build_campaign=build)
sim.init()
d = sim.diseases[0]
for ti in range(560):
    sim.run_one_step()
    if ti in [199, 200, 201, 499, 500, 501, 545]:
        print(f'day {ti}: vaccinated fraction {d._vaccinated_fraction.raw[0]:.3f}, dose time {d._dose_time.raw[0]:.3f}, vaccine_efficacy {d.vaccine_efficacy.raw[0]:.3f}, box_end {d.vaccine_box_end.raw[0]:.3f}, prob_acquire {d._prob_acquire_indoor.raw[0]:.3f}')
