""" malariasim (v3): human SIS + mosquito ODE, one node, bednets with and without killing, and a vaccine """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
import malariasim as ms
from scen import *
from run_v3 import NodeCounts, wan

def campaign(name):
    t = 2000 + T_INTV/365
    def build(sim, eir_only=False):
        items = [ms.InterventionDistributorScheduled(name='seed', products=ms.Outbreak(), start=2000 + RM['seed_day']/365, coverage=RM['seed_cov'])]
        if name in ['itn', 'itn_nokill']:
            kill = (0.0, ITN['kill'][1], ITN['kill'][2]) if name == 'itn_nokill' else ITN['kill']
            net = ms.Bednet(repelling=wan(ITN['rep']), blocking=wan(ITN['block']), killing=wan(kill), usage=wan(ITN['usage']))
            items.append(ms.InterventionDistributorScheduled(name='itn', products=net, start=t, coverage=ITN['cov']))
        elif name == 'vax':
            vax = ms.MalariaVaccine(waning=ms.WaningEffect(initial_efficacy=VAX['eff'], box_duration=ss.days(VAX['box']), decay_time_constant=ss.days(VAX['tau'])))
            items.append(ms.InterventionDistributorScheduled(name='vax', products=vax, start=t, coverage=VAX['cov']))
        return ms.Campaign(sim, items)
    return build

class MosCounts(ss.Analyzer):
    def init_results(self):
        super().init_results()
        self.mos = np.zeros((self.t.npts, 3))
        self.eir = np.zeros(self.t.npts)
    def step(self):
        m = [c for c in self.sim.connectors.values() if isinstance(c, ms.MosquitoesODE)][0]
        self.mos[self.ti] = [m._Sm.sum(), m._Em.sum(), m._Im.sum()]
        self.eir[self.ti] = m.results.eir_annual[self.ti]

def make(name, seed):
    nodes = ms.NodeGrid(populations=[RM['pop']], emergence=[[RM['em']]*12], f_indoor=[F_INDOOR])
    human_pars = dict(infection_duration_years=ss.lognorm_ex(mean=DUR_INF/365, std=DUR_INF/365), prob_human_infected_on_bite=RM['b'], prob_mosquito_infected_on_bite=RM['c'])
    mos_pars = dict(bites_per_female_per_day=RM['a'], life_expectancy=ss.days(RM['life']), eip_days=ss.days(RM['eip']), f_indoor=F_INDOOR)
    return ms.Sim(human='sis', mosquito='ode', nodes=nodes, start=2000, stop=2002, dt=ss.days(1), human_pars=human_pars, mosquito_pars=mos_pars,
                  rand_seed=seed, verbose=0, analyzers=[NodeCounts(name='counts'), MosCounts(name='mos')], build_campaign=campaign(name))

if __name__ == '__main__':
    out = {}
    for name in RM_SCENARIOS:
        out[name] = []
        for seed in SEEDS:
            sim = make(name, seed)
            t0 = sc.tic(); sim.run(); el = sc.toc(t0, output=True)
            a = sim.analyzers.counts; m = sim.analyzers.mos
            out[name].append(dict(inf=a.inf.sum(1)[RM_REPORT].tolist(), mos=m.mos.sum(1)[RM_REPORT].tolist(), Iv=m.mos[RM_REPORT, 2].tolist(),
                                  eir=m.eir[RM_REPORT].tolist(), cum_y2=float(a.new[365:].sum()), time=el))
        f = lambda k: np.round(np.mean([o[k] for o in out[name]], axis=0), 1).tolist()
        print(f'{name:10s} n_inf {f("inf")}; mosquitoes {f("mos")}; Iv {f("Iv")}; EIR/yr {f("eir")}; cum y2 {np.mean([o["cum_y2"] for o in out[name]]):.0f}; t {np.mean([o["time"] for o in out[name]]):.2f}s', flush=True)
    sc.savejson('compare_v3_rm.json', out)
