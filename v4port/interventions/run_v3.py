""" malariasim (v3) intervention scenarios on human SIS + EIR-only mosquitoes, 2 nodes, several seeds """
import starsim as ss; print(ss.__version__, ss.__file__)
import sys, numpy as np, sciris as sc
import malariasim as ms
from scen import *

class NodeCounts(ss.Analyzer):
    """ Infected per node, and new infections, at every step """
    def init_results(self):
        super().init_results()
        self.inf = np.zeros((self.t.npts, len(POPS)))
        self.new = np.zeros(self.t.npts)
    def step(self):
        d = self.sim.diseases[0]
        ti = self.ti
        alive = self.sim.people.auids
        nid = np.asarray(d.node_id[alive], dtype=int)
        inf = np.asarray(d.infected[alive], dtype=bool)
        self.inf[ti] = np.bincount(nid[inf], minlength=len(POPS))
        self.new[ti] = int((d.ti_infected[alive] == ti).sum())

def wan(p):
    e, box, tau = p
    return ms.WaningEffect(initial_efficacy=e, box_duration=ss.days(box), decay_time_constant=ss.days(tau))

def campaign(name):
    t = 2000 + T_INTV/365
    def build(sim, eir_only=False):
        items = []
        if name in ['vax', 'vax_node0']:
            vax = ms.MalariaVaccine(waning=ms.WaningEffect(initial_efficacy=VAX['eff'], box_duration=ss.days(VAX['box']), decay_time_constant=ss.days(VAX['tau'])))
            elig = ms.HumanEligibility(node_ids=[0]) if name == 'vax_node0' else None
            items.append(ms.InterventionDistributorScheduled(name='vax', products=vax, start=t, eligibility=elig, coverage=VAX['cov']))
        elif name == 'boost':
            vax = ms.MalariaVaccine(waning=ms.WaningEffect(initial_efficacy=VAX['eff'], box_duration=ss.days(VAX['box']), decay_time_constant=ss.days(VAX['tau'])))
            for i, day in enumerate([BOOST['day1'], BOOST['day2']]):
                items.append(ms.InterventionDistributorScheduled(name=f'vax{i}', products=vax, start=2000 + day/365, coverage=VAX['cov']))
        elif name == 'mda':
            items.append(ms.InterventionDistributorScheduled(name='mda', products=ms.AntiMalarialDrug(mean_duration=ss.days(DRUG['mean'])), start=t, coverage=DRUG['cov']))
        elif name == 'itn':
            net = ms.Bednet(repelling=wan(ITN['rep']), blocking=wan(ITN['block']), killing=wan(ITN['kill']), usage=wan(ITN['usage']))
            items.append(ms.InterventionDistributorScheduled(name='itn', products=net, start=t, coverage=ITN['cov']))
        elif name == 'seed':
            items.append(ms.InterventionDistributorScheduled(name='seed', products=ms.Outbreak(), start=2000 + SEED['day']/365,
                         eligibility=ms.HumanEligibility(node_ids=[SEED['node']]), coverage=SEED['cov']))
        return ms.Campaign(sim, items)
    return build

def make(name, seed):
    eir = [0.0, 0.0] if name == 'seed' else EIR
    nodes = ms.NodeGrid(populations=POPS, eir=eir, f_indoor=[F_INDOOR]*len(POPS))
    human_pars = dict(infection_duration_years=ss.lognorm_ex(mean=DUR_INF/365, std=DUR_INF/365))
    return ms.Sim(human='sis', mosquito='eir_only', nodes=nodes, start=2000, stop=2002, dt=ss.days(1),
                  human_pars=human_pars, rand_seed=seed, verbose=0, analyzers=NodeCounts(name='counts'),
                  build_campaign=campaign(name))

if __name__ == '__main__':
    out = {}
    T = sc.timer()
    for name in SCENARIOS:
        out[name] = []
        for seed in SEEDS:
            sim = make(name, seed)
            t0 = sc.tic(); sim.run(); el = sc.toc(t0, output=True)
            a = sim.analyzers.counts
            out[name].append(dict(inf=a.inf.sum(1)[REPORT].tolist(), inf_nodes=a.inf[REPORT].tolist(), cum_y2=float(a.new[365:].sum()), cum=float(a.new.sum()), time=el))
        m = np.mean([o['inf'] for o in out[name]], axis=0); s = np.std([o['inf'] for o in out[name]], axis=0)
        mn = np.mean([o['inf_nodes'] for o in out[name]], axis=0)
        print(f'{name:10s} n_inf at {REPORT}: {np.round(m).astype(int).tolist()} (sd {np.round(s).astype(int).tolist()}); nodes end {np.round(mn[-1]).astype(int).tolist()}; nodes d395 {np.round(mn[2]).astype(int).tolist()}; cum y2 {np.mean([o["cum_y2"] for o in out[name]]):.0f}; t {np.mean([o["time"] for o in out[name]]):.2f}s')
    sc.savejson('compare_v3.json', out)
    print('total', T.toc(output=True))
